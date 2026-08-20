import time
import signal
import argparse
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np

from arc_bridge.config import Config
from arc_bridge.bridges import *


def signal_handler(signal, frame):
    print("\nCTRL-C received, exiting...")
    bridge.is_running = False


def simulate_mujoco():
    next_time = time.perf_counter()
    while viewer.is_running() and bridge.is_running:
        if args.block and not bridge.low_cmd_received:
            bridge.publish_low_state(bridge.topic_state)
        else:
            with viewer.lock():
                mujoco.mj_step(mj_model, mj_data)

            bridge.publish_gamepad_cmd()
            if not args.replay:
                bridge.publish_low_state(bridge.topic_state)
                bridge.update_motor_cmd()
                bridge.low_cmd_received = False
            else:
                # Publish to topic_state.upper() for real robot
                bridge.publish_low_state(bridge.topic_state.upper(), skip_common_state=True)

        # Wait to sync wall clock with simulation time
        next_time += mj_model.opt.timestep
        remaining = next_time - time.perf_counter()
        if remaining > 0:
            if args.busywait:
                while time.perf_counter() < next_time:
                    time.sleep(0)  # Yield to other threads
            else:
                time.sleep(remaining)
        else:
            # Over-ran, skip sleep to catch up
            next_time = time.perf_counter()

    print("Simulation thread exited")


def main():
    global mj_data, mj_model, viewer, bridge, args
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", action="store_true", help="block the simulation thread if no control is received")
    parser.add_argument("--track", action="store_true", help="make camera track the robot's motion")
    parser.add_argument("--replay", action="store_true", help="replay state trajectory from LCM")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--busywait", action="store_true", help="busywait in simulation thread")
    parser.add_argument("--use_gamepad", action="store_true", help="use gamepad to control the robot")
    parser.add_argument("--control_delay", type=float, default=0.0, help="control delay in milliseconds before commands are applied")
    parser.add_argument("--sensor_delay", type=float, default=0.0, help="sensor delay in milliseconds before states are published")
    parser.add_argument("--lcm_cmd_online_jitter_time_ms", type=float, default=10.0, help="LCM command watchdog; just online wait time (unstable time); wait time after offline to consider online (ms)")
    parser.add_argument("--lcm_cmd_offline_time_ms", type=float, default=200.0, help="LCM command watchdog; offline timeout; time to consider offline if no update received (ms)")
    parser.add_argument("--disable_daemon", action="store_true", help="flag to disable the watchdog")
    args = parser.parse_args()

    # Select robot type
    for i, r_type in enumerate(Config.valid_robot_types):
        print(f"{i}: {r_type}")

    robot_type_idx = int(input("Please select the robot type: "))
    robot_type = Config.valid_robot_types[robot_type_idx]

    robot_config = Config(
        robot_type,
        control_delay=args.control_delay,
        sensor_delay=args.sensor_delay,
        lcm_cmd_online_jitter_time_ms=args.lcm_cmd_online_jitter_time_ms,
        lcm_cmd_offline_time_ms=args.lcm_cmd_offline_time_ms,
        launch_args=args,
    )

    # Initialize Mujoco
    mj_model = mujoco.MjModel.from_xml_path(robot_config.robot_xml_path)
    mj_data = mujoco.MjData(mj_model)

    # Modify MjOption
    mj_model.opt.timestep = Config.dt_sim

    if args.replay:
        # Disable gravity and all constraints (e.g. contact, friction ...)
        mj_model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_GRAVITY
    elif args.debug:
        mj_model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_GRAVITY

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    if args.track:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0

    # Enable visualization flags
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # Initialize bridge
    try:
        bridge_name = "".join([s.capitalize() for s in robot_type.split("_")]) + "Bridge"
        bridge = eval(bridge_name)(mj_model, mj_data, robot_config)
    except NameError as e:
        bridge = Lcm2MujocoBridge(mj_model, mj_data, robot_config)
        print(f"=> Error: {e}")
        print(f"=> Constructing {bridge_name} failed. Using default bridge.")

    if args.replay:
        # Subscribe to topic_state from real robot to parse common states
        bridge.register_low_state_subscriber(bridge.topic_state)
        # Subscribe to topic_cmd.upper() from upper level controller to prevent wrong command sources
        bridge.register_low_cmd_subscriber(bridge.topic_cmd.upper())
    else:
        bridge.register_low_cmd_subscriber(bridge.topic_cmd)

    # Handle SIGINT to exit gracefully
    signal.signal(signal.SIGINT, signal_handler)

    # Reset data keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_step(mj_model, mj_data)

    # Start threads
    bridge.start_lcm_thread()
    if args.use_gamepad:
        bridge.start_gamepad_thread()

    sim_thread = Thread(target=simulate_mujoco, daemon=True)
    sim_thread.start()

    # Ghost arm: a second MjData on the same model, posed at the joint angles the
    # upper level asks for. Only the visual meshes (group 2) are drawn, so the
    # frame-axis geoms (group 1) and collision geoms (group 3) stay out of it.
    ghost_data = mujoco.MjData(mj_model)
    ghost_opt = mujoco.MjvOption()
    ghost_opt.geomgroup[:] = 0
    ghost_opt.geomgroup[2] = 1
    ghost_pert = mujoco.MjvPerturb()

    while viewer.is_running() and bridge.is_running:
        n_geom = 0
        viewer.user_scn.ngeom = 0

        # Add geom of estimated position and velocity
        if bridge.vis_se:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[n_geom],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=bridge.vis_box_size,
                pos=bridge.vis_pos_est,
                mat=bridge.vis_R_body.flatten(),
                rgba=[1, 0, 0, 0.3]
            )
            n_geom += 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[n_geom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.zeros(9),
                rgba=[0, 0, 1, 1]
            )
            mujoco.mjv_connector( # scn, type, width, from, to
                viewer.user_scn.geoms[n_geom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.02,
                bridge.vis_pos_est,
                bridge.vis_pos_est + bridge.vis_vel_est*0.5)
            n_geom += 1

        # Add geom of the end-effector pose reported by the upper-level FK:
        # a translucent box plus an RGB axis triad, so a mismatch against the
        # real flange is visible at a glance.
        if bridge.vis_fk:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[n_geom],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=bridge.vis_fk_box_size,
                pos=bridge.vis_fk_pos,
                mat=bridge.vis_fk_R.flatten(),
                rgba=[1, 0, 0, 0.3]
            )
            n_geom += 1
            for axis in range(3):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[n_geom],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=[float(axis == 0), float(axis == 1), float(axis == 2), 1]
                )
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[n_geom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.006,
                    bridge.vis_fk_pos,
                    bridge.vis_fk_pos + bridge.vis_fk_R[:, axis]*0.06)
                n_geom += 1

        viewer.user_scn.ngeom = n_geom

        # Ghost arm. mjv_addGeoms appends starting at scn.ngeom and advances it,
        # so this has to run after ngeom is set above.
        if bridge.vis_ghost:
            ghost_data.qpos[:6] = bridge.vis_ghost_q
            mujoco.mj_kinematics(mj_model, ghost_data)
            ghost_start = viewer.user_scn.ngeom
            mujoco.mjv_addGeoms(
                mj_model, ghost_data, ghost_opt, ghost_pert,
                mujoco.mjtCatBit.mjCAT_DYNAMIC, viewer.user_scn)
            for i in range(ghost_start, viewer.user_scn.ngeom):
                viewer.user_scn.geoms[i].rgba[:] = [0.2, 0.6, 1.0, 0.35]

        with viewer.lock():
            # Turn state_only on to make sync() really fast.
            # No mj_model modification on the fly is allowed instead.
            viewer.sync(state_only=True) # state_only is introduced in mujoco 3.3.4

        time.sleep(Config.dt_viewer)

    # Wait for threads to exit
    viewer.close()
    sim_thread.join()
    bridge.stop_lcm_thread()


if __name__ == "__main__":
    main()
