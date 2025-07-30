import mujoco
import mujoco.viewer

import time
from threading import Thread
import threading
import argparse

import numpy as np

from arc_bridge.config import Config
from arc_bridge.bridges import *

def SimulationThread():
    global mj_data, mj_model, viewer, bridge, locker, args
    while viewer.is_running():
        step_start = time.perf_counter()
        if args.block and not bridge.low_cmd_received:
            bridge.publish_low_state(bridge.topic_state)
        else:
            locker.acquire()
            mujoco.mj_step(mj_model, mj_data)
            locker.release()
            bridge.publish_gamepad_cmd()
            if not args.replay:
                bridge.publish_low_state(bridge.topic_state)
                bridge.update_motor_cmd()
                bridge.low_cmd_received = False
            else:
                # Publish to topic_state.upper() for real robot
                bridge.publish_low_state(bridge.topic_state.upper(), skip_common_state=True)

        #* time.sleep() can be inaccurate depending on OS timer
        # Instead, use busy-waiting (spinlock) for short delay to ensure precision
        while (time.perf_counter() - step_start) < mj_model.opt.timestep:
            pass

    print("Simulation thread exited")


def ViewerThread():
    while viewer.is_running():
        locker.acquire()
        if bridge.vis_se:
            # Add geom of estimated position and velocity
            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=bridge.vis_box_size,
                pos=bridge.vis_pos_est,
                mat=bridge.vis_R_body.flatten(),
                rgba=[1, 0, 0, 0.3]
            )
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.zeros(9),
                rgba=[0, 0, 1, 1]
            )
            mujoco.mjv_connector( # scn, type, width, from, to
                viewer.user_scn.geoms[1],
                mujoco.mjtGeom.mjGEOM_ARROW, 
                0.02,
                bridge.vis_pos_est, 
                bridge.vis_pos_est + bridge.vis_vel_est*2)
            viewer.user_scn.ngeom = 2
        viewer.sync()
        locker.release()
        time.sleep(Config.dt_viewer)

    print("Viewer thread exited")


def main():
    global mj_data, mj_model, viewer, bridge, locker, args
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", action="store_true", help="block the simulation thread if no control is received")
    parser.add_argument("--track", action="store_true", help="make camera track the robot's motion")
    parser.add_argument("--replay", action="store_true", help="replay state trajectory from LCM")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    # Select robot type
    for i, r_type in enumerate(Config.valid_robot_types):
        print(f"{i}: {r_type}")

    robot_type_idx = int(input("Please select the robot type: "))
    robot_type = Config.valid_robot_types[robot_type_idx]

    robot_config = Config(robot_type)

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

    # Separate simulation and visualization threads
    locker = threading.Lock()

    # Reset data keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_step(mj_model, mj_data)

    # Start threads
    bridge.start_lcm_thread()

    viewer_thread = Thread(target=ViewerThread, daemon=True)
    sim_thread = Thread(target=SimulationThread, daemon=True)

    viewer_thread.start()
    sim_thread.start()

    # Wait for threads to exit after viewer is closed
    viewer_thread.join()
    sim_thread.join()
    bridge.stop_lcm_thread()

if __name__ == "__main__":
    main()
