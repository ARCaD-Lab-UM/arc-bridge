import lcm
import time
import signal
import argparse
import numpy as np

from arc_bridge.utils import Gamepad, FirstOrderLowPassTD
from arc_bridge.lcm_msgs import gamepad_cmd_t


VX_FILTER_MODE = "vx_rate_limit"  # "vx_lowpass" "vx_rate_limit" "raw"
WZ_FILTER_MODE = "wz_rate_limit"  # "wz_lowpass" "wz_rate_limit" "raw"

VX_LPF_TAU_SEC = 0.12
WZ_LPF_TAU_SEC = 0.12

VX_RATE_LIMIT_MAX_ACC = 3.0  # m/s^2 acceleration limit on command tracking; 3.5=aggressive
VX_RATE_LIMIT_MIN_VEL = -10  # min m/s
VX_RATE_LIMIT_MAX_VEL = 10  # max m/s
VX_RATE_LIMIT_EPSILON = 0.001  # m/s deadband; |vx_raw - vx_rate_limited| below this freezes; -1 to disable

WZ_RATE_LIMIT_MAX_ACC = 3.0 * np.pi / 2.0  # rad/s^2 acceleration limit on command tracking
WZ_RATE_LIMIT_MIN_VEL = -np.pi * 2.0  # min rad/s
WZ_RATE_LIMIT_MAX_VEL = np.pi * 2.0  # max rad/s
WZ_RATE_LIMIT_EPSILON = 0.001  # rad/s deadband; |wz_raw - wz_rate_limited| below this freezes; -1 to disable

# For different modes
VX_SCALE_SPIN = 0.1
ROT_SCALE_SPIN = 1.5 * np.pi

VX_SCALE_FAST = 3.0
ROT_SCALE_FAST = 0.5

VX_SCALE_NORMAL = 2.0
ROT_SCALE_NORMAL = 1.0


def main():
    parser = argparse.ArgumentParser(description="Standalone gamepad LCM publisher")
    parser.add_argument("--rate", type=float, default=1000.0, help="Publishing rate in Hz (default: 100)")
    parser.add_argument("--topic", type=str, default="gamepad_cmd", help="LCM topic name (default: gamepad_cmd)")
    parser.add_argument("--vel-scale-x", type=float, default=1.5, help="Max forward/backward velocity scale (default: 0.8)") # 1.5;  1.5 or 2 or 3.0 in last video
    parser.add_argument("--vel-scale-y", type=float, default=0.5, help="Max left/right velocity scale (default: 0.5)")
    parser.add_argument("--vel-scale-rot", type=float, default=np.pi, help="Max rotation velocity scale (default: pi/2)") # 1.5 * np.pi;  1
    parser.add_argument("--scale-pitch", type=float, default=np.pi / 2.0, help="Pitch scale (default: pi/2)")
    parser.add_argument("--triggers-scale", type=float, default=1.0, help="Trigger axis scale (default: 1.0)")
    parser.add_argument("--ttl", type=int, default=None, help="Multicast time-to-live value (default: None, uses LCM default)")
    args = parser.parse_args()

    # Initialize LCM
    if args.ttl is not None:
        udp_multicast_group = f"udpm://239.255.76.67:7667?ttl={args.ttl}"
        lc = lcm.LCM(udp_multicast_group)
    else:
        udp_multicast_group = None
        lc = lcm.LCM()


    # Initialize gamepad
    try:
        gamepad = Gamepad(
            vel_scale_x=args.vel_scale_x,
            vel_scale_y=args.vel_scale_y,
            vel_scale_rot=args.vel_scale_rot,
            scale_pitch=args.scale_pitch,
            triggers_scale=args.triggers_scale,
        )
        print("=> Gamepad found")
    except Exception as e:
        print(f"=> No gamepad found: {e}")
        return

    gamepad_cmd = gamepad_cmd_t()
    is_running = True

    def signal_handler(sig, frame):
        nonlocal is_running
        print("\nCTRL-C received, exiting...")
        is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    dt = 1.0 / args.rate
    vx_lpf = FirstOrderLowPassTD(tau_sec=VX_LPF_TAU_SEC, frame_dt=dt, dim=1, init=[0.0])
    wz_lpf = FirstOrderLowPassTD(tau_sec=WZ_LPF_TAU_SEC, frame_dt=dt, dim=1, init=[0.0])
    vx_rate_limited = 0.0
    wz_rate_limited = 0.0
    prev_mode = None
    print(f"=> LCM URL: {udp_multicast_group or 'default'}")
    print(f"=> Publishing gamepad commands on '{args.topic}' at {args.rate} Hz")

    while is_running:
        cmd = gamepad.get_command()
        pitch = gamepad.get_pitch()
        params = gamepad.get_params()
        buttons = gamepad.get_buttons()
        lbrb = gamepad.get_lbrb()
        ljrj = gamepad.get_ljrj()
        lt, rt = gamepad.get_triggers()
        vx_raw = cmd[0]
        wz_raw = cmd[2]

        if lbrb[0]:
            mode = "spin"
        elif lbrb[1]:
            mode = "fast"
        else:
            mode = "normal"
        if mode != prev_mode:
            if mode == "spin":
                gamepad.set_vel_scale_x(VX_SCALE_SPIN)
                gamepad.set_vel_scale_rot(ROT_SCALE_SPIN)
            elif mode == "fast":
                gamepad.set_vel_scale_x(VX_SCALE_FAST)
                gamepad.set_vel_scale_rot(ROT_SCALE_FAST)
            else:
                gamepad.set_vel_scale_x(VX_SCALE_NORMAL)
                gamepad.set_vel_scale_rot(ROT_SCALE_NORMAL)
            prev_mode = mode

        if cmd[3]:
            # e-stop pressed
            vx_filtered = 0.0
            wz_filtered = 0.0
            vx_lpf.reset([0.0])
            wz_lpf.reset([0.0])
            vx_rate_limited = 0.0
            wz_rate_limited = 0.0
        elif VX_FILTER_MODE == "vx_lowpass":
            vx_filtered = float(vx_lpf.update(vx_raw, dt=dt)[0])
        elif VX_FILTER_MODE == "vx_rate_limit":
            vx_diff = vx_raw - vx_rate_limited
            vx_sign = 0.0 if abs(vx_diff) < VX_RATE_LIMIT_EPSILON else np.sign(vx_diff)
            vx_rate_limited += float(vx_sign * VX_RATE_LIMIT_MAX_ACC * dt)
            vx_rate_limited = float(np.clip(vx_rate_limited, VX_RATE_LIMIT_MIN_VEL, VX_RATE_LIMIT_MAX_VEL))
            vx_filtered = vx_rate_limited
        else:
            vx_filtered = vx_raw

        if cmd[3]:
            pass
        elif WZ_FILTER_MODE == "wz_lowpass":
            wz_filtered = float(wz_lpf.update(wz_raw, dt=dt)[0])
        elif WZ_FILTER_MODE == "wz_rate_limit":
            wz_diff = wz_raw - wz_rate_limited
            wz_sign = 0.0 if abs(wz_diff) < WZ_RATE_LIMIT_EPSILON else np.sign(wz_diff)
            wz_rate_limited += float(wz_sign * WZ_RATE_LIMIT_MAX_ACC * dt)
            wz_rate_limited = float(np.clip(wz_rate_limited, WZ_RATE_LIMIT_MIN_VEL, WZ_RATE_LIMIT_MAX_VEL))
            wz_filtered = wz_rate_limited
        else:
            wz_filtered = wz_raw

        gamepad_cmd.timestamp = time.time_ns()
        gamepad_cmd.vx = vx_filtered
        gamepad_cmd.vy = cmd[1]
        gamepad_cmd.wz = wz_filtered
        gamepad_cmd.e_stop = cmd[3]
        gamepad_cmd.pitch = pitch
        gamepad_cmd.params[:] = params[:]
        gamepad_cmd.btn_up = buttons[0]
        gamepad_cmd.btn_left = buttons[1]
        gamepad_cmd.btn_down = buttons[2]
        gamepad_cmd.btn_right = buttons[3]
        gamepad_cmd.btn_lb = lbrb[0]
        gamepad_cmd.btn_rb = lbrb[1]
        gamepad_cmd.btn_lstick = ljrj[0]
        gamepad_cmd.btn_rstick = ljrj[1]
        gamepad_cmd.lt = lt
        gamepad_cmd.rt = rt

        lc.publish(args.topic, gamepad_cmd.encode())
        time.sleep(dt)

    gamepad.stop()
    print("Gamepad publisher stopped")


if __name__ == "__main__":
    main()
