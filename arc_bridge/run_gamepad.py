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
VX_RATE_LIMIT_MIN_VEL = -1.0  # min m/s
VX_RATE_LIMIT_MAX_VEL = 1.0  # max m/s

WZ_RATE_LIMIT_MAX_ACC = 3.0 * np.pi / 2.0  # rad/s^2 acceleration limit on command tracking
WZ_RATE_LIMIT_MIN_VEL = -np.pi / 2.0  # min rad/s
WZ_RATE_LIMIT_MAX_VEL = np.pi / 2.0  # max rad/s

OVERRIDE_VX = 1.0
OVERRIDE_WZ = 1.0

def toggle_override_mode(btn_down, prev_btn_down, override_mode):
    if btn_down and not prev_btn_down:
        override_mode = not override_mode
        print(f"=> Override mode: {'ON' if override_mode else 'OFF'}")
    return override_mode


def main():
    parser = argparse.ArgumentParser(description="Standalone gamepad LCM publisher")
    parser.add_argument("--rate", type=float, default=1000.0, help="Publishing rate in Hz (default: 100)")
    parser.add_argument("--topic", type=str, default="gamepad_cmd", help="LCM topic name (default: gamepad_cmd)")
    parser.add_argument("--vel-scale-x", type=float, default=1.0, help="Max forward/backward velocity scale (default: 0.8)")
    parser.add_argument("--vel-scale-y", type=float, default=0.5, help="Max left/right velocity scale (default: 0.5)")
    parser.add_argument("--vel-scale-rot", type=float, default=np.pi / 2.0, help="Max rotation velocity scale (default: pi/2)")
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
    override_mode = False
    prev_btn_down = False
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

        override_mode = toggle_override_mode(buttons[2], prev_btn_down, override_mode)
        prev_btn_down = buttons[2]
        if override_mode:
            vx_raw = OVERRIDE_VX
            wz_raw = OVERRIDE_WZ

        if cmd[3]:
            vx_filtered = 0.0
            wz_filtered = 0.0
            vx_lpf.reset([0.0])
            wz_lpf.reset([0.0])
            vx_rate_limited = 0.0
            wz_rate_limited = 0.0
        elif VX_FILTER_MODE == "vx_lowpass":
            vx_filtered = float(vx_lpf.update(vx_raw, dt=dt)[0])
        elif VX_FILTER_MODE == "vx_rate_limit":
            vx_rate_limited += float(np.sign(vx_raw - vx_rate_limited) * VX_RATE_LIMIT_MAX_ACC * dt)
            vx_rate_limited = float(np.clip(vx_rate_limited, VX_RATE_LIMIT_MIN_VEL, VX_RATE_LIMIT_MAX_VEL))
            vx_filtered = vx_rate_limited
        else:
            vx_filtered = vx_raw

        if cmd[3]:
            pass
        elif WZ_FILTER_MODE == "wz_lowpass":
            wz_filtered = float(wz_lpf.update(wz_raw, dt=dt)[0])
        elif WZ_FILTER_MODE == "wz_rate_limit":
            wz_rate_limited += float(np.sign(wz_raw - wz_rate_limited) * WZ_RATE_LIMIT_MAX_ACC * dt)
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
