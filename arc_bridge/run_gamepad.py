import lcm
import time
import signal
import argparse
import numpy as np

from arc_bridge.utils import Gamepad
from arc_bridge.lcm_msgs import gamepad_cmd_t


def main():
    parser = argparse.ArgumentParser(description="Standalone gamepad LCM publisher")
    parser.add_argument("--rate", type=float, default=100.0, help="Publishing rate in Hz (default: 100)")
    parser.add_argument("--topic", type=str, default="gamepad_cmd", help="LCM topic name (default: gamepad_cmd)")
    parser.add_argument("--vel-scale-x", type=float, default=2.0, help="Max forward/backward velocity scale (default: 2.0)")
    parser.add_argument("--vel-scale-y", type=float, default=0.5, help="Max left/right velocity scale (default: 0.5)")
    parser.add_argument("--vel-scale-rot", type=float, default=np.pi, help="Max rotation velocity scale (default: pi)")
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
    print(f"=> LCM URL: {udp_multicast_group or 'default'}")
    print(f"=> Publishing gamepad commands on '{args.topic}' at {args.rate} Hz")

    while is_running:
        axis = gamepad.get_axis()
        params = gamepad.get_params()
        buttons = gamepad.get_buttons()
        lbrb = gamepad.get_lbrb()
        ljrj = gamepad.get_ljrj()
        lt, rt = gamepad.get_triggers()

        gamepad_cmd.timestamp = time.time_ns()
        gamepad_cmd.axis[:] = axis  # [L West, L North, R West, R North] for positive directions
        gamepad_cmd.e_stop = gamepad.get_estop()
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
