from pathlib import Path


class Config:
    dt_sim = 0.001     # Dynamics update rate
    dt_viewer = 0.02   # Viewer update rate

    package_root = Path(__file__).parent.parent
    print("Package root:", package_root)

    asset_root = package_root / "robot_assets"

    robot_path_dict = {
        # "hopper":           "Hopper/hopper_scene.xml",
        "sliding":          "Tron1Wheeled/xml/robot_object.xml",
        "tron1_wheeled":    "Tron1Wheeled/xml/robot.xml",
        "hopper":           "Hopper_v2/hopper_scene.xml",
        "tron1_pointfoot":  "Tron1Pointfoot/xml/robot.xml",
        "tron1_wheeled":    "Tron1Wheeled/xml/robot.xml", # "Tron1Wheeled/xml/robot_lock_abad.xml", # "Tron1Wheeled/xml/robot_object.xml",
        "tron1_linefoot":   "Tron1Linefoot/xml/robot.xml",
        "pendulum":         "Pendulum/pendulum_scene.xml",
        "biped_pointfoot":  "BipedPointfoot/biped_pointfoot_scene.xml",
        "biped_linefoot":   "BipedLinefoot/biped_linefoot_scene.xml",
        # "arm2link":         "Arm2Link/arm2link_sensing.xml",
    }
    valid_robot_types = list(robot_path_dict.keys())

    def __init__(self, robot_type, control_delay: float = 0.0, sensor_delay: float = 0.0, launch_args=None):
        self.launch_args = launch_args
        self.robot_type = robot_type
        if self.robot_type not in self.valid_robot_types:
            raise ValueError(f"Invalid robot type: {self.robot_type}. Valid robot types are: {self.valid_robot_types}")

        self.robot_state_topic = self.robot_type + "_state"
        self.robot_cmd_topic = self.robot_type + "_control"

        self.robot_xml_path = str(self.asset_root / self.robot_path_dict[self.robot_type])

        if control_delay < 0 or sensor_delay < 0:
            raise ValueError("control_delay and sensor_delay must be non-negative")
        self.control_delay = float(control_delay) # default 0.0
        self.sensor_delay = float(sensor_delay)
