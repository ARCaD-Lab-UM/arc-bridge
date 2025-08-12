import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys

# Mock mujoco module before any imports that might use it
mock_mujoco = Mock()
# Add common mujoco functions that the bridge uses
mock_mujoco.mj_id2name = Mock()
mock_mujoco.mjtObj = Mock()
mock_mujoco.mjtObj.mjOBJ_SENSOR = 6  # Sensor object type
sys.modules['mujoco'] = mock_mujoco

from arc_bridge.bridges.lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs.hopper_control_t import hopper_control_t
from arc_bridge.lcm_msgs.hopper_state_t import hopper_state_t
from arc_bridge.config import Config


class TestBridgeFunctionality:
    """Test bridge functionality with mocked MuJoCo dependencies"""

    @pytest.fixture
    def mock_mujoco_model(self):
        """Create a mock MuJoCo model with realistic properties"""
        mock_model = Mock()
        mock_model.nu = 2  # 2 actuators
        mock_model.nq = 9  # 9 DOF (7 floating base + 2 joints)
        mock_model.nsensor = 30  # Total sensors
        mock_model.opt.timestep = 0.001
        mock_model.actuator_ctrlrange = np.array([[-10, 10]] * 2)  # 2 actuators
        mock_model.sensor_noise = np.ones(30) * 0.01  # Small noise values
        mock_model.sensor_dim = [1] * 30  # Each sensor has dimension 1

        # Add attributes needed for print_scene_info
        mock_model.nbody = 5  # Number of bodies
        mock_model.njnt = 8  # Number of joints (6 floating base + 2 joints)
        mock_model.nact = 2  # Number of actuators
        mock_model.body_mass = [1.0] * 5
        mock_model.body_inertia = [[0.1, 0.1, 0.1]] * 5

        return mock_model

    @pytest.fixture
    def mock_mujoco_data(self):
        """Create mock MuJoCo data with sensor readings"""
        mock_data = Mock()
        # Mock sensor data: sensor configurations
        mock_data.sensordata = np.random.randn(23) * 10
        mock_data.sensordata[6:10] = [1, 0, 0, 0]  # IMU quaternion
        mock_data.ctrl = np.zeros(2)
        mock_data.time = 1.0
        mock_data.qvel = np.random.randn(8)  # 6 floating base + 2 joints
        return mock_data

    @pytest.fixture
    def mock_config(self):
        """Create a hopper config using the Config class"""
        return Config("hopper")

    @pytest.fixture
    def hopper_state_msg(self):
        """Create a reusable hopper state message"""
        hopper_state = hopper_state_t()
        hopper_state.qj_pos = [0.0] * 2
        hopper_state.qj_vel = [0.0] * 2
        hopper_state.qj_tau = [0.0] * 2
        hopper_state.position = [0.0, 0.0, 1.0]
        hopper_state.velocity = [0.0, 0.0, 0.0]
        hopper_state.rpy = [0.0, 0.0, 0.0]
        hopper_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        hopper_state.omega = [0.0, 0.0, 0.0]
        hopper_state.acceleration = [0.0, 0.0, -9.81]
        hopper_state.foot_force = [0.0]
        hopper_state.timestamp = 0
        return hopper_state

    @pytest.fixture
    def hopper_control_msg(self):
        """Create a reusable hopper control message"""
        hopper_control = hopper_control_t()
        hopper_control.qj_pos = [0.0] * 2
        hopper_control.qj_vel = [0.0] * 2
        hopper_control.qj_tau = [0.0] * 2
        hopper_control.kp = [100.0] * 2
        hopper_control.kd = [10.0] * 2
        hopper_control.contact = True
        hopper_control.reset_se = False
        return hopper_control

    @pytest.fixture
    @patch("arc_bridge.bridges.lcm2mujuco_bridge.lcm.LCM")
    @patch("arc_bridge.bridges.lcm2mujuco_bridge.Gamepad")
    def mock_bridge(
        self,
        mock_gamepad_class,
        mock_lcm_class,
        mock_mujoco_model,
        mock_mujoco_data,
        mock_config,
        hopper_state_msg,
        hopper_control_msg,
    ):
        """Create bridge with mocked dependencies"""

        # Mock mujoco.mj_id2name to return realistic sensor names
        def mock_mj_id2name(model, obj_type, sensor_id):
            sensor_names = {
                6: "imu_quat",  # IMU quaternion sensor
                16: "frame_pos",  # Frame position sensor
                22: "foot_force",  # Foot force sensor
            }
            return sensor_names.get(sensor_id, f"sensor_{sensor_id}")

        mock_mujoco.mj_id2name.side_effect = mock_mj_id2name

        # Mock gamepad to raise exception (no gamepad found)
        mock_gamepad_class.side_effect = Exception("No gamepad found")

        # Use real LCM message classes
        with patch("arc_bridge.bridges.lcm2mujuco_bridge.eval") as mock_eval:

            def eval_side_effect(expr):
                if "state" in expr:
                    return hopper_state_t
                else:
                    return hopper_control_t

            mock_eval.side_effect = eval_side_effect

            bridge = Lcm2MujocoBridge(mock_mujoco_model, mock_mujoco_data, mock_config)
            bridge.low_state = hopper_state_msg
            bridge.low_cmd = hopper_control_msg

            return bridge

    def test_bridge_initialization(self, mock_bridge, mock_mujoco_model):
        """Test that bridge initializes correctly with mocked dependencies"""
        assert mock_bridge.mj_model == mock_mujoco_model
        assert mock_bridge.num_motor == 2
        assert mock_bridge.num_body_state == 7  # nq - nu = 9 - 2 = 7
        assert mock_bridge.num_joint_state == 2
        assert mock_bridge.dim_motor_sensor == 6  # 3 * 2 motors
        assert len(mock_bridge.joint_offsets) == 2
        assert mock_bridge.dt == 0.001

    def test_sensor_detection(self, mock_bridge):
        """Test sensor detection logic"""
        assert mock_bridge.have_imu == True
        assert mock_bridge.have_frame_sensor == True
        assert mock_bridge.have_foot_sensor == True
        assert mock_bridge.num_foot_sensor == 1

    def test_no_gamepad_handling(self, mock_bridge):
        """Test that bridge handles missing gamepad gracefully"""
        assert mock_bridge.gamepad is None
        # Should not raise exception when publishing gamepad commands
        mock_bridge.publish_gamepad_cmd()

    def test_parse_common_low_state(self, mock_bridge):
        """Test parsing of common sensor data"""

        # Set up sensor data
        mock_bridge.mj_data.sensordata[0:2] = [0.1, 0.2]  # Joint positions
        mock_bridge.mj_data.sensordata[2:4] = [1.1, 1.2]  # Joint velocities
        mock_bridge.mj_data.sensordata[4:6] = [2.1, 2.2]  # Joint torques

        result = mock_bridge.parse_common_low_state()

        assert result == 0  # Success
        # Check that joint values are close (allowing for added noise)
        assert abs(mock_bridge.low_state.qj_pos[0] - 0.1) < 0.1
        assert abs(mock_bridge.low_state.qj_vel[0] - 1.1) < 0.1
        assert abs(mock_bridge.low_state.qj_tau[0] - 2.1) < 0.1

    def test_lcm_cmd_handler(self, mock_bridge, hopper_control_msg):
        """Test LCM command handler with a control message"""
        # Use the fixture control message and modify as needed
        hopper_cmd = hopper_control_msg
        hopper_cmd.qj_pos = [0.5] * 2
        hopper_cmd.qj_vel = [0.1] * 2
        hopper_cmd.qj_tau = [1.0] * 2

        # Encode the message to bytes
        encoded_data = hopper_cmd.encode()

        with patch("builtins.eval", return_value=hopper_control_t):
            mock_bridge.lcm_cmd_handler("hopper_control", encoded_data)

            assert mock_bridge.low_cmd_received == True
            # Verify the decoded message has the expected values
            assert len(mock_bridge.low_cmd.qj_pos) == 2
            assert len(mock_bridge.low_cmd.qj_vel) == 2
            assert len(mock_bridge.low_cmd.qj_tau) == 2
            assert mock_bridge.low_cmd.qj_pos[0] == 0.5
            assert mock_bridge.low_cmd.contact == True

    def test_lcm_cmd_handler_with_none_data(self, mock_bridge):
        """Test command handler when mj_data is None"""
        original_received = mock_bridge.low_cmd_received
        mock_bridge.mj_data = None

        mock_bridge.lcm_cmd_handler("test_channel", b"data")

        # Should return early without processing
        assert mock_bridge.low_cmd_received == original_received

    def test_update_motor_cmd(self, mock_bridge):
        """Test motor command update with PD control"""
        # Set up test data with smaller gains to avoid clipping
        mock_bridge.low_cmd.qj_tau = [1.0] * 2
        mock_bridge.low_cmd.kp = [50.0] * 2  # Reduced gain
        mock_bridge.low_cmd.kd = [5.0] * 2  # Reduced gain
        mock_bridge.low_cmd.qj_pos = [0.5] * 2
        mock_bridge.low_cmd.qj_vel = [0.1] * 2
        mock_bridge.low_state.qj_pos = [0.4] * 2  # 0.1 position error
        mock_bridge.low_state.qj_vel = [0.0] * 2  # 0.1 velocity error

        mock_bridge.update_motor_cmd()

        # Check that control commands are calculated correctly
        # Expected: tau_cmd + kp*pos_error + kd*vel_error = 1.0 + 50*0.1 + 5*0.1 = 6.5
        expected_torque = 6.5
        for i in range(2):
            assert mock_bridge.mj_data.ctrl[i] == pytest.approx(
                expected_torque, abs=0.01
            )

    def test_update_motor_cmd_with_limits(self, mock_bridge):
        """Test motor command update with torque limits"""
        # Set up command that exceeds limits
        mock_bridge.low_cmd.qj_tau = [5.0] * 2
        mock_bridge.low_cmd.kp = [1000.0] * 2  # High gain
        mock_bridge.low_cmd.kd = [100.0] * 2
        mock_bridge.low_cmd.qj_pos = [1.0] * 2
        mock_bridge.low_cmd.qj_vel = [0.0] * 2
        mock_bridge.low_state.qj_pos = [0.0] * 2  # Large position error
        mock_bridge.low_state.qj_vel = [0.0] * 2

        mock_bridge.update_motor_cmd()

        # Should be clipped to actuator limits [-10, 10]
        for i in range(2):
            assert -10 <= mock_bridge.mj_data.ctrl[i] <= 10

    def test_publish_low_state_skip_common(self, mock_bridge):
        """Test state publishing with skip_common_state=True"""
        with patch.object(mock_bridge, "parse_common_low_state") as mock_parse_common:
            with patch.object(
                mock_bridge, "parse_robot_specific_low_state"
            ) as mock_parse_specific:
                mock_bridge.publish_low_state(skip_common_state=True)

                mock_parse_common.assert_not_called()
                mock_parse_specific.assert_called_once()

    def test_publish_low_state_with_custom_topic(self, mock_bridge):
        """Test state publishing to custom topic"""
        custom_topic = "CUSTOM_STATE"

        with patch.object(mock_bridge, "parse_robot_specific_low_state"):
            mock_bridge.publish_low_state(topic=custom_topic)

            mock_bridge.lc.publish.assert_called_with(
                custom_topic, mock_bridge.low_state.encode()
            )

    def test_lcm_subscriber_registration(self, mock_bridge):
        """Test LCM subscriber registration"""
        # Test command subscriber registration
        mock_bridge.register_low_cmd_subscriber()
        mock_bridge.lc.subscribe.assert_called()

        # Test state subscriber registration
        mock_bridge.register_low_state_subscriber()
        assert mock_bridge.lc.subscribe.call_count == 2

    def test_lcm_subscriber_with_custom_topic(self, mock_bridge):
        """Test subscriber registration with custom topics"""
        custom_topic = "CUSTOM_TOPIC"

        mock_bridge.register_low_cmd_subscriber(topic=custom_topic)

        # Verify subscribe was called with custom topic
        calls = mock_bridge.lc.subscribe.call_args_list
        assert any(call[0][0] == custom_topic for call in calls)

    @patch("arc_bridge.bridges.lcm2mujuco_bridge.Thread")
    def test_lcm_thread_lifecycle(self, mock_thread_class, mock_bridge):
        """Test LCM thread start and stop"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        # Test thread start
        mock_bridge.start_lcm_thread()
        assert mock_bridge.is_running == True
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()

        # Test thread stop
        mock_bridge.stop_lcm_thread()
        assert mock_bridge.is_running == False
        mock_thread.join.assert_called_once()

    def test_parse_common_low_state_error_handling(self, mock_bridge):
        """Test error handling in parse_common_low_state"""
        # Test with None mj_data
        mock_bridge.mj_data = None
        result = mock_bridge.parse_common_low_state()
        assert result == -1

    def test_bridge_without_sensors(
        self, mock_mujoco_model, mock_mujoco_data, mock_config, hopper_control_msg
    ):
        """Test bridge initialization with no special sensors"""
        # Reset mj_id2name to return no special sensor names
        mock_mujoco.mj_id2name.side_effect = lambda model, obj_type, sensor_id: f"generic_sensor_{sensor_id}"
        
        bridge = Lcm2MujocoBridge(
            mock_mujoco_model, mock_mujoco_data, mock_config
        )

        assert bridge.have_imu == False
        assert bridge.have_frame_sensor == False
        assert bridge.have_foot_sensor == False
        assert bridge.num_foot_sensor == 0

    def test_joint_offsets_functionality(self, mock_bridge):
        """Test joint offset functionality"""
        # Test that joint offsets are applied in sensor parsing
        mock_bridge.joint_offsets = np.array([0.1, 0.2])
        mock_bridge.mj_data.sensordata[0:2] = [1.0] * 2  # All joint positions = 1.0

        mock_bridge.parse_common_low_state()

        # Check that offsets are applied
        for i in range(2):
            expected_pos = 1.0 + mock_bridge.joint_offsets[i]
            # Allow some tolerance for noise
            assert abs(mock_bridge.low_state.qj_pos[i] - expected_pos) < 0.1

    def test_full_message_processing_cycle(self, mock_bridge, hopper_control_msg):
        """Test complete message processing from command to state publication"""
        # Use the fixture control message and modify as needed
        hopper_cmd = hopper_control_msg
        hopper_cmd.qj_pos = [0.5] * 2
        hopper_cmd.qj_vel = [0.1] * 2
        hopper_cmd.qj_tau = [1.0] * 2

        # Encode the message to bytes
        encoded_cmd_data = hopper_cmd.encode()

        with patch("builtins.eval", return_value=hopper_control_t):
            # Process command using the correct channel name that matches bridge topic_cmd
            mock_bridge.lcm_cmd_handler(mock_bridge.topic_cmd, encoded_cmd_data)

            # Verify command was received
            assert mock_bridge.low_cmd_received == True

            # Update motor commands
            mock_bridge.update_motor_cmd()

            # Verify control values were updated
            assert len(mock_bridge.mj_data.ctrl) == 2

            # Publish state
            with patch.object(mock_bridge, "parse_robot_specific_low_state"):
                mock_bridge.publish_low_state()

                mock_bridge.lc.publish.assert_called()
