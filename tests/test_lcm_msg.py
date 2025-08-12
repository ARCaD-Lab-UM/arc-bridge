import pytest
import importlib
from pathlib import Path


def get_available_lcm_message_types():
    """Dynamically discover available LCM message types from lcm_types folder"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    lcm_types_dir = project_root / "lcm_types"
    
    message_types = []
    if lcm_types_dir.exists():
        # Find all .lcm files and extract message type names
        for lcm_file in lcm_types_dir.glob("*.lcm"):
            # Convert filename like "hopper_control_t.lcm" to "hopper_control_t"
            message_type = lcm_file.stem
            message_types.append(message_type)
    
    return sorted(message_types)  # Sort for consistent test ordering


class TestLcmTypeFunctionality:
    """Tests for LCM message import, instantiation, encoding, and decoding functionality"""

    def test_lcm_msgs_package_exists(self):
        """Test that the lcm_msgs package can be imported"""
        try:
            import arc_bridge.lcm_msgs

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import arc_bridge.lcm_msgs: {e}")

    def test_import_common_gamepad_message(self):
        """Test importing gamepad command message"""
        try:
            from arc_bridge.lcm_msgs import gamepad_cmd_t

            assert gamepad_cmd_t is not None
        except ImportError as e:
            pytest.skip(f"gamepad_cmd_t not available: {e}")

    def test_dynamic_message_import(self):
        """Test dynamic import of LCM messages like the bridge does"""
        # Test the eval-based import pattern used in the bridge
        try:
            # This simulates how the bridge dynamically imports message types
            topic_name = "gamepad_cmd"
            msg_class = eval(
                f"__import__('arc_bridge.lcm_msgs', fromlist=['{topic_name}_t']).{topic_name}_t"
            )
            assert msg_class is not None

            # Test instantiation
            msg = msg_class()
            assert msg is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Dynamic import test skipped: {e}")

    @pytest.mark.parametrize(
        "message_type",
        get_available_lcm_message_types(),
    )
    def test_message_type_instantiation(self, message_type):
        """Test that message types can be instantiated"""
        try:
            module = importlib.import_module("arc_bridge.lcm_msgs")
            msg_class = getattr(module, message_type)
            msg_instance = msg_class()
            assert msg_instance is not None
        except (ImportError, AttributeError):
            pytest.skip(f"{message_type} not available")

    def test_gamepad_message_encoding(self):
        """Test encoding gamepad command messages"""
        try:
            from arc_bridge.lcm_msgs import gamepad_cmd_t

            msg = gamepad_cmd_t()
            msg.timestamp = 1234567890
            msg.vx = 1.5
            msg.vy = -0.5
            msg.wz = 0.8
            msg.e_stop = False

            encoded_data = msg.encode()

            assert isinstance(encoded_data, bytes)
            assert len(encoded_data) > 0

            # Basic sanity check - encoded data should be non-empty
            assert encoded_data != b""
        except ImportError:
            pytest.skip("gamepad_cmd_t not available")

    def test_gamepad_message_round_trip(self):
        """Test encoding and then decoding a gamepad message"""
        try:
            from arc_bridge.lcm_msgs import gamepad_cmd_t

            # Create original message
            original_msg = gamepad_cmd_t()
            original_msg.timestamp = 1234567890
            original_msg.vx = 1.5
            original_msg.vy = -0.5
            original_msg.wz = 0.8
            original_msg.e_stop = False

            # Encode
            encoded_data = original_msg.encode()

            # Decode
            decoded_msg = gamepad_cmd_t.decode(encoded_data)

            # Verify all fields match
            assert decoded_msg.timestamp == original_msg.timestamp
            assert decoded_msg.vx == pytest.approx(original_msg.vx, abs=1e-6)
            assert decoded_msg.vy == pytest.approx(original_msg.vy, abs=1e-6)
            assert decoded_msg.wz == pytest.approx(original_msg.wz, abs=1e-6)
            assert decoded_msg.e_stop == original_msg.e_stop

        except ImportError:
            pytest.skip("gamepad_cmd_t not available")

    def test_bridge_pattern_decode(self):
        """Test the decoding pattern used in the bridge"""
        try:
            from arc_bridge.lcm_msgs import gamepad_cmd_t

            # Create a message using the bridge's eval pattern
            topic_cmd = "gamepad_cmd"
            original_msg = eval(f"{topic_cmd}_t")()
            original_msg.timestamp = 1111111111
            original_msg.vx = 2.0
            original_msg.vy = 1.0
            original_msg.wz = -1.0
            original_msg.e_stop = True

            # Encode
            encoded_data = original_msg.encode()

            # Decode using bridge pattern
            decoded_msg = eval(f"{topic_cmd}_t").decode(encoded_data)

            # Verify
            assert decoded_msg.timestamp == original_msg.timestamp
            assert decoded_msg.vx == pytest.approx(original_msg.vx, abs=1e-6)
            assert decoded_msg.vy == pytest.approx(original_msg.vy, abs=1e-6)
            assert decoded_msg.wz == pytest.approx(original_msg.wz, abs=1e-6)
            assert decoded_msg.e_stop == original_msg.e_stop

        except (ImportError, NameError):
            pytest.skip("Bridge pattern test not available")
