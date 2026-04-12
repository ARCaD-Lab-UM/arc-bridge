import os
from threading import Lock

import numpy as np
import pinocchio as pin

from arc_bridge.utils import wrap_to_pi_format

class Tron1WheeledRobot:
    """Pinocchio-backed kinematic model + state estimator for the Tron1 wheeled.
    """

    # 8-d joint vector [abad_L, hip_L, knee_L, wheel_L, abad_R, hip_R, knee_R, wheel_R]
    NUM_JOINTS = 8
    LEFT_WHEEL_JOINT_IDX = 3  # index inside the 8-d joint vector
    RIGHT_WHEEL_JOINT_IDX = 7

    def __init__(
        self,
        urdf_path,
        joint_offsets=None,
        wheel_link_names=("wheel_L_Link", "wheel_R_Link"),
        wheel_radius=0.127,
    ):
        """Build the Pinocchio model from a URDF file.

        Args:
            urdf_path (string): absolute path to the Tron1 wheeled URDF.
            joint_offsets (array-like, optional): 8-vector of joint offsets. Defaults to None.
            wheel_link_names (tuple, optional): URDF link names of the two wheels. Defaults to ("wheel_L_Link", "wheel_R_Link").
            wheel_radius (float, optional): wheel radius in meters. Defaults to 0.127.

        Raises:
            FileNotFoundError: if the URDF file does not exist at the given path.
            ValueError: if joint_offsets is provided but does not have length 8.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Pinocchio model with FreeFlyer floating base.
        # nq = 7 (floating base) + 8 (joints) = 15
        # nv = 6 (floating base) + 8 (joints) = 14
        self.pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.pin_data = self.pin_model.createData()

        # Joint offsets between our convention and the URDF zero pose
        # note: the URDF here is the un-straightened one.
        if joint_offsets is None:
            joint_offsets = np.zeros(self.NUM_JOINTS)
        self.joint_offsets = np.asarray(joint_offsets, dtype=float).reshape(-1)
        if self.joint_offsets.shape[0] != self.NUM_JOINTS:
            raise ValueError(f"joint_offsets must have length {self.NUM_JOINTS}, got {self.joint_offsets.shape}")

        self.wheel_indices = [self.pin_model.getFrameId(name) for name in wheel_link_names]
        self.wheel_radius = float(wheel_radius)

        # assume flat ground plane
        self.ground_normal_vec = np.array([0.0, 0.0, 1.0])

        # latest state
        self.q_curr = pin.neutral(self.pin_model)   # nq
        self.v_curr = np.zeros(self.pin_model.nv)   # nv
        self.last_yaw = 0.0

        # Odometry state (thread-safe: reset_odometry may be called from vicon callback)
        self._odom_lock = Lock()
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self._odom_initialized = False
        self._last_wheel_angles = np.zeros(2)  # [phi_L, phi_R] in rad
        self.odometry_threshold = 0.004 #0.01  # rad, switching threshold

    def estimate_torso_state(
        self,
        joint_pos,
        joint_vel,
        rpy=None,
        quaternion=None,
        omega_body=None,
    ):
        """Estimate torso height and linear velocity from FK + rolling contact.

        Args:
            joint_pos (array-like of length 8): joint positions in our convention.
            joint_vel (array-like of length 8): joint velocities.
            rpy (array-like, optional): roll/pitch/yaw from the IMU. rpy is preferred over quat. Defaults to None.
            quaternion (array-like, optional): full body quaternion (w, x, y, z). Defaults to None.
            omega_body (array-like, optional): body-frame angular velocity (3,) from the IMU. Defaults to None.

        Raises:
            ValueError: mis-aligned input shapes, or both rpy and quaternion missing.
            ValueError: if joint_pos or joint_vel does not have length 8.

        Returns:
            pos_world (np.ndarray, shape (3,)): torso position. x/y from odometry, z from FK.
            vel_world (np.ndarray, shape (3,)): torso linear velocity in world frame.
            info (dict): per-leg estimates and debug info.
        """
        joint_pos = np.asarray(joint_pos, dtype=float).reshape(-1)
        joint_vel = np.asarray(joint_vel, dtype=float).reshape(-1)
        if joint_pos.shape[0] != self.NUM_JOINTS or joint_vel.shape[0] != self.NUM_JOINTS:
            raise ValueError("Expected {self.NUM_JOINTS} joints, got pos={joint_pos.shape}, vel={joint_vel.shape}")

        # Resolve roll/pitch/yaw
        if rpy is not None:
            rpy_arr = np.asarray(rpy, dtype=float).reshape(3)
        elif quaternion is not None:
            quat_arr = np.asarray(quaternion, dtype=float).reshape(4)
            # pin.Quaternion constructor takes (w, x, y, z) as positional args
            quat_pin = pin.Quaternion(float(quat_arr[0]), float(quat_arr[1]), float(quat_arr[2]), float(quat_arr[3]))
            rpy_arr = pin.rpy.matrixToRpy(quat_pin.toRotationMatrix())
        else:
            raise ValueError("Either rpy or quaternion must be provided")

        yaw = float(rpy_arr[2])

        # Pinocchio world frame is the yaw-aligned frame.
        # keep roll/pitch, yaw=0 is the body frame relative to yaw-aligned frame
        rpy_aligned = np.array([rpy_arr[0], rpy_arr[1], 0.0])
        R_aligned = pin.rpy.rpyToMatrix(rpy_aligned)
        quat_aligned = pin.Quaternion(R_aligned)

        # Build q in pinocchio convention
        # Floating-base nq layout: [x, y, z, qx, qy, qz, qw, joint(8)...]
        q_curr = np.zeros(self.pin_model.nq)
        q_curr[0:3] = 0.0  # set torso translation to 0 so contact z directly gives the negative torso height after FK.
        q_curr[3:7] = quat_aligned.coeffs()  # (x, y, z, w) order
        q_curr[7:] = joint_pos - self.joint_offsets
        # wheel positions should have no effect on kinematics and dynamics
        q_curr[7 + self.LEFT_WHEEL_JOINT_IDX] = 0.0
        q_curr[7 + self.RIGHT_WHEEL_JOINT_IDX] = 0.0

        # Build v in pinocchio convention
        # Floating-base nv layout: [vlin_body(3), vang_body(3), joint_vel(8)...]
        v_curr = np.zeros(self.pin_model.nv)
        v_curr[0:3] = 0.0  # set body linear velocity to 0 so resulting contact velocity = -v_torso
        if omega_body is not None:
            v_curr[3:6] = np.asarray(omega_body, dtype=float).reshape(3)
        else:
            v_curr[3:6] = 0.0
        v_curr[6:] = joint_vel

        # Forward kinematics + frame placements
        pin.computeAllTerms(self.pin_model, self.pin_data, q_curr, v_curr)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # Handle per-wheel rolling contact point
        p_wheel = np.zeros((2, 3))
        R_wheel = np.zeros((2, 3, 3))
        # Fetch wheel frame poses and rotations
        for idx, wheel_idx in enumerate(self.wheel_indices):
            oMf = self.pin_data.oMf[wheel_idx]
            p_wheel[idx] = oMf.translation
            R_wheel[idx] = oMf.rotation

        # wheel frame y axis in world frame
        wheel_axis_vec = R_wheel[:, :, 1]  # (2, 3)

        # r_vec: vector from wheel center to the contact point on the groud plane, expressed in world frame
        r_vecs_world = np.cross(np.cross(self.ground_normal_vec, wheel_axis_vec), wheel_axis_vec)  # (2, 3)
        r_vecs_world /= np.linalg.norm(r_vecs_world, axis=1, keepdims=True)
        r_vecs_world *= self.wheel_radius  # scale to wheel radius

        # Per-wheel height and velocity estimates
        height_estimates = []  # yaw-aligned frame relative to on-the-ground world frame
        vel_estimates = []  # yaw-aligned frame relative to on-the-ground world frame
        for idx, wheel_idx in enumerate(self.wheel_indices):
            p_contact = p_wheel[idx] + r_vecs_world[idx]  # contact point position
            # the torso was placed at the origin (world frame) so torso height = -contact_z
            height_estimates.append(-p_contact[2])

            # wheel frame velocity (with v_torso=0) in LOCAL_WORLD_ALIGNED
            v_wheel_spatial = pin.getFrameVelocity(self.pin_model, self.pin_data, wheel_idx, pin.LOCAL_WORLD_ALIGNED)
            v_wheel_lin = np.asarray(v_wheel_spatial.linear, dtype=float)
            omega_wheel = np.asarray(v_wheel_spatial.angular, dtype=float)

            # assume contact point has 0 velocity relative to ground plane, so torso velocity = -contact velocity computed from FK
            v_contact = v_wheel_lin + np.cross(omega_wheel, r_vecs_world[idx])
            vel_estimates.append(-v_contact)

        height_estimates = np.array(height_estimates, dtype=float)
        vel_estimates = np.vstack(vel_estimates)
        height_mean = float(np.mean(height_estimates))
        vel_mean_yaw_aligned = np.mean(vel_estimates, axis=0)

        # Rotate back to real world frame
        R_yaw = pin.rpy.rpyToMatrix(np.array([0.0, 0.0, yaw]))
        vel_world = R_yaw @ vel_mean_yaw_aligned

        # Update odometry
        wheel_angles = np.array([joint_pos[self.LEFT_WHEEL_JOINT_IDX], joint_pos[self.RIGHT_WHEEL_JOINT_IDX]])
        p_contact = p_wheel + r_vecs_world  # (2, 3) in yaw-aligned frame
        wheel_base_y = abs(p_contact[0, 1] - p_contact[1, 1])
        self._update_odometry(wheel_angles, yaw, wheel_base_y)

        # x/y from odometry, z from FK
        pos_world = np.zeros(3, dtype=float)
        with self._odom_lock:
            pos_world[0] = self.odom_x
            pos_world[1] = self.odom_y
        pos_world[2] = height_mean

        self.q_curr = q_curr
        self.v_curr = v_curr
        self.last_yaw = yaw

        info = {
            "yaw": yaw,
            "rpy_aligned": rpy_aligned,
            "p_wheel": p_wheel,
            "r_vecs_world": r_vecs_world,
            "height_per_leg": height_estimates,
            "vel_per_leg_yaw_aligned": vel_estimates,
            "vel_yaw_aligned": vel_mean_yaw_aligned,
            "vel_world": vel_world,
            "height_mean": height_mean,
            "odom_x": self.odom_x,
            "odom_y": self.odom_y,
            "wheel_base_y": wheel_base_y,
        }
        return pos_world, vel_world, self.odom_yaw, info

    def _update_odometry(self, wheel_angles, imu_yaw, wheel_base_y):
        """Differential drive odometry via wheel encoder delta accumulation.

        Gyrodometry: fuse wheel-odom yaw with IMU yaw. When odom and gyro agree on delta_yaw (within threshold), use odom (less drift).
        When they disagree (e.g. wheel slip), trust the gyro.

        Args:
            wheel_angles: [phi_L, phi_R] total wheel encoder angles (rad).
            imu_yaw: torso yaw from IMU (rad).
            wheel_base_y: distance between left/right contact points in y (m).
        """
        with self._odom_lock:
            if not self._odom_initialized:
                self._last_wheel_angles[:] = wheel_angles
                self.odom_yaw = imu_yaw
                self._last_imu_yaw = imu_yaw
                self._odom_initialized = True
                return

            delta_phi = wheel_angles - self._last_wheel_angles
            self._last_wheel_angles[:] = wheel_angles

            # Delta yaw from differential drive; [0] left
            delta_yaw_odom = wrap_to_pi_format((delta_phi[1] - delta_phi[0]) * self.wheel_radius / max(wheel_base_y, 1e-3))
            # Delta yaw from IMU
            delta_yaw_gyro = wrap_to_pi_format(imu_yaw - self._last_imu_yaw)
            self._last_imu_yaw = imu_yaw

            # use odom delta when consistent, gyro when disagreement
            if abs(delta_yaw_gyro - delta_yaw_odom) > self.odometry_threshold:
                delta_yaw = delta_yaw_gyro  # wheel slip
            else:
                delta_yaw = delta_yaw_odom  # less drift

            # Use mid-point yaw for better arc approximation
            mid_yaw = wrap_to_pi_format(self.odom_yaw + 0.5 * delta_yaw)
            self.odom_yaw = wrap_to_pi_format(self.odom_yaw + delta_yaw)

            # Forward displacement
            delta_s = (delta_phi[0] + delta_phi[1]) / 2.0 * self.wheel_radius
            self.odom_x += delta_s * np.cos(mid_yaw)
            self.odom_y += delta_s * np.sin(mid_yaw)

    def reset_odometry(self, x=0.0, y=0.0):
        with self._odom_lock:
            self.odom_x = float(x)
            self.odom_y = float(y)
