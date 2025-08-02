import mujoco
import numpy as np
import pinocchio as pin

from arc_bridge.state_estimators import FloatingBaseLinearStateEstimator, MovingWindowFilter
from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import tron1_pointfoot_state_t, tron1_pointfoot_control_t
from arc_bridge.utils import *

class Tron1PointfootBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

        self.torso_name = "base_Link" # body
        self.left_foot_link_name = "foot_L_Link" # body
        self.right_foot_link_name = "foot_R_Link" # body
        self.left_foot_name = "foot_L_collision" # geom
        self.right_foot_name = "foot_R_collision" # geom
        self.num_legs = 2

        # Override motor offsets (rad)
        self.joint_offsets = np.array([0, 0.53 - 0.06, -0.55 - 0.54,  # right leg
                                       0, 0.53 - 0.06, -0.55 - 0.54]) # left leg

        # Pinocchio model
        self.pin_model = pin.buildModelFromMJCF(self.config.robot_xml_path)
        self.pin_data = self.pin_model.createData()

        # State estimator
        self.height_init = 0.7
        # Process noise (px, py, pz, vx, vy, vz)
        KF_Q = np.diag([0.002, 0.002, 0.002, 0.02, 0.02, 0.02])
        # Measurement noise (pz, vx, vy, vz)
        KF_R = np.diag([0.001, 1, 1, 50])
        self.KF = FloatingBaseLinearStateEstimator(self.config.dt_sim, KF_Q, KF_R, self.height_init)
        self.low_state.position = [0, 0, self.height_init]
        self.low_state.quaternion = [1, 0, 0, 0] # wxyz
        self.low_cmd.contact = [1, 1]
        self.foot_radius = 0.032
        self.gravity = np.array([0, 0, -9.81])
        self.hip_pos_body_frame = np.array([0.05556 + 0.03, -0.105, -0.2602,
                                            0.05556 + 0.03,  0.105, -0.2602]).reshape(2, 3)

        # Contact estimation
        self.P_hat = np.zeros(self.pin_model.nv) # estimated generalized momentum
        self.Ko = 100 # observer gain
        self.contact_threshold = -4
        self.selection_mat = np.eye(self.pin_model.nv, self.num_motor, k=-6)

        # Signal smoothing
        self.se_filter = MovingWindowFilter(window_size=10, dim=6)

        # Visualization
        self.vis_se = True # override default flag
        self.vis_pos_est = np.array([0, 0, self.height_init])
        self.vis_vel_est = np.zeros(3)
        self.vis_R_body = np.eye(3)
        self.vis_box_size = [0.1, 0.1, 0.08]

    def update_state_estimation(self):
        #* torso twist linear in pin.LOCAL_WORLD_ALIGNED is the same as v_world!!!

        # Retrive states from IMU readings
        omega_body = self.low_state.omega
        acc_body = self.low_state.acceleration
        R_body_to_world = quat_to_rot(Quaternion(*self.low_state.quaternion))

        # Predict based on accelerations
        acc_world = R_body_to_world @ acc_body
        se_state = self.KF.predict(acc_world + self.gravity)
        # print(f"GT Vel: {self.mj_data.sensordata[self.dim_motor_sensor + 13:self.dim_motor_sensor + 16]}")
        # print(f"GT Pz: {self.mj_data.sensordata[self.dim_motor_sensor + 12]}")

        pf, vf = self.calculate_foot_position_and_velocity() # body frame
        # Correct based on foot contact
        for idx in range(self.num_legs):
            if self.low_cmd.contact[idx] > 0:
            # if self.low_state.foot_force[idx] > 0:
                foot_vel_body = vf[idx]
                vel_measured = -R_body_to_world @ (foot_vel_body + np.cross(omega_body, pf[idx]))
                height_measured = -(R_body_to_world @ pf[idx])[2] + self.foot_radius
                se_state = self.KF.correct(np.append(height_measured, vel_measured))
                # print(f"FK Vel: {vel_measured}\t 
                # phase: {self.low_cmd.contact[idx]}\t 
                # force: {self.low_state.foot_force[idx]}")
                # print(f"FK Pz: {height_measured}")

        se_state_smoothed = self.se_filter.calculate_average(se_state)
        self.vis_pos_est = se_state_smoothed[:3]
        self.vis_vel_est = se_state_smoothed[3:]
        self.vis_R_body = R_body_to_world

        # Write estimated states into low_state
        # self.low_state.position[2] = self.height_init
        self.low_state.position[2] = se_state_smoothed[2]
        self.low_state.velocity[:] = se_state_smoothed[3:]

        if self.low_cmd.reset_se:
            self.KF.reset(np.array([0, 0, self.height_init, 0, 0, 0]))

    def parse_robot_specific_low_state(self, backend="pinocchio"):
        # Parse common robot states to low_state first
        # Required fields: position, quaternion, velocity, omega, qj_pos, qj_vel

        # Update low_state.position[2] and low_state.velocity
        self.update_state_estimation()

        if backend == "pinocchio":
            # Overwrite position to (0, 0, pz)
            self.low_state.position[0] = 0
            self.low_state.position[1] = 0

            #* q = [pos, quat(xyzw), qj_pos] in Pinocchio conventions
            pin_q = np.zeros(self.pin_model.nq)
            pin_q[:3] = self.low_state.position.copy()
            pin_q[3:7] = quat_wxyz_to_xyzw(self.low_state.quaternion) # Pinocchio uses xyzw
            pin_q[7:] = np.array(self.low_state.qj_pos) - self.joint_offsets
            # print(f"q: {pin_q}")

            #* v = [v_body, omega_body, qj_vel] in Pinocchio conventions
            R_body_to_world = pin.Quaternion(pin_q[3:7]).toRotationMatrix()
            pin_v = np.concatenate((R_body_to_world.T @ self.low_state.velocity, self.low_state.omega, self.low_state.qj_vel), axis=0)
            # print(f"dq: {pin_v}")

            # Update forward kinematics, dynamics and frame placements in Pinocchio
            pin.computeAllTerms(self.pin_model, self.pin_data, pin_q, pin_v)
            pin.updateFramePlacements(self.pin_model, self.pin_data)


            # Retrive states from Pinocchio data
            self.update_kinematics_and_dynamics_pinocchio(pin_q, pin_v) # TODO change a better name
        else:
            self.update_kinematics_and_dynamics_mujoco()

    def update_kinematics_and_dynamics_mujoco(self):
        # Send inertia matrix and bias force
        temp_inertia_mat = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_mat, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_mat.tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias

        # Parse J and dJdq
        # np.set_printoptions(precision=4, suppress=True, linewidth=1000)
        #* dq = [v_world, omega_body, qj_vel]
        dq = np.concatenate((self.low_state.velocity, self.low_state.omega, self.low_state.qj_vel), axis=0)
        
        # Torso jacobians
        # J_tor
        torso_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, self.torso_name)
        torso_pos = self.mj_data.xpos[torso_id]
        J_tor = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_tor[:3, :], J_tor[3:, :], torso_pos, torso_id)
        self.low_state.J_tor = J_tor.tolist()
        # assert(np.linalg.norm((J_tor @ dq)[3:6] - omega_world) < 1e-6)

        # dJdq_tor
        dJ_tor = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_tor[:3, :], dJ_tor[3:, :], torso_pos, torso_id)
        dJdq_tor = dJ_tor @ dq
        self.low_state.dJdq_tor = dJdq_tor.tolist()

        # Left foot jacobians
        left_foot_link_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, self.left_foot_link_name)

        # J_lin_foot_L
        left_foot_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_GEOM, self.left_foot_name)
        left_foot_pos = self.mj_data.geom_xpos[left_foot_id]
        J_lin_foot_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_foot_L, None, left_foot_pos, left_foot_link_id)

        # dJdq_foot_L
        dJ_foot_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_foot_L, None, left_foot_pos, left_foot_link_id)
        dJdq_foot_L = dJ_foot_L @ dq

        # Right foot jacobians
        right_foot_link_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, self.right_foot_link_name)

        # J_lin_foot_R
        right_foot_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_GEOM, self.right_foot_name)
        right_foot_pos = self.mj_data.geom_xpos[right_foot_id]
        J_lin_foot_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_foot_R, None, right_foot_pos, right_foot_link_id)

        # dJdq_foot_R
        dJ_foot_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_foot_R, None, right_foot_pos, right_foot_link_id)
        dJdq_foot_R = dJ_foot_R @ dq

        # Assemble gc jacobians
        J_gc = np.concatenate((J_lin_foot_R, J_lin_foot_L), axis=0)
        self.low_state.J_gc = J_gc.tolist()

        dJdq_gc = np.concatenate((dJdq_foot_R, dJdq_foot_L), axis=0)
        self.low_state.dJdq_gc = dJdq_gc.tolist()

        p_gc = np.concatenate((right_foot_pos, left_foot_pos), axis=0)
        self.low_state.p_gc = p_gc.tolist()

    def update_kinematics_and_dynamics_pinocchio(self, pin_q, pin_v):
        # Pinocchio computations
        # np.set_printoptions(precision=4, suppress=True, linewidth=1000)

        R_body_to_world = pin.Quaternion(pin_q[3:7]).toRotationMatrix()
        #* dq = [v_world, omega_body, qj_vel] in Mujoco conventions
        dq = np.concatenate((self.low_state.velocity, self.low_state.omega, self.low_state.qj_vel), axis=0)

        # H and C
        transform_mat = np.eye(self.pin_model.nv)
        transform_mat[:3, :3] = R_body_to_world
        H_prime = transform_mat @ self.pin_data.M @ transform_mat.T
        v_temp = np.zeros((self.pin_model.nv))
        v_temp[:3] = np.cross(R_body_to_world @ pin_v[3:6], dq[:3])
        C_prime = transform_mat @ self.pin_data.nle - H_prime @ v_temp
        # print(f"H:\n{self.pin_data.M}")
        # print(f"H:\n{temp_inertia_mat}")
        # print(f"H_diff:\n{H_prime - temp_inertia_mat}")
        # print(f"C_diff:\n{C_prime - self.mj_data.qfrc_bias}")
        # assert(np.allclose(self.low_state.inertia_mat, H_prime, atol=1e-5))
        # assert(np.allclose(self.low_state.bias_force, C_prime, atol=1e-5))
        self.low_state.inertia_mat = H_prime.tolist()
        self.low_state.bias_force = C_prime.tolist()

        # Momentum observer
        coriolis_mat = pin.computeCoriolisMatrix(self.pin_model, self.pin_data, pin_q, pin_v)
        generalized_gravity = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, pin_q)
        tau_motor = np.array(self.low_cmd.qj_tau)
        P_curr = self.pin_data.M @ pin_v
        tau_ext_hat = self.Ko * (P_curr - self.P_hat)[6:]
        dP_hat = coriolis_mat.T @ pin_v - generalized_gravity + self.selection_mat @ tau_motor + self.selection_mat @ tau_ext_hat
        self.P_hat += dP_hat * self.config.dt_sim
        # print(f"right foot: {tau_ext_hat[2]:.4f} phase: {self.low_cmd.contact[0]:.4f}\t\
        #        left foot: {tau_ext_hat[5]:.4f}, phase: {self.low_cmd.contact[1]:.4f}")
        
        # TODO use pinv(J_gc') * tau_ext_hat to estimate contact forces
        knee_impact = tau_ext_hat[[2, 5]]
        contact_mask = knee_impact < self.contact_threshold
        self.low_state.foot_force = contact_mask.astype(np.float32).tolist()
        # print(f"Detected contact: {self.low_state.foot_force}")

        # J_tor
        J_geom_tor = pin.getFrameJacobian(self.pin_model, 
                                          self.pin_data, 
                                          self.pin_model.getFrameId("base_Link"), 
                                          pin.WORLD) # J_G maps body velocity to world twist
        J_G_to_A = np.eye(3, 6)
        J_G_to_A[:3, 3:] = - pin.skew(pin_q[:3])
        J_lin_tor = J_G_to_A @ J_geom_tor @ transform_mat.T
        J_tor = np.concatenate((J_lin_tor, J_geom_tor[3:, :]), axis=0)
        # print(f"J_geom_tor:\n{J_geom_tor}")
        # print(f"J_tor:\n{J_tor}")
        # assert(np.allclose(self.low_state.J_tor, J_tor, atol=1e-5))
        self.low_state.J_tor = J_tor.tolist()

        # dJdq_tor
        dJ_geom_tor = pin.frameJacobianTimeVariation(self.pin_model, 
                                                self.pin_data,
                                                pin_q,
                                                pin_v,
                                                self.pin_model.getFrameId("base_Link"), 
                                                pin.WORLD)
        transform_mat_derivative = np.zeros((self.pin_model.nv, self.pin_model.nv))
        transform_mat_derivative[:3, :3] = - pin.skew(pin_v[3:6]) @ R_body_to_world.T
        dJ_lin_tor = J_G_to_A @ (dJ_geom_tor @ transform_mat.T + J_geom_tor @ transform_mat_derivative) #- pin.skew(dq[:3]) @ J_geom_tor[3:, :] @ transform_mat.T
        dJ_tor = np.concatenate((dJ_lin_tor, dJ_geom_tor[3:, :]), axis=0)
        dJdq_tor = dJ_tor @ dq
        # assert(np.allclose(self.low_state.dJdq_tor, dJdq_tor, atol=1e-5))
        self.low_state.dJdq_tor = dJdq_tor.tolist()

        # J_lin_foot_L
        J_geom_foot_L = pin.getFrameJacobian(self.pin_model, 
                                             self.pin_data, 
                                             self.pin_model.getFrameId("foot_L_Link"), 
                                             pin.WORLD)
        left_foot_pos = self.pin_data.oMf[self.pin_model.getFrameId("foot_L_Link")].translation
        J_lin_foot_L_G_to_A = np.eye(3, 6)
        J_lin_foot_L_G_to_A[:, 3:] = -pin.skew(left_foot_pos)
        J_lin_foot_L = J_lin_foot_L_G_to_A @ J_geom_foot_L @ transform_mat.T

        # dJdq_foot_L
        dJ_geom_foot_L = pin.frameJacobianTimeVariation(self.pin_model, 
                                                        self.pin_data,
                                                        pin_q,
                                                        pin_v,
                                                        self.pin_model.getFrameId("foot_L_Link"), 
                                                        pin.WORLD)
        dJ_lin_foot_L = J_lin_foot_L_G_to_A @ (dJ_geom_foot_L @ transform_mat.T + J_geom_foot_L @ transform_mat_derivative)
        dJdq_foot_L = dJ_lin_foot_L @ dq

        # J_lin_foot_R
        J_geom_foot_R = pin.getFrameJacobian(self.pin_model, 
                                             self.pin_data, 
                                             self.pin_model.getFrameId("foot_R_Link"), 
                                             pin.WORLD)
        right_foot_pos = self.pin_data.oMf[self.pin_model.getFrameId("foot_R_Link")].translation
        J_lin_foot_R_G_to_A = np.eye(3, 6)
        J_lin_foot_R_G_to_A[:, 3:] = -pin.skew(right_foot_pos)
        J_lin_foot_R = J_lin_foot_R_G_to_A @ J_geom_foot_R @ transform_mat.T

        # dJdq_foot_R
        dJ_geom_foot_R = pin.frameJacobianTimeVariation(self.pin_model, 
                                                        self.pin_data,
                                                        pin_q,
                                                        pin_v,
                                                        self.pin_model.getFrameId("foot_R_Link"), 
                                                        pin.WORLD)
        dJ_lin_foot_R = J_lin_foot_R_G_to_A @ (dJ_geom_foot_R @ transform_mat.T + J_geom_foot_R @ transform_mat_derivative)
        dJdq_foot_R = dJ_lin_foot_R @ dq

        # Assemble gc jacobians
        # J_gc
        J_gc = np.concatenate((J_lin_foot_R, J_lin_foot_L), axis=0)
        # assert(np.allclose(self.low_state.J_gc, J_gc, atol=1e-5))
        self.low_state.J_gc = J_gc.tolist()

        # dJdq_gc
        dJdq_gc = np.concatenate((dJdq_foot_R, dJdq_foot_L), axis=0)
        # assert(np.allclose(self.low_state.dJdq_gc, dJdq_gc, atol=1e-5))
        self.low_state.dJdq_gc = dJdq_gc.tolist()

        # p_gc
        p_gc = np.concatenate((right_foot_pos, left_foot_pos), axis=0)
        # assert(np.allclose(self.low_state.p_gc, p_gc, atol=1e-5))
        self.low_state.p_gc = p_gc.tolist()

    def calculate_foot_position_and_velocity(self):
        l1 = 0.077 # abad to hip
        l2 = 0.3   # hip to knee
        l3 = 0.3   # knee to foot

        qj_pos_np = np.array(self.low_state.qj_pos).reshape((2, 3))
        qj_vel_np = np.array(self.low_state.qj_vel).reshape((2, 3))
        th1, th2, th3 = qj_pos_np[:, 0], qj_pos_np[:, 1], qj_pos_np[:, 2]
        dth1, dth2, dth3 = qj_vel_np[:, 0], qj_vel_np[:, 1], qj_vel_np[:, 2]

        p_foot = np.array([-l1 - l3*np.sin(th2 + th3) - l2*np.sin(th2),
                           np.sin(th1)*(l3*np.cos(th2 + th3) + l2*np.cos(th2)),
                           -np.cos(th1)*(l3*np.cos(th2 + th3) + l2*np.cos(th2))]).T

        v_foot = np.array([-l2*dth2*np.cos(th2) - l3*dth2*np.cos(th2 + th3) - l3*dth3*np.cos(th2 + th3),
                           l2*dth1*np.cos(th1)*np.cos(th2) 
                           - l2*dth2*np.sin(th1)*np.sin(th2) 
                           + l3*dth1*np.cos(th1)*np.cos(th2)*np.cos(th3) 
                           - l3*dth1*np.cos(th1)*np.sin(th2)*np.sin(th3) 
                           - l3*dth2*np.cos(th2)*np.sin(th1)*np.sin(th3) 
                           - l3*dth2*np.cos(th3)*np.sin(th1)*np.sin(th2) 
                           - l3*dth3*np.cos(th2)*np.sin(th1)*np.sin(th3) 
                           - l3*dth3*np.cos(th3)*np.sin(th1)*np.sin(th2),
                           l2*dth1*np.sin(th1)*np.cos(th2) 
                           + l2*dth2*np.cos(th1)*np.sin(th2) 
                           + l3*dth1*np.sin(th1)*np.cos(th2)*np.cos(th3) 
                           + l3*dth2*np.cos(th1)*np.cos(th2)*np.sin(th3) 
                           + l3*dth2*np.cos(th1)*np.cos(th3)*np.sin(th2) 
                           + l3*dth3*np.cos(th1)*np.cos(th2)*np.sin(th3) 
                           + l3*dth3*np.cos(th1)*np.cos(th3)*np.sin(th2) 
                           - l3*dth1*np.sin(th1)*np.sin(th2)*np.sin(th3)]).T

        return p_foot + self.hip_pos_body_frame, v_foot

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return

        msg = eval(self.topic_state+"_t").decode(data)
        self.mj_data.qpos[0] = msg.position[0]
        self.mj_data.qpos[1] = msg.position[1]
        self.mj_data.qpos[2] = self.low_state.position[2]
        self.mj_data.qpos[3] = msg.quaternion[0]
        self.mj_data.qpos[4] = msg.quaternion[1]
        self.mj_data.qpos[5] = msg.quaternion[2]
        self.mj_data.qpos[6] = msg.quaternion[3]
        self.mj_data.qpos[7:7+6] = msg.qj_pos # ! This one doesn't need offsets since it matchs with xml
        self.mj_data.qvel[:] = 0

        # Partially update low_state
        self.low_state.qj_pos[:] = (np.array(msg.qj_pos) + self.joint_offsets).tolist() # ! This one needs offsets since it should match with controller's model
        self.low_state.qj_vel[:] = msg.qj_vel
        self.low_state.qj_tau[:] = msg.qj_tau
        self.low_state.acceleration[:] = msg.acceleration
        self.low_state.omega[:] = msg.omega
        self.low_state.quaternion[:] = msg.quaternion
        self.low_state.rpy[:] = msg.rpy
