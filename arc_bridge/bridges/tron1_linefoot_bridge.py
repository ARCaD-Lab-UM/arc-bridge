import mujoco
import numpy as np
import pinocchio as pin

from arc_bridge.state_estimators import FloatingBaseLinearStateEstimator, MovingWindowFilter
from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import tron1_linefoot_state_t, tron1_linefoot_control_t
from arc_bridge.utils import *

class Tron1LinefootBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

        # Override motor offsets (rad)
        self.joint_offsets = np.array([0, 0.53, -1.04, 0.536,  
                                       0, 0.53, -1.04, 0.536])
        
        # CoM offsets (m)
        self.com_offsets = np.array([0, 0, 0.2602]) * 0

        self.torso_name = "imu" # site
        self.left_foot_link_name = "ankle_L_Link" # body
        self.right_foot_link_name = "ankle_R_Link" # body
        self.left_heel_name = "heel_L_site" # site
        self.left_toe_name = "toe_L_site" # site
        self.right_heel_name = "heel_R_site" # site
        self.right_toe_name = "toe_R_site" # site
        self.gc_names = [self.right_heel_name, 
                         self.right_toe_name, 
                         self.left_heel_name, 
                         self.left_toe_name] # name of links for ground contact
        self.num_legs = 2

        # Pinocchio model
        self.pin_model = pin.buildModelFromMJCF(self.config.robot_xml_path)
        self.pin_data = self.pin_model.createData()

        # State estimator
        self.height_init = 0.75
        # Process noise (px, py, pz, vx, vy, vz)
        KF_Q = np.diag([0.002, 0.002, 0.002, 0.02, 0.02, 0.02])
        # Measurement noise (pz, vx, vy, vz)
        KF_R = np.diag([0.001, 1, 1, 100])
        self.KF = FloatingBaseLinearStateEstimator(self.config.dt_sim, KF_Q, KF_R, self.height_init)
        self.low_state.position = [0, 0, self.height_init]
        self.low_state.quaternion = [1, 0, 0, 0] # wxyz
        self.low_cmd.contact = [1, 1]
        self.foot_radius = 0.058
        self.gravity = np.array([0, 0, -9.81])
        self.hip_pos_body_frame = np.array([0.05556, -0.105, -0.2602,
                                            0.05556,  0.105, -0.2602]).reshape(2, 3)\
                                            + self.com_offsets

        # Signal smoothing
        self.se_pos_filter = MovingWindowFilter(window_size=10, dim=3)
        self.se_vel_filter = MovingWindowFilter(window_size=20, dim=3)

        # Contact estimation
        self.P_hat = np.zeros(self.pin_model.nv) # estimated generalized momentum
        self.Ko = 100 # observer gain
        self.contact_threshold = -5
        self.contact_counter = np.zeros(self.num_legs)
        self.selection_mat = np.eye(self.pin_model.nv, self.num_motor, k=-6)

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
        # print(f"phase: {self.low_cmd.contact[0]:.2f}, {self.low_cmd.contact[1]:.2f}")
        for idx in range(self.num_legs):
            if self.low_cmd.contact[idx] > 0.2:
            # if self.low_state.foot_force[idx*2] > 10:
                foot_vel_body = vf[idx]
                foot_vel_body[-1] = 0 #! set vz to zero, vf is still not very accurate
                vel_measured = -R_body_to_world @ (foot_vel_body + np.cross(omega_body, pf[idx]))
                height_measured = -(R_body_to_world @ pf[idx])[2]
                se_state = self.KF.correct(np.append(height_measured, vel_measured))
                # se_state = self.KF.correct(np.append(self.low_state.position[2], vel_measured))
                # print(f"FK Vel: {vel_measured}\t \
                #         phase: {self.low_cmd.contact[idx]}\t \
                #         force: {self.low_state.foot_force[idx]}")
                # print(f"FK Pz: {height_measured:.4f}")

        se_pos_smoothed = self.se_pos_filter.calculate_average(se_state[:3])
        se_vel_smoothed = self.se_vel_filter.calculate_average(se_state[3:])

        # Update visualization
        self.vis_pos_est = se_pos_smoothed
        self.vis_vel_est = se_vel_smoothed
        self.vis_R_body = R_body_to_world

        # Write estimated states into low_state
        # self.low_state.position[2] = self.height_init
        self.low_state.position[2] = se_pos_smoothed[2]
        self.low_state.velocity[:] = se_vel_smoothed

        # Handle reset
        if self.low_cmd.reset_se:
            self.KF.reset(np.array([0, 0, self.height_init, 0, 0, 0]))

    def parse_robot_specific_low_state(self, backend="pinocchio"):
        # Parse common robot states to low_state first
        # Required fields: position, quaternion, velocity, omega, qj_pos, qj_vel

        # Update low_state.position[2] and low_state.velocity
        # TODO may need move this as a separate call in the outter loop to not update during blocking
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

            # Momentum observer
            coriolis_mat = pin.computeCoriolisMatrix(self.pin_model, self.pin_data, pin_q, pin_v)
            generalized_gravity = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, pin_q)
            tau_motor = np.array(self.low_cmd.qj_tau)
            P_curr = self.pin_data.M @ pin_v
            tau_ext_hat = self.Ko * (P_curr - self.P_hat)[6:]
            dP_hat = coriolis_mat.T @ pin_v - generalized_gravity + self.selection_mat @ tau_motor + self.selection_mat @ tau_ext_hat
            self.P_hat += dP_hat * self.config.dt_sim
            
            F_gc = np.linalg.pinv(np.array(self.low_state.J_gc)[:, 6:].T) @ tau_ext_hat
            F_leg_gc = F_gc.reshape(2, -1)
            F_foot_gc = np.sum(F_leg_gc, axis=1)
            # print(f"right foot: {F_foot_gc[0]:.4f} phase: {self.low_cmd.contact[0]:.4f}\t\
            #        left foot: {F_foot_gc[1]:.4f}, phase: {self.low_cmd.contact[1]:.4f}")

            contact_mask = F_foot_gc > 20

            self.low_state.foot_force = [0, 0, 0, 0]
            for idx in range(self.num_legs):
                if contact_mask[idx]:
                    self.contact_counter[idx] += 1
                    if self.contact_counter[idx] > 2:
                        self.low_state.foot_force[idx*2] = 1
                        self.low_state.foot_force[idx*2 + 1] = 1
                else:
                    self.contact_counter[idx] = 0

            # Retrive states from Pinocchio data
            self.update_kinematics_and_dynamics_pinocchio(pin_q, pin_v)
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
        torso_site_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, self.torso_name)
        torso_id = self.mj_model.site_bodyid[torso_site_id] # find the body id where the COM site is attached
        torso_pos = self.mj_data.site_xpos[torso_site_id] # find the site position in world frame
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
        left_ankle_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, self.left_foot_link_name)

        # J_lin_heel_L
        left_heel_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, self.left_heel_name)
        left_heel_pos = self.mj_data.site_xpos[left_heel_id]
        J_lin_heel_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_heel_L, None, left_heel_pos, left_ankle_id)

        # dJdq_heel_L
        dJ_heel_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_heel_L, None, left_heel_pos, left_ankle_id)
        dJdq_heel_L = dJ_heel_L @ dq

        # J_lin_toe_L
        left_toe_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, self.left_toe_name)
        left_toe_pos = self.mj_data.site_xpos[left_toe_id]
        J_lin_toe_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_toe_L, None, left_toe_pos, left_ankle_id)

        # dJdq_toe_L
        dJ_toe_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_toe_L, None, left_heel_pos, left_ankle_id)
        dJdq_toe_L = dJ_toe_L @ dq

        # Right foot jacobians
        right_ankle_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, self.right_foot_link_name)

        # J_lin_heel_R
        right_heel_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, self.right_heel_name)
        right_heel_pos = self.mj_data.site_xpos[right_heel_id]
        J_lin_heel_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_heel_R, None, right_heel_pos, right_ankle_id)

        # dJdq_heel_R
        dJ_heel_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_heel_R, None, right_heel_pos, right_ankle_id)
        dJdq_heel_R = dJ_heel_R @ dq

        # J_lin_toe_R
        right_toe_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, self.right_toe_name)
        right_toe_pos = self.mj_data.site_xpos[right_toe_id]
        J_lin_toe_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_toe_R, None, right_toe_pos, right_ankle_id)

        # dJdq_toe_R
        dJ_toe_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_toe_R, None, right_heel_pos, right_ankle_id)
        dJdq_toe_R = dJ_toe_R @ dq

        # Assemble gc jacobians
        J_gc = np.concatenate((J_lin_heel_R, J_lin_toe_R, J_lin_heel_L, J_lin_toe_L), axis=0)
        self.low_state.J_gc = J_gc.tolist()

        dJdq_gc = np.concatenate((dJdq_heel_R, dJdq_toe_R, dJdq_heel_L, dJdq_toe_L), axis=0)
        self.low_state.dJdq_gc = dJdq_gc.tolist()

        p_gc = np.concatenate((right_heel_pos, right_toe_pos, left_heel_pos, left_toe_pos), axis=0)
        self.low_state.p_gc = p_gc.tolist()

        right_ankle_pos = self.mj_data.xpos[right_ankle_id]
        left_ankle_pos = self.mj_data.xpos[left_ankle_id]
        self.low_state.p_ankle = np.concatenate((right_ankle_pos, left_ankle_pos), axis=0).tolist()

    def update_kinematics_and_dynamics_pinocchio(self, pin_q, pin_v):
        # Pinocchio computations
        R_body_to_world = pin.Quaternion(pin_q[3:7]).toRotationMatrix()
        #* dq = [v_world, omega_body, qj_vel] in Mujoco conventions
        dq = np.concatenate((self.low_state.velocity, self.low_state.omega, self.low_state.qj_vel), axis=0)

        # Mass matrix and bias force
        transform_mat = np.eye(self.pin_model.nv)
        transform_mat[:3, :3] = R_body_to_world
        H_prime = transform_mat @ self.pin_data.M @ transform_mat.T
        v_temp = np.zeros((self.pin_model.nv))
        v_temp[:3] = np.cross(R_body_to_world @ pin_v[3:6], dq[:3])
        C_prime = transform_mat @ self.pin_data.nle - H_prime @ v_temp

        self.low_state.inertia_mat = H_prime.tolist()
        self.low_state.bias_force = C_prime.tolist()

        # Torso Jacobians and their derivatives
        # J_tor
        J_geom_tor = pin.getFrameJacobian(self.pin_model, 
                                          self.pin_data, 
                                          self.pin_model.getFrameId(self.torso_name), 
                                          pin.WORLD) # J_G maps body velocity to world twist
        J_G_to_A = np.eye(3, 6)
        J_G_to_A[:3, 3:] = - pin.skew(pin_q[:3])
        J_lin_tor = J_G_to_A @ J_geom_tor @ transform_mat.T
        J_tor = np.concatenate((J_lin_tor, J_geom_tor[3:, :]), axis=0)

        self.low_state.J_tor = J_tor.tolist()

        # dJdq_tor
        dJ_geom_tor = pin.frameJacobianTimeVariation(self.pin_model, 
                                                self.pin_data,
                                                pin_q,
                                                pin_v,
                                                self.pin_model.getFrameId(self.torso_name), 
                                                pin.WORLD)
        transform_mat_derivative = np.zeros((self.pin_model.nv, self.pin_model.nv))
        transform_mat_derivative[:3, :3] = - pin.skew(pin_v[3:6]) @ R_body_to_world.T
        dJ_lin_tor = J_G_to_A @ (dJ_geom_tor @ transform_mat.T + J_geom_tor @ transform_mat_derivative) #- pin.skew(dq[:3]) @ J_geom_tor[3:, :] @ transform_mat.T
        dJ_tor = np.concatenate((dJ_lin_tor, dJ_geom_tor[3:, :]), axis=0)
        dJdq_tor = dJ_tor @ dq

        self.low_state.dJdq_tor = dJdq_tor.tolist()

        # Contact Jacobians and their derivatives
        J_gc = np.zeros((12, self.pin_model.nv))
        dJdq_gc = np.zeros((12, ))
        p_gc = np.zeros((12, ))
        for idx, name in enumerate(self.gc_names):
            J_geom = pin.getFrameJacobian(self.pin_model,
                                          self.pin_data, 
                                          self.pin_model.getFrameId(name), 
                                          pin.WORLD)
            link_pos = self.pin_data.oMf[self.pin_model.getFrameId(name)].translation
            J_G_to_A = np.eye(3, 6)
            J_G_to_A[:, 3:] = -pin.skew(link_pos)
            J_lin = J_G_to_A @ J_geom @ transform_mat.T

            dJ_geom = pin.frameJacobianTimeVariation(self.pin_model,
                                                    self.pin_data,
                                                    pin_q,
                                                    pin_v,
                                                    self.pin_model.getFrameId(name), 
                                                    pin.WORLD)
            dJ_lin = J_G_to_A @ (dJ_geom @ transform_mat.T + J_geom @ transform_mat_derivative)
            dJdq = dJ_lin @ dq

            J_gc[idx*3:(idx+1)*3, :] = J_lin
            dJdq_gc[idx*3:(idx+1)*3] = dJdq
            p_gc[idx*3:(idx+1)*3] = link_pos

        self.low_state.J_gc = J_gc.tolist()
        self.low_state.dJdq_gc = dJdq_gc.tolist()
        self.low_state.p_gc = p_gc.tolist()

        # Ankle positions in world frame
        right_ankle_pos = self.pin_data.oMf[self.pin_model.getFrameId(self.right_foot_link_name)].translation
        left_ankle_pos = self.pin_data.oMf[self.pin_model.getFrameId(self.left_foot_link_name)].translation
        self.low_state.p_ankle = np.concatenate((right_ankle_pos, left_ankle_pos), axis=0).tolist()

    def calculate_foot_position_and_velocity(self):
        l1 = 0.077 # abad to hip
        l2 = 0.3   # hip to knee
        l3 = 0.3   # knee to foot

        qj_pos_np = np.array(self.low_state.qj_pos).reshape((2, 4))
        qj_vel_np = np.array(self.low_state.qj_vel).reshape((2, 4))
        th1, th2, th3 = qj_pos_np[:, 0], qj_pos_np[:, 1], qj_pos_np[:, 2]
        dth1, dth2, dth3 = qj_vel_np[:, 0], qj_vel_np[:, 1], qj_vel_np[:, 2]

        p_foot = np.array([-l1 - l3*np.sin(th2 + th3) - l2*np.sin(th2),
                           np.sin(th1)*(l3*np.cos(th2 + th3) + l2*np.cos(th2)),
                           -np.cos(th1)*(l3*np.cos(th2 + th3) + l2*np.cos(th2)) - self.foot_radius]).T

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

    def lowStateHandler(self, channel, data):
        if self.mj_data == None:
            return
        # Get state msg from robot SDK topic
        msg = eval(self.topic_state+"_t").decode(data)

        # Update mj_data for visualization
        self.mj_data.qpos[0] = self.low_state.position[0]
        self.mj_data.qpos[1] = self.low_state.position[1]
        self.mj_data.qpos[2] = self.low_state.position[2]
        self.mj_data.qpos[3] = msg.quaternion[0]
        self.mj_data.qpos[4] = msg.quaternion[1]
        self.mj_data.qpos[5] = msg.quaternion[2]
        self.mj_data.qpos[6] = msg.quaternion[3]
        self.mj_data.qpos[7:7+8] = msg.qj_pos - self.joint_offsets
        self.mj_data.qvel[:] = 0

        # Partially update low_state
        # self.low_state.qj_pos[:] = (np.array(msg.qj_pos) + self.joint_offsets).tolist() # ! This one needs offsets since it should match with controller's model
        self.low_state.qj_pos[:] = msg.qj_pos
        self.low_state.qj_vel[:] = msg.qj_vel
        self.low_state.qj_tau[:] = msg.qj_tau
        self.low_state.acceleration[:] = msg.acceleration
        self.low_state.omega[:] = msg.omega
        self.low_state.quaternion[:] = msg.quaternion
        self.low_state.rpy[:] = msg.rpy
