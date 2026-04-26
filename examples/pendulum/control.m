clc; clear; 

% Setup LCM protocol
lcm_cmd_topic = "pendulum_control";
lcm_state_topic = "pendulum_state";
lc = lcm.lcm.LCM("udpm://239.255.76.67:7667?ttl=1");

% Subscribe to state messages
lcm_agg_state = lcm.lcm.MessageAggregator();
lcm_agg_state.setMaxMessages(1);
lc.subscribe(lcm_state_topic, lcm_agg_state);

control_freq = 100; % control frequency in Hz

rate_ctrl = rateControl(control_freq);

%% Main Loop
while true
    msg = lcm_agg_state.getNextMessage(0);
    if isempty(msg)
        continue
    end

    lc_state = lcm_msgs.pendulum_state_t(msg.data);


    % Parse state
    joint_pos = lc_state.qj_pos;
    joint_vel = lc_state.qj_vel;

    % Controller
    kp = 15;
    kd = 1;
    des_joint_pos = pi;
    des_joint_vel = 0.0;

    pos_err = des_joint_pos - joint_pos;
    vel_err = des_joint_vel - joint_vel;

    ctrl_torque = kp * pos_err + kd * vel_err;

    % Publish joint torque command to LCM
    lc_cmd = lcm_msgs.pendulum_control_t();
    lc_cmd.qj_tau = ctrl_torque;

    % Enable low-level joint PD control
    % lc_cmd.qj_pos = des_joint_pos;
    % lc_cmd.qj_vel = des_joint_vel;
    % lc_cmd.kp(:) = kp;
    % lc_cmd.kd(:) = kd;
    lc.publish(lcm_cmd_topic, lc_cmd);

    rate_ctrl.waitfor();
end
