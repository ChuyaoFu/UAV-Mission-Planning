# Retention, gravity-aligned yaw reference and covariance chart

A=R_W_Ihome, B=R_W_Icurrent, C=R_I_BODY, l=t_I_BODY. These are Hamilton active rotations; native JPL Rot() is transposed once. Define initial estimated BODY forward f=A*C*e_x, psi=atan2(f_y,f_x), Z=Rz(psi)=R_W_H. Home physical reference origin is the initial BODY origin p0+A*l, not the initial IMU and not a CAD or surveyed home position. World Z/gravity-up is unchanged, and initial BODY roll/pitch are not assumed zero.

```
d = p1 + B*l - p0 - A*l
t_H = Z.T*d
R_H_BODY = Z.T*B*C
```

The native perturbation convention was checked against actual PoseJPL::update: A'=A*Exp(dtheta0), B'=B*Exp(dtheta1), p'=p+dp_G to first order. Let k=[-f_y,f_x,0]/(f_x²+f_y²) and jpsi=k*(-A*skew(C*e_x)). Then for output error [additive position_H, LEFT small-angle orientation_H], input order[thetaIhome,pGhome,thetaIcurrent,pGcurrent], the complete Jacobian is

```
J_t_theta0 = -skew(e_z)*t_H*jpsi + Z.T*A*skew(l)
J_t_p0     = -Z.T
J_t_theta1 = -Z.T*B*skew(l)
J_t_p1     =  Z.T
J_theta_theta0 = -e_z*jpsi
J_theta_theta1 = Z.T*B
J_theta_p0 = J_theta_p1 = 0
P_out = J*P_joint*J.T
```

All original home/current and angle/position cross terms are retained. No independent prior, guessed .15m subtraction, diagonal-only approximation, eigenvalue clipping or zero measurement is introduced. Output symmetry is restored only by averaging transpose; invalid finite/symmetry/PSD telemetry skips the optional message rather than modifying state or original outputs. At the identity same-time reference, common position uncertainty cancels, but current BODY roll/pitch uncertainty does NOT vanish: the native pitched-IMU/non-level-BODY test retains angular covariance trace0.00503635rad². Home-angle uncertainty also enters the origin lever and yaw derivative. Near-vertical BODY forward makes yaw undefined; optional telemetry rejects that chart instead of assuming level orientation.

This primary yaw-frame covariance differs from the full-body relative diagnostic's C0 translation/C1 right-tangent chart. Both primary blocks are in gravity-aligned H. It is not quaternion-component or finite Euler-angle covariance. To transform a BODY point x into H with this output, y=R*x+t and point Jacobian is[I,-skew(R*x)] for the LEFT orientation chart. Native local IMU-angle interfaces require a real covariance chart transformation including cross terms. Merely renaming the frame or feeding this covariance into sensor_frames' existing native-angle input is incorrect.

Retention lifecycle: capture the existing first clone AFTER normal augmentation and time-covariance logic. When its normal eviction arrives, existing callers still erase it from recent `_clones_IMU`; the central marginalize guard preserves its original variable and full P rows/columns. That catches regular cleanup, initialization catch-up and direct explicit-ZUPT removal. All feature cleanup, SLAM reanchor and recent-window decisions stay unchanged. Subsequent EKF updates condition the retained home mean/cross covariance normally; it is not frozen to its first numeric mean. No new factor refers to it after eviction. FEJ of the active baseline variables and the home copy follows the original implementation. A new State/process defines a new anchor epoch; never combine across sessions by frame-id string alone.

Marginal-equivalence argument: partition joint state as active x and unmeasured retained h. Measurement Jacobian is[H_x,0]. S=H_x*P_xx*H_x.T+R and K_x=P_xx*H_x.T*S^-1 use only the same active marginal as baseline. K_h updates the real correlated home posterior without influencing x. Propagation and delayed feature initialization preserve this block property, and deleting other variables uses ordinary marginal submatrix removal. The recent feature map is identical, so the same data/means produce the same active measurement systems and reanchor choices. Finite arithmetic, threading and input-drop effects still need root's full native replay check.

Native extracted ACTUAL StateHelper bodies, original PoseJPL/IMU/Vec types and a minimal State registry shim were run through240cycles propagation/augmentation/update/window eviction plus14delayed variable initializations and11deletions. All active means/full active covariance marginals matched with max differences0; recent map11, at most+6 dimensions, PSD, same FEJ, real updated home mean and nonzero cross terms. This is not a full ROS/VioManager/frontend replay. Root must independently compare the original output trajectories and posterior diagnostics, native resource use, accepted visual updates and physical grading before any adapter or flight promotion.
