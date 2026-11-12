"""Shared frame conventions for the official-lab (Sky Beta / OAK-D) stack.

World  : PX4 local NED (x north, y east, z down), metres.
Body   : FRD. ROS base_link is FLU; F_FLU_FRD converts between them.
Camera : optical frame of oak_left (x right, y down, z forward). The official
         rig mounts it horizontal (optical axis ~0 deg off body forward).
PX4 attitude q is FRD->NED, ordered (w, x, y, z).
"""
from dataclasses import dataclass, field
import math
import numpy as np

F_FLU_FRD = np.diag([1., -1., -1.])


def quat_wxyz_to_mat(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def camera_pose_ned(q_wxyz, p_ned, R_bc_flu, t_bc_flu):
    """Return (R_nc optical->NED, camera position NED) from PX4 attitude/position
    and the TF base_link(FLU) <- camera(optical) extrinsic."""
    R_nb = quat_wxyz_to_mat(*q_wxyz)
    R_nc = R_nb @ F_FLU_FRD @ np.asarray(R_bc_flu, float)
    cam = np.asarray(p_ned, float) + R_nb @ (F_FLU_FRD @ np.asarray(t_bc_flu, float))
    return R_nc, cam


@dataclass
class Frame:
    """One synchronized perception input. depth: metres along the optical z axis,
    aligned to `image` (same K), NaN or <=0 = no return."""
    image: np.ndarray            # uint8 HxW grayscale (left camera)
    depth: np.ndarray            # float32 HxW
    K: tuple                     # (fx, fy, cx, cy)
    R_nc: np.ndarray             # 3x3 optical -> NED
    cam_ned: np.ndarray          # (3,) camera position NED
    stamp: float = 0.0
    body_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: float = 0.0         # PX4 heading (rad, NED yaw)

    def points_ned(self, stride=4, max_range=6.0):
        """Back-project valid depth pixels to NED points: (N,3) and their (v,u)."""
        fx, fy, cx, cy = self.K
        d = self.depth[::stride, ::stride]
        vv, uu = np.mgrid[0:self.depth.shape[0]:stride, 0:self.depth.shape[1]:stride]
        ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
        z = d[ok]; u = uu[ok]; v = vv[ok]
        pc = np.stack([(u-cx)/fx*z, (v-cy)/fy*z, z], -1)
        return pc @ self.R_nc.T + self.cam_ned, np.stack([v, u], -1)

    def project(self, p_ned):
        """NED points (N,3) -> pixel (u,v) and optical depth z."""
        pc = (np.atleast_2d(p_ned) - self.cam_ned) @ self.R_nc
        fx, fy, cx, cy = self.K
        z = pc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.stack([fx*pc[:, 0]/z + cx, fy*pc[:, 1]/z + cy], -1), z


# Official rig (from TF / camera_info, 2026-11-07); use live TF/CameraInfo in flight.
OFFICIAL_K = (421.0159629821777, 421.0159629821777, 319.24780082825066, 200.65893581327742)
OFFICIAL_SIZE = (640, 400)
OFFICIAL_T_BASE_CAM = (0.116, 0.037, 0.029)          # base_link FLU <- oak_left, metres
# optical (x right, y down, z fwd) -> FLU (x fwd, y left, z up) for a horizontal camera
OFFICIAL_R_BASE_CAM = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
SKY_BETA_HALF_SPAN_M = 0.20                           # 7-inch quad incl. props (approx., verify)
