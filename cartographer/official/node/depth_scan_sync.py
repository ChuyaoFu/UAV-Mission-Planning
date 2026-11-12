"""/team13/depth_scan (realdrone/depth_scan.py, sensor_msgs/LaserScan) -> mission.occgrid.Scan for Perception.scan.

The stereo node computes the scan from a depth image and stamps it with that image's stamp; the scan frame is that
rectified camera (x = optical z).  So the camera pose a scan needs is exactly the one mission_node computes for the
depth image of the SAME stamp (camera_pose_ned in on_depth).  Scans and depth images arrive in either order: this
pairs them by stamp (within tol_s) and converts with occgrid.scan_from_laserscan.  ROS-free (node/test_depth_scan_sync.py).
"""
from collections import OrderedDict

import numpy as np

from mission.occgrid import scan_from_laserscan


class DepthScanSync:
    def __init__(self, keep=20, tol_s=0.005, carve_max_m=3.0):
        self.keep, self.tol_s, self.carve_max_m = int(keep), float(tol_s), float(carve_max_m)
        self.poses = OrderedDict()                 # stamp -> (R_nc, cam_ned) of recent depth images
        self.pending = OrderedDict()               # stamp -> scan fields waiting for their depth pose
        self.matched = 0

    def _find(self, d, stamp):
        best, bk = None, None
        for k in d:
            e = abs(k - stamp)
            if e <= self.tol_s and (best is None or e < best):
                best, bk = e, k
        return bk

    @staticmethod
    def _trim(d, keep):
        while len(d) > keep:
            d.popitem(last=False)

    def _convert(self, stamp, pose, f):
        self.matched += 1
        return scan_from_laserscan(stamp, pose[0], pose[1], f['angle_min'], f['angle_increment'], f['ranges'],
                                   f['range_max'], carve_max_m=self.carve_max_m)

    def add_pose(self, stamp, R_nc, cam_ned):
        """A depth image's camera pose (optical -> NED, camera NED).  Returns the pending scans it completes."""
        stamp = float(stamp)
        pose = (np.asarray(R_nc, float).copy(), np.asarray(cam_ned, float).reshape(3).copy())
        self.poses[stamp] = pose
        self._trim(self.poses, self.keep)
        out = []
        k = self._find(self.pending, stamp)
        if k is not None:
            out.append(self._convert(k, pose, self.pending.pop(k)))
        return out

    def add_scan(self, stamp, angle_min, angle_increment, ranges, range_max):
        """A LaserScan.  Returns its occgrid.Scan if the depth pose of that stamp is known, else keeps it pending."""
        stamp = float(stamp)
        f = dict(angle_min=float(angle_min), angle_increment=float(angle_increment),
                 ranges=np.asarray(ranges, float).copy(), range_max=float(range_max))
        k = self._find(self.poses, stamp)
        if k is not None:
            return self._convert(stamp, self.poses[k], f)
        self.pending[stamp] = f
        self._trim(self.pending, self.keep)
        return None
