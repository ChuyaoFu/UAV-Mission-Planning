"""Bounded synchronous helper suitable for the existing LatestWorker.

Use only direct measured stereo and exposure-time camera pose. The role comes
from the executive's current task, never a world coordinate or turn sequence.
No ROS imports and no flight control side effects.
"""
import numpy as np
from copy import deepcopy
from task_perception import OpeningDetector
from task_observations import TargetTracker,measured_forward_evidence
from task_surfaces import LandingPlaneDetector,WallTopDetector,HomePlaneDetector


class TaskPerceptionHelper:
    def __init__(self,foot_calibration=None):
        self.openings=OpeningDetector(); self.landing=LandingPlaneDetector(); self.wall=WallTopDetector();self.home=HomePlaneDetector();self.foot_calibration=foot_calibration
        self.tracker=TargetTracker()

    def observe(self,image,measured_depth,K,exposure,role,now,measured_mask=None):
        T=exposure.validate(now)
        if role in ('tunnel_entry','tunnel_exit','window','door'):
            candidates=self.openings.detect(image,measured_depth,K,measured_mask)
            shape='ellipse' if role.startswith('tunnel') else 'rectangle'
            candidates=[c for c in candidates if c['kind']==shape]
        elif role=='home':
            candidates=self.home.detect(measured_depth,K,exposure,self.foot_calibration,now,measured_mask)
        elif role in ('dock','wall'):
            detector=self.landing if role=='dock' else self.wall
            rows=detector.detect(image,measured_depth,K,exposure,now,measured_mask)
            candidates=[]
            for row in rows:
                boundary=row.get('marker_boundary_camera_m',row.get('edge_camera_m'))
                points=np.asarray(boundary)
                # These are association extents, not aperture clearance.
                dims=np.sort(np.ptp(points,axis=0))[-2:]
                dims=np.maximum(.01,dims)
                details={key:value for key,value in row.items() if key in ('marker_kind','semantic_identity_verified','edge_height_upper_local_m','region_unknown_fraction')}
                if 'support_points_camera_m' in row:
                    support=np.asarray(row['support_points_camera_m'])
                    details['support_points_local_m']=(support@T[:3,:3].T+T[:3,3]).tolist()
                if role=='wall':
                    details['edge_local_m']=(points@T[:3,:3].T+T[:3,3]).tolist()
                candidates.append({**row,'rim_measured_support':row['measured_plane_support'],
                                   'clearance_lower_m':dims.tolist(), 'rim_boundary_camera_m':boundary,
                                   'details':details})
        else:
            raise ValueError('Unknown task role')
        # Return detached snapshots: LatestWorker may process a newer frame
        # while the control thread is still reading its previous result.
        return deepcopy(self.tracker.update(candidates,exposure,role,now))

    def frame(self,image,measured_depth,K,exposure,role,now,measured_mask=None):
        """One synchronized worker result with forward-only post-crossing proof."""
        tracks=[] if role is None else self.observe(image,measured_depth,K,exposure,role,now,measured_mask)
        forward=measured_forward_evidence(measured_depth,K,exposure,now,measured_mask)
        return {'stamp':exposure.stamp,'epoch':exposure.epoch,'tracks':tracks,'forward_evidence':forward}
