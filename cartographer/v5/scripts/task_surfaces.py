"""Bounded measured-stereo proposals for marked landing planes and wall tops.

These are task hypotheses, never load-bearing, traversability, or flight proofs.
No inferred depth, map filling, absolute height, field dimensions or CAD input.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from task_perception import OpeningDetector, OpeningLimits, _fit_plane, _pixels_to_points, _intersect_plane
from task_observations import vector


def _inputs(image, depth, K, mask):
    image = np.asarray(image); depth = np.asarray(depth); K = vector(K,4)
    if image.dtype != np.uint8 or image.ndim != 2 or image.shape != depth.shape or min(image.shape)<32 or max(image.shape)>1920:
        raise ValueError('Require bounded synchronized mono8 and same-grid direct depth')
    if min(K[:2]) <= 0: raise ValueError('Invalid focal length')
    direct = np.isfinite(depth) & (depth >= .25) & (depth <= 8.)
    if mask is not None:
        if np.asarray(mask).shape != depth.shape: raise ValueError('Mask mismatch')
        direct &= np.asarray(mask,dtype=bool)
    return image,depth,K,direct


class LandingPlaneDetector:
    """High-contrast H/closed-shape marks anchored to a horizontal measured plane.

    H is a bounded contour-shape heuristic, not a trained semantic guarantee.
    Closed rectangles/circles stay `closed_mark` proposals. The adapter must
    choose a task marker class; plane geometry alone cannot identify a terminal.
    """
    def __init__(self):
        self.openings = OpeningDetector()
        template = np.array([[0,0],[2,0],[2,4],[6,4],[6,0],[8,0],[8,10],[6,10],[6,6],[2,6],[2,10],[0,10]],np.float32)
        self.h_template = template.reshape(-1,1,2)
        self.h_masks=[];self.cross_masks=[]
        for bar in (4,6,8):
            for bridge in (4,6,8):
                mask=np.zeros((32,32),bool); mask[:,:bar]=True; mask[:,-bar:]=True
                mask[16-bridge//2:16+bridge//2,:]=True
                self.h_masks.extend([mask,np.rot90(mask)])
        for width in (4,6,8,10,12):
            mask=np.zeros((32,32),bool);mask[:,16-width//2:16+width//2]=True
            mask[16-width//2:16+width//2,:]=True
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            box=cv2.boxPoints(cv2.minAreaRect(contours[0])).astype(np.float32)
            H=cv2.getPerspectiveTransform(box,np.array([[0,0],[31,0],[31,31],[0,31]],np.float32))
            normalized=cv2.warpPerspective(mask.astype(np.uint8),H,(32,32),flags=cv2.INTER_NEAREST)
            self.cross_masks.append(normalized!=0)

    def _marker_score(self,contour):
        box=cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        H=cv2.getPerspectiveTransform(box,np.array([[0,0],[31,0],[31,31],[0,31]],np.float32))
        normalized=cv2.perspectiveTransform(contour.astype(np.float32),H)
        silhouette=np.zeros((32,32),np.uint8)
        cv2.fillPoly(silhouette,[np.rint(normalized).astype(np.int32)],1)
        silhouette=silhouette!=0
        def score(masks):
            return 1-max(np.logical_and(silhouette,m).sum()/max(1,np.logical_or(silhouette,m).sum()) for m in masks)
        scores={'h':score(self.h_masks),'cross':score(self.cross_masks)}
        kind=min(scores,key=scores.get)
        return kind,float(scores[kind])

    def detect(self, image, measured_depth, K, exposure, now, measured_mask=None):
        T = exposure.validate(now)
        return self.detect_sensor_frame(image,measured_depth,K,T[:3,:3].T@np.array([0.,0.,1.]),
                                        int(round(exposure.stamp*1e9)),measured_mask)

    def detect_sensor_frame(self,image,measured_depth,K,up_camera,stamp_ns,measured_mask=None):
        """Current raw plane proposals using gravity direction, no world position.

        No contact/depth accuracy qualification is asserted. Caller must retain
        the actual exposure key and independently qualify rig/depth/attitude.
        """
        image,depth,K,direct = _inputs(image,measured_depth,K,measured_mask)
        up_camera=vector(up_camera)
        if abs(np.linalg.norm(up_camera)-1.)>.001 or not isinstance(stamp_ns,(int,np.integer)) or stamp_ns<0:
            raise ValueError('Unit camera-frame gravity direction and exact exposure ns required')
        candidates = [{'boundary':c['boundary'], 'center':c['center'], 'marker_kind':'closed_mark', 'score':None}
                      for c in self.openings._image_candidates(image)]
        scale = min(1.,640/image.shape[1]); small = cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        for mode in (cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV):
            _,binary = cv2.threshold(small,0,255,mode|cv2.THRESH_OTSU)
            contours,_ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:80]:
                area = cv2.contourArea(contour)
                if not .002*small.size < area < .5*small.size: continue
                x,y,w,h = cv2.boundingRect(contour)
                if min(w,h)<12 or x<2 or y<2 or x+w>=small.shape[1]-2 or y+h>=small.shape[0]-2: continue
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                if not .35 < area/max(1.,hull_area) < .85: continue
                marker_kind,score = self._marker_score(contour)
                if score > .25: continue
                hull = (cv2.convexHull(contour).reshape(-1,2)+.5)/scale-.5
                candidates.insert(0,{'boundary':hull,'center':hull.mean(0),'marker_kind':marker_kind,'score':float(score)})
        results=[]
        for c in candidates[:12]:
            boundary = np.asarray(c['boundary']); center = c['center']
            # Fit around AND on the marker, retaining missing/foreign samples.
            region=np.zeros(image.shape,np.uint8)
            expanded=np.rint(center+1.25*(boundary-center)).astype(np.int32)
            cv2.fillConvexPoly(region,cv2.convexHull(expanded),1)
            yy,xx=np.nonzero((region!=0)&direct)
            if len(xx)<32: continue
            ids=np.linspace(0,len(xx)-1,min(400,len(xx))).astype(int)
            uv=np.column_stack((xx[ids],yy[ids])); points=_pixels_to_points(uv,depth[yy[ids],xx[ids]],K)
            limits=OpeningLimits(min_rim_fraction=.8)
            fit=_fit_plane(points,limits)
            if fit is None: continue
            n,d,inliers,residual=fit
            if abs(n@up_camera) < math.cos(math.radians(15)): continue
            intersections=_intersect_plane(np.vstack((center,boundary)),K,n,d)
            if intersections is None: continue
            # Measured support must cover the mark's four image quadrants.
            q=(uv[inliers]>center).astype(int); quadrants=set(map(tuple,q))
            if len(quadrants)<4: continue
            results.append({'kind':'landing_plane','marker_kind':c['marker_kind'],
                            'marker_shape_score':c['score'],'frame':'left_optical',
                            'center_camera_m':intersections[0].tolist(),'normal_camera':n.tolist(),
                            'marker_boundary_camera_m':intersections[1:].tolist(),
                            'measured_plane_support':int(inliers.sum()),'sampled_direct_pixels':len(ids),
                            'support_points_camera_m':points[inliers].tolist(),
                            'region_unknown_fraction':float(1-direct[region!=0].mean()),
                            'plane_residual_p90_m':residual,'geometry_margin_m':max(.02,2*residual),
                            'landing_area_verified':False,'verified':False,
                            'semantic_identity_verified':False,'stamp':int(stamp_ns)*1e-9,'stamp_ns':int(stamp_ns),
                            'source':'mono mark and direct measured stereo plane'})
        return results[:6]


class WallTopDetector:
    """Observed wall/background discontinuities with direct points on both sides.

    A top edge is proposed only on a locally vertical measured wall, with the
    farther sample above it in measured local ENU. A low obstacle or tunnel lip
    remains possible. No body clearance or ceiling height is inferred.
    """
    def detect(self,image,measured_depth,K,exposure,now,measured_mask=None):
        image,depth,K,direct=_inputs(image,measured_depth,K,measured_mask)
        T=exposure.validate(now); h,w=depth.shape
        # Bound scan to <=640 columns and <=480 rows on native calibrated grid.
        xs=np.linspace(3,w-4,min(160,w//4)).astype(int)
        ys=np.linspace(3,h-4,min(240,h-6)).astype(int)
        edge=[]; wall=[]; background=[]
        for x in xs:
            pairs=direct[ys-2,x]&direct[ys+2,x]
            jump=depth[ys-2,x]-depth[ys+2,x]
            choices=np.flatnonzero(pairs&(jump>.2))
            if not len(choices): continue
            # Highest measured positive jump in each image column is a proposal.
            y=ys[choices[0]]
            edge.append([x,y+2]); wall.append([x,y+2]); background.append([x,y-2])
            for dy in (6,12,24,40):
                if y+dy<h and direct[y+dy,x]: wall.append([x,y+dy])
        if len(edge)<12 or len(wall)<24: return []
        wall=np.asarray(wall); points=_pixels_to_points(wall,depth[wall[:,1],wall[:,0]],K)
        fit=_fit_plane(points,OpeningLimits(min_rim_fraction=.8))
        if fit is None: return []
        n,d,inliers,residual=fit; n_local=T[:3,:3]@n
        if abs(n_local[2])>.25: return []
        edge=np.asarray(edge); bg=np.asarray(background)
        near=_pixels_to_points(edge,depth[edge[:,1],edge[:,0]],K)
        far=_pixels_to_points(bg,depth[bg[:,1],bg[:,0]],K)
        edge_local=near@T[:3,:3].T+T[:3,3]
        far_local=far@T[:3,:3].T+T[:3,3]
        valid=(np.abs(near@n+d)<.025)&(far_local[:,2]>edge_local[:,2])
        if valid.sum()<12: return []
        edge_local=edge_local[valid]; near=near[valid]
        # Reject jagged/disjoint tops and nearly point-like support.
        heights=edge_local[:,2]
        if np.ptp(heights)>.12 or np.linalg.norm(np.ptp(edge_local[:,:2],axis=0))<.2: return []
        center=near.mean(0)
        return [{'kind':'wall_top','frame':'left_optical','center_camera_m':center.tolist(),
                 'normal_camera':n.tolist(),'edge_camera_m':near.tolist(),
                 'edge_height_upper_local_m':float(heights.max()+exposure.pose_error_m+.03),
                 'measured_plane_support':int(inliers.sum()),'background_pairs':int(valid.sum()),
                 'geometry_margin_m':max(.03,2*residual),'verified':False,
                 'overflight_clearance_verified':False,'stamp':exposure.stamp,
                 'source':'direct stereo wall and farther above-edge samples'}]


class HomePlaneDetector:
    """Reobserve the own measured launch surface, including an unmarked floor.

    The location/height hint comes only from the startup landed ToF/body record.
    Plane anchors are fitted from CURRENT direct stereo; a raised platform or a
    different VIO gauge cannot silently become the home floor or a z correction.
    """
    def detect(self,measured_depth,K,exposure,foot_calibration,now,measured_mask=None):
        depth=np.asarray(measured_depth)
        dummy=np.zeros(depth.shape,np.uint8)
        _,depth,K,direct=_inputs(dummy,depth,K,measured_mask)
        T=exposure.validate(now);cal=foot_calibration
        if cal is None or cal.epoch!=exposure.epoch:return []
        yy,xx=np.nonzero(direct)
        if len(xx)<32:return []
        # A regular bounded native pixel subsample proposes a local plane;
        # final contact support uses the dense independent checker.
        ids=np.linspace(0,len(xx)-1,min(len(xx),4096)).astype(int)
        uv=np.column_stack((xx[ids],yy[ids]))
        points=_pixels_to_points(uv,depth[yy[ids],xx[ids]],K)
        local=points@T[:3,:3].T+T[:3,3];home=np.array(cal.resting_body_local_m)
        near=(np.max(np.abs(local[:,:2]-home[:2]),axis=1)<=.75)&(np.abs(local[:,2]-cal.resting_ground_z_local_m)<=cal.error_m+.06)
        local=local[near]
        if len(local)<32:return []
        if len(local)>400:local=local[np.linspace(0,len(local)-1,400).astype(int)]
        fit=_fit_plane(local,OpeningLimits(min_rim_fraction=.9))
        if fit is None:return []
        n,d,inliers,residual=fit
        if n[2]<0:n=-n;d=-d
        if n[2]<math.cos(math.radians(5)):return []
        center=home.copy();center[2]=-(d+n[:2]@home[:2])/n[2]
        if abs(center[2]-cal.resting_ground_z_local_m)>cal.error_m+exposure.pose_error_m+.02:return []
        observed=local[inliers]
        if len(set(map(tuple,(observed[:,:2]>center[:2]).astype(int))))<4:return []
        normal_camera=T[:3,:3].T@n;center_camera=T[:3,:3].T@(center-T[:3,3])
        return [{'kind':'landing_plane','frame':'left_optical','center_camera_m':center_camera.tolist(),
                 'normal_camera':normal_camera.tolist(),'clearance_lower_m':np.maximum(.01,np.ptp(observed[:,:2],axis=0)).tolist(),
                 'rim_measured_support':int(inliers.sum()),'geometry_margin_m':max(.01,2*residual),
                 'details':{'marker_kind':'own_launch_plane','support_points_local_m':observed.tolist(),
                            'observed_ground_z_local_m':float(center[2]),
                            'ground_height_above_measured_start_m':float(center[2]-cal.resting_ground_z_local_m),
                            'vio_global_z_correction_applied':False},'passage_clearance_verified':False}]
