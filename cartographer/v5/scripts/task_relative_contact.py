"""Sensor-frame-only support/ToF evidence. No world position or flight commands.

Qualification records are externally supplied declarations from independent
sensor tests. Synthetic/unqualified records can produce diagnostics only. A
local result never authorizes motion or replaces the global map's veto.
"""
from dataclasses import dataclass,replace
from collections import deque
import hashlib,math,re
import numpy as np
import cv2


class RelativeContractError(ValueError):pass


def vec(x):
    a=np.asarray(x,float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise RelativeContractError('Expected finite 3-vector')
    return a


def unit(x):
    a=vec(x)
    if abs(np.linalg.norm(a)-1)>.001:raise RelativeContractError('Expected unit direction')
    return a/np.linalg.norm(a)


def rigid(x):
    a=np.asarray(x,float)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1]) or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-7) or abs(np.linalg.det(a[:3,:3])-1)>1e-7:
        raise RelativeContractError('Invalid calibrated rigid transform')
    return a


def digest_matrix(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def digest_tof(mount,axis):return digest_matrix(np.r_[vec(mount),unit(axis)])


def require(ok,reason):
    if not ok:raise RelativeContractError(reason)


@dataclass(frozen=True)
class SensorErrorDomain:
    calibration_id: str
    camera_profile_id: str
    qualification_record_sha256: str
    qualification_scope: str
    qualified_sensors: tuple
    body_from_camera_sha256: str
    tof_extrinsics_sha256: str
    resolution: tuple
    intrinsics: tuple
    independent_stride_px: float
    depth_interval_m: tuple
    optical_z_error_m: object
    camera_translation_error_m: float
    camera_rotation_error_rad: float
    plane_normal_error_rad: object
    tof_range_error_m: float
    tof_mount_error_m: float
    tof_axis_error_rad: float
    up_error_rad: float
    speed_error_mps: float
    gyro_error_radps: float

    def validate(self,stereo=False):
        require(bool(self.calibration_id),'Missing actual robot calibration')
        numbers=[self.tof_range_error_m,self.tof_mount_error_m,self.tof_axis_error_rad,self.up_error_rad,self.speed_error_mps,self.gyro_error_radps]
        require(np.isfinite(numbers).all() and min(numbers)>=0,'Missing/invalid independent local error bounds')
        require(0<self.tof_range_error_m<=.02 and self.tof_mount_error_m<=.02 and max(self.tof_axis_error_rad,self.up_error_rad)<=.05,'Local qualification domain exceeds supported precision')
        if stereo:
            require(bool(self.camera_profile_id),'Missing actual stereo profile')
            require(self.camera_translation_error_m is not None and self.camera_rotation_error_rad is not None and np.isfinite([self.camera_translation_error_m,self.camera_rotation_error_rad]).all() and self.camera_translation_error_m>=0 and 0<=self.camera_rotation_error_rad<=.05,'Camera extrinsic error bounds missing/invalid')
            require(self.resolution is not None and len(self.resolution)==2 and all(isinstance(x,(int,np.integer)) and 32<=x<=1920 for x in self.resolution),'Actual native resolution missing/invalid')
            require(self.intrinsics is not None and len(self.intrinsics)==4 and np.isfinite(self.intrinsics).all() and min(self.intrinsics[:2])>0,'Actual intrinsics missing/invalid')
            require(self.optical_z_error_m is not None and math.isfinite(self.optical_z_error_m) and 0<self.optical_z_error_m<=.02,'Absolute stereo error qualification missing; residual is not accuracy')
            require(self.plane_normal_error_rad is not None and math.isfinite(self.plane_normal_error_rad) and 0<=self.plane_normal_error_rad<=.03,'Current plane normal bound missing; fit residual cannot supply it')
            require(self.depth_interval_m is not None and len(self.depth_interval_m)==2 and 0<self.depth_interval_m[0]<self.depth_interval_m[1]<=8.,'Invalid qualified stereo range interval')
            require(self.independent_stride_px is not None and math.isfinite(self.independent_stride_px) and 1<=self.independent_stride_px<=8,'Independent measurement stride missing')
        return self

    def runtime_qualified(self,sensors):
        return bool(self.qualification_scope=='independent-sensor-bench' and
                    re.fullmatch('[0-9a-f]{64}',self.qualification_record_sha256 or '') and
                    set(sensors)<=set(self.qualified_sensors))


@dataclass(frozen=True)
class FootRangeSample:
    stamp_ns: int
    attitude_stamp_ns: int
    epoch: str
    calibration_id: str
    range_m: float
    up_body: tuple
    velocity_body_mps: tuple
    gyro_body_radps: tuple
    landed: bool
    armed: bool
    native_received_ns: int
    range_valid: bool=True
    attitude_valid: bool=True

    def validate(self,now_ns,domain):
        domain.validate()
        require(self.calibration_id==domain.calibration_id and bool(self.epoch),'Range/attitude calibration or epoch mismatch')
        require(self.stamp_ns==self.attitude_stamp_ns,'ToF needs attitude at its actual exposure, not nearest unsynchronized attitude')
        require(0<=now_ns-self.stamp_ns<=150000000,'Stale/future ToF exposure')
        require(0<=now_ns-self.native_received_ns<=350000000,'Fresh native status/landed receipt required; PX4 source clock is separate')
        require(self.range_valid and self.attitude_valid and math.isfinite(self.range_m) and .015<=self.range_m<=4.,'Invalid measured ToF/attitude')
        return unit(self.up_body),vec(self.velocity_body_mps),vec(self.gyro_body_radps)


def tof_geometry(sample,domain,tof_in_body,tof_axis_body,now_ns):
    up,velocity,gyro=sample.validate(now_ns,domain)
    mount=vec(tof_in_body);axis=unit(tof_axis_body)
    require(digest_tof(mount,axis)==domain.tof_extrinsics_sha256,'ToF extrinsics differ from qualified robot calibration')
    hit=mount+axis*sample.range_m
    require(math.acos(float(np.clip(-up@axis,-1,1)))+domain.up_error_rad+domain.tof_axis_error_rad<=math.radians(8),'Actual ToF ray not qualified downward')
    point_error=domain.tof_range_error_m+domain.tof_mount_error_m+2*sample.range_m*math.sin(domain.tof_axis_error_rad/2)
    vertical_error=point_error+2*np.linalg.norm(hit)*math.sin(domain.up_error_rad/2)
    return up,velocity,gyro,hit,float(vertical_error),float(point_error)


@dataclass(frozen=True)
class FootOffsetCalibration:
    offset_m: float
    error_m: float
    epoch: str
    calibration_id: str
    first_stamp_ns: int
    last_stamp_ns: int
    sample_count: int
    runtime_qualified: bool
    qualification_record_sha256: str
    source: str='native-landed stationary ToF/attitude; no world position supplied'


class RelativeFootCalibrator:
    """Foot offset only. Home/world anchoring is deliberately a separate input."""
    def __init__(self,domain,tof_in_body,tof_axis_body,diagnostic_only=False):
        self.domain=domain;self.mount=vec(tof_in_body);self.axis=unit(tof_axis_body)
        self.diagnostic_only=diagnostic_only;self.samples=deque(maxlen=100);self.epoch=None;self.last_rejection=None

    def ingest(self,s,now_ns):
        try:
            up,v,w,hit,error,_=tof_geometry(s,self.domain,self.mount,self.axis,now_ns)
            require(s.landed and not s.armed,'Confirmed disarmed native landed required')
            require(np.linalg.norm(v)+self.domain.speed_error_mps<=.03 and np.linalg.norm(w)+self.domain.gyro_error_radps<=.03,'Stationary local velocity/gyro bounds not satisfied')
            require(math.acos(float(np.clip(up[2],-1,1)))+self.domain.up_error_rad<=math.radians(2),'Startup upright tilt/error bound exceeded')
            offset=-float(up@hit)
            require(.01<=offset<=.6,'Measured foot offset outside robot bounds')
            qualified=self.domain.runtime_qualified(('tof','tof_extrinsics','attitude','velocity','gyro'))
            require(qualified or self.diagnostic_only,'Independent ToF/rig/attitude/stationarity qualification missing')
            if self.epoch!=s.epoch:self.samples.clear();self.epoch=s.epoch
            if self.samples:
                require(0<s.stamp_ns-self.samples[-1][0]<=150000000,'Duplicate/reordered/discontinuous stationary range window')
                require(abs(offset-np.median([x[1] for x in self.samples]))<=.01,'Unstable local range/foot offset')
            self.samples.append((s.stamp_ns,offset,error));self.last_rejection=None;return True
        except (RelativeContractError,TypeError,OverflowError) as e:
            self.samples.clear();self.last_rejection=str(e);return False

    def result(self,now_ns):
        if len(self.samples)<20 or self.samples[-1][0]-self.samples[0][0]<1000000000 or not 0<=now_ns-self.samples[-1][0]<=150000000:return None
        offsets=np.array([x[1] for x in self.samples]);offset=float(np.median(offsets))
        error=max(x[2] for x in self.samples)+float(np.max(abs(offsets-offset)))
        if np.ptp(offsets)>.012 or error>.025:return None
        return FootOffsetCalibration(offset,error,self.epoch,self.domain.calibration_id,self.samples[0][0],self.samples[-1][0],len(self.samples),
            self.domain.runtime_qualified(('tof','tof_extrinsics','attitude','velocity','gyro')),self.domain.qualification_record_sha256)

    def diagnostics(self,now_ns):
        return {'sample_count':len(self.samples),'span_s':0. if not self.samples else (self.samples[-1][0]-self.samples[0][0])*1e-9,
                'latest_age_s':None if not self.samples else (now_ns-self.samples[-1][0])*1e-9,'last_rejection':self.last_rejection,
                'offset_ready':self.result(now_ns) is not None,'global_position_used':False,'home_anchor_created':False}


@dataclass(frozen=True)
class CurrentPlane:
    stamp_ns: int
    epoch: str
    calibration_id: str
    center_camera_m: tuple
    normal_camera: tuple
    source: str='single-current-raw-depth-plane'


@dataclass(frozen=True)
class RelativeSupport:
    source_stamp_ns: int
    reference_stamp_ns: int
    epoch: str
    calibration_id: str
    camera_profile_id: str
    center_body_m: tuple
    normal_body: tuple
    up_body: tuple
    corners_body_m: tuple
    requested_halfwidth_m: float
    safe_halfwidth_m: float
    local_error_m: float
    normal_error_rad: float
    direct_pixels: int
    max_plane_residual_m: float
    independent_spacing_m: float
    source_region_sha256: str
    qualification_record_sha256: str
    runtime_qualified: bool
    max_age_ns: int
    static_environment: bool=False
    transported: bool=False
    motion_qualification_records: tuple=()
    source: str='strict direct native stereo in exposure BODY frame'


def build_relative_support(depth,measured_mask,K,T_body_from_camera,plane,domain,*,image_stamp_ns,depth_stamp_ns,up_stamp_ns,up_body,
                           now_ns,required_halfwidth_m,actual_profile_id,coverage_margin_m=.04,max_support_age_s=.25,static_environment=False,diagnostic_only=False):
    domain.validate(stereo=True)
    require(actual_profile_id==domain.camera_profile_id,'Running matcher/confidence/rectification profile differs from qualified profile')
    require(isinstance(plane,CurrentPlane) and plane.source=='single-current-raw-depth-plane','Current raw plane required; tracked/averaged world plane is not a same-frame measurement')
    require(image_stamp_ns==depth_stamp_ns==up_stamp_ns==plane.stamp_ns,'Exact same exposure required for image/depth/plane/BODY attitude')
    require(0<=now_ns-depth_stamp_ns<=150000000,'Stale/future raw depth')
    require(plane.calibration_id==domain.calibration_id and bool(plane.epoch),'Plane/calibration/epoch mismatch')
    T=rigid(T_body_from_camera);K=np.asarray(K,float);d=np.asarray(depth)
    require(d.ndim==2 and d.shape==(domain.resolution[1],domain.resolution[0]) and K.shape==(4,) and np.allclose(K,domain.intrinsics,rtol=0,atol=1e-9),'Wrong actual calibration/profile/grid')
    require(digest_matrix(T)==domain.body_from_camera_sha256,'Actual T_BODY_FROM_CAMERA differs from qualified extrinsics')
    provenance=np.asarray(measured_mask)
    require(provenance.shape==d.shape and (provenance.dtype==np.bool_ or (provenance.dtype==np.uint8 and np.isin(provenance,[0,1]).all())),'Explicit boolean/0-1 native direct provenance mask required')
    require(np.isfinite(K).all() and min(K[:2])>0 and d.dtype.kind=='f','Metric floating depth and positive actual intrinsics required')
    require(.01<required_halfwidth_m<=1.5 and .02<=coverage_margin_m<=.08 and 0<max_support_age_s<=10.,'Invalid bounded coverage/retention request')
    require(max_support_age_s<=.25 or static_environment,'Historical static-support retention must be declared; never a dynamic obstacle guarantee')
    up=unit(up_body);c=T[:3,:3]@vec(plane.center_camera_m)+T[:3,3];n=T[:3,:3]@unit(plane.normal_camera)
    if n@up<0:n=-n
    normal_error=domain.camera_rotation_error_rad+domain.plane_normal_error_rad
    require(math.acos(float(np.clip(n@up,-1,1)))+normal_error+domain.up_error_rad<=math.radians(5),'Support tilt/error interval not horizontal')
    seed=np.array([1.,0,0]) if abs(up[0])<.9 else np.array([0.,1,0])
    u=seed-up*(seed@up);u/=np.linalg.norm(u);v=np.cross(up,u)
    half=required_halfwidth_m+coverage_margin_m
    offsets=np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    tangents=offsets[:,0,None]*u+offsets[:,1,None]*v
    corners=c+tangents-((tangents@n)/(up@n))[:,None]*up
    cam=(corners-T[:3,3])@T[:3,:3]
    require((cam[:,2]>.25).all(),'Complete support footprint behind/too near camera')
    uv=np.column_stack((cam[:,0]/cam[:,2]*K[0]+K[2],cam[:,1]/cam[:,2]*K[1]+K[3]))
    h,w=d.shape
    require((uv[:,0]>=2).all() and (uv[:,0]<=w-3).all() and (uv[:,1]>=2).all() and (uv[:,1]<=h-3).all(),'Complete support footprint outside image; no clipped certification')
    roi=np.zeros(d.shape,np.uint8);cv2.fillConvexPoly(roi,np.rint(uv).astype(np.int32),1);yy,xx=np.nonzero(roi)
    require(64<=len(xx)<=250000,'Native support pixel budget exceeded/insufficient')
    z=d[yy,xx];direct=np.asarray(measured_mask,bool)[yy,xx]
    require(direct.all() and np.isfinite(z).all(),'Unknown/hole/inferred native pixel: no filling or completion')
    dz=domain.optical_z_error_m
    require((z-dz>=domain.depth_interval_m[0]).all() and (z+dz<=domain.depth_interval_m[1]).all(),'Measurements outside independently qualified depth domain')
    rays=np.column_stack(((xx-K[2])/K[0],(yy-K[3])/K[1],np.ones(len(xx))))
    points_camera=rays*z[:,None];points=points_camera@T[:3,:3].T+T[:3,3]
    residual=float(np.max(abs((points-c)@n)))
    require(residual<=.02,'Foreground/edge/off-plane direct measurement exceeds strict 2 cm shape filter')
    normal_camera=T[:3,:3].T@n
    denominator=abs(normal_camera[2])-math.sin(normal_error)
    require(denominator>.1,'Grazing-angle ray spacing cannot be bounded')
    spacing=float(domain.independent_stride_px*np.max(z+dz)/min(K[:2])/denominator)
    require(spacing<=.01,'Independent measured rays exceed 1 cm spacing; native copies do not count')
    lever=float(max(np.linalg.norm(points_camera,axis=1).max(),np.linalg.norm(corners,axis=1).max()))
    point_error=dz*float(np.linalg.norm(rays,axis=1).max())+domain.camera_translation_error_m+2*lever*math.sin(domain.camera_rotation_error_rad/2)
    local_error=point_error+residual+2*lever*math.sin(domain.plane_normal_error_rad/2)+2*lever*math.sin(domain.up_error_rad/2)
    require(local_error<=.04,'Local BODY-to-support error exceeds unchanged 4 cm limit')
    safe_half=half-local_error
    require(safe_half>=required_halfwidth_m,'Coverage does not contain uncertainty-eroded whole footprint')
    qualified=domain.runtime_qualified(('stereo','camera_extrinsics','attitude','plane_geometry'))
    require(qualified or diagnostic_only,'Independent sensor/rig/plane qualification missing; residual does not qualify accuracy')
    hashed=hashlib.sha256(np.asarray(z,dtype='<f4').tobytes()+np.asarray(direct,np.uint8).tobytes()+np.asarray(xx,dtype='<i4').tobytes()+np.asarray(yy,dtype='<i4').tobytes()).hexdigest()
    return RelativeSupport(depth_stamp_ns,depth_stamp_ns,plane.epoch,domain.calibration_id,domain.camera_profile_id,tuple(c),tuple(n),tuple(up),
        tuple(tuple(x) for x in corners),required_halfwidth_m,safe_half,float(local_error),float(normal_error),len(xx),residual,spacing,hashed,
        domain.qualification_record_sha256,qualified,int(max_support_age_s*1e9),static_environment)


@dataclass(frozen=True)
class RelativeMotionLink:
    source_stamp_ns: int
    target_stamp_ns: int
    epoch: str
    calibration_id: str
    current_body_from_source_body: tuple
    translation_error_m: float
    rotation_error_rad: float
    qualification_record_sha256: str
    qualification_scope: str
    method: str
    source: str='propagated-relative-motion; not two marginal world covariance subtraction'


def transport_support(patch,link,now_ns,diagnostic_only=False):
    require(link.source_stamp_ns==patch.reference_stamp_ns and link.target_stamp_ns>=link.source_stamp_ns,'Relative motion link does not connect exact reference exposures')
    require(link.epoch==patch.epoch and link.calibration_id==patch.calibration_id,'Relative motion reset/rig mismatch')
    require(0<=now_ns-patch.source_stamp_ns<=patch.max_age_ns,'Original measured support expired; transport cannot refresh acquisition time')
    require(now_ns-patch.source_stamp_ns<=250000000 or patch.static_environment,'Static historical support assumption missing')
    require(0<=now_ns-link.target_stamp_ns<=150000000,'Relative motion destination stale/future')
    require(np.isfinite([link.translation_error_m,link.rotation_error_rad]).all() and min(link.translation_error_m,link.rotation_error_rad)>=0 and link.rotation_error_rad<=.1,'Relative motion error missing/invalid')
    require(link.method in ('relative-imu-propagation','joint-vio-cross-covariance','bounded-stationary-motion'),'Two marginal global covariances do not define a relative motion error')
    T=rigid(link.current_body_from_source_body);corners=np.asarray(patch.corners_body_m)
    delta=link.translation_error_m+2*float(np.linalg.norm(corners,axis=1).max())*math.sin(link.rotation_error_rad/2)
    require(patch.local_error_m+delta<=.04,'Accumulated relative motion error exceeds local 4 cm gate')
    safe=patch.safe_halfwidth_m-delta
    require(safe>=patch.requested_halfwidth_m,'Relative drift erodes verified footprint edge')
    qualified=bool(link.qualification_scope=='independent-relative-motion-bench' and re.fullmatch('[0-9a-f]{64}',link.qualification_record_sha256 or ''))
    require(qualified or diagnostic_only,'Qualified relative IMU/VIO motion proof required for historical support')
    c=T[:3,:3]@vec(patch.center_body_m)+T[:3,3];n=T[:3,:3]@unit(patch.normal_body);up=T[:3,:3]@unit(patch.up_body)
    transformed=corners@T[:3,:3].T+T[:3,3]
    return replace(patch,reference_stamp_ns=link.target_stamp_ns,center_body_m=tuple(c),normal_body=tuple(n),up_body=tuple(up),
                   corners_body_m=tuple(tuple(x) for x in transformed),safe_halfwidth_m=safe,local_error_m=patch.local_error_m+delta,
                   normal_error_rad=patch.normal_error_rad+link.rotation_error_rad,runtime_qualified=patch.runtime_qualified and qualified,transported=True,
                   motion_qualification_records=patch.motion_qualification_records+(link.qualification_record_sha256,))


@dataclass(frozen=True)
class GlobalMapVeto:
    stamp_ns: int
    reference_stamp_ns: int
    epoch: str
    source_region_sha256: str
    allows_contact_sweep: bool
    full_upper_side_sweep_checked: bool
    unknown_blocked: bool
    reference_registration_qualified: bool
    certificate_id: str
    mode: str=''
    body_goal_m: tuple=()
    local_goal_error_m: float=float('nan')
    sweep_radius_m: float=0.


def assess_relative_contact(patch,sample,foot,domain,tof_in_body,tof_axis_body,now_ns,*,global_map_veto=None,mode='descent',liftoff_height_m=None):
    require(mode in ('descent','liftoff'),'Only bounded contact descent/liftoff geometry')
    up,vel,gyro,hit,tof_vertical_error,tof_point_error=tof_geometry(sample,domain,tof_in_body,tof_axis_body,now_ns)
    require(sample.stamp_ns==patch.reference_stamp_ns,'Historical support needs an explicit qualified relative-motion link, not nearest timestamps')
    require(sample.epoch==patch.epoch==foot.epoch and sample.calibration_id==patch.calibration_id==foot.calibration_id,'Contact frame/robot epoch mismatch')
    require(foot.qualification_record_sha256==domain.qualification_record_sha256==patch.qualification_record_sha256 and patch.camera_profile_id==domain.camera_profile_id,'Qualification/profile changed since proof acquisition')
    require(.01<=foot.offset_m<=.6 and 0<foot.error_m<=.025 and foot.sample_count>=20 and foot.last_stamp_ns-foot.first_stamp_ns>=1000000000 and foot.last_stamp_ns<=sample.stamp_ns,'Stable independent foot offset calibration required')
    require(0<=now_ns-patch.source_stamp_ns<=patch.max_age_ns,'Original raw support stale')
    require(math.acos(float(np.clip(up[2],-1,1)))+domain.up_error_rad<=math.radians(3),'Contact upright tilt/error limit')
    require(np.linalg.norm(gyro)+domain.gyro_error_radps<=.2,'Contact angular-rate/error limit')
    vertical=float(vel@up);lateral=vel-up*vertical
    require(np.linalg.norm(lateral)+domain.speed_error_mps<=.08,'Contact lateral speed/error limit')
    require((-.25+domain.speed_error_mps<=vertical<=.04-domain.speed_error_mps) if mode=='descent' else (-.04+domain.speed_error_mps<=vertical<=.25-domain.speed_error_mps),'Contact vertical speed/error limit')
    require(sample.armed or sample.landed,'Airborne contact without native armed confirmation')
    require(mode!='liftoff' or sample.armed,'Liftoff needs actual native arming; this module never arms')
    c=vec(patch.center_body_m);n=unit(patch.normal_body)
    require(math.acos(float(np.clip(unit(patch.up_body)@up,-1,1)))<=domain.up_error_rad+patch.normal_error_rad,'Relative motion/gravity direction inconsistent with current attitude')
    require(math.acos(float(np.clip(n@up,-1,1)))+patch.normal_error_rad+domain.up_error_rad<=math.radians(5),'Transported plane tilt/error limit')
    lateral_center=c-up*(c@up)
    require(np.linalg.norm(lateral_center)+patch.local_error_m<=.04,'BODY-to-pad center interval exceeds unchanged 4 cm drift gate')
    require(np.linalg.norm(lateral_center)+patch.requested_halfwidth_m<=patch.safe_halfwidth_m,'Current whole footprint leaves uncertainty-eroded measured coverage')
    residual=float(n@(hit-c));agreement_error=patch.local_error_m+tof_point_error
    require(abs(residual)<=agreement_error,'Actual ToF hit disagrees with same-frame direct support')
    clearance=-float(up@hit)-foot.offset_m*up[2]
    clearance_error=tof_vertical_error+foot.error_m+2*foot.offset_m*math.sin(domain.up_error_rad/2)
    require(clearance_error<=.04 and -.015-clearance_error<=clearance<=.5,'Local foot-clearance interval invalid/insufficient precision')
    plane_clearance=float(n@(np.array([0.,0.,-foot.offset_m])-c)/(n@up))
    require(abs(plane_clearance-clearance)<=agreement_error+clearance_error,'Plane and ToF foot clearances disagree')
    if mode=='liftoff':
        require(liftoff_height_m is not None and .01<liftoff_height_m<=.5,'Explicit bounded relative liftoff height required')
        target_height=float(liftoff_height_m)
        require(target_height-plane_clearance>=-.015,'Liftoff target is below current foot clearance')
    else:target_height=0.
    body_goal=up*(target_height-plane_clearance)
    goal_error=(patch.local_error_m+foot.error_m)/math.cos(math.radians(5))+2*(np.linalg.norm(c)+foot.offset_m+target_height)*math.sin((patch.normal_error_rad+domain.up_error_rad)/2)
    require(goal_error<=.04,'Combined relative contact-goal interval exceeds unchanged local 4 cm limit')
    local_qualified=patch.runtime_qualified and foot.runtime_qualified and domain.runtime_qualified(('tof','tof_extrinsics','attitude','velocity','gyro'))
    veto=global_map_veto
    map_allows=bool(veto is not None and veto.mode==mode and np.asarray(veto.body_goal_m).shape==(3,) and
                    np.isfinite(veto.body_goal_m).all() and np.allclose(veto.body_goal_m,body_goal,rtol=0,atol=1e-6) and
                    math.isfinite(veto.local_goal_error_m) and veto.local_goal_error_m>=goal_error and veto.sweep_radius_m>=patch.requested_halfwidth_m and
                    veto.reference_stamp_ns==patch.reference_stamp_ns and veto.epoch==patch.epoch and veto.source_region_sha256==patch.source_region_sha256 and
                    0<=now_ns-veto.stamp_ns<=150000000 and veto.allows_contact_sweep and veto.full_upper_side_sweep_checked and
                    veto.unknown_blocked and veto.reference_registration_qualified and veto.certificate_id)
    return {'source':'same-exposure/qualified-transport sensor-frame contact geometry','mode':mode,
            'local_geometry_passes':True,'local_precision_qualified':bool(local_qualified),'global_map_allows_contact':map_allows,
            'contact_candidate_eligible':bool(local_qualified and map_allows),'motion_certificate_issued':False,'native_command_issued':False,
            'source_stamp_ns':patch.source_stamp_ns,'reference_stamp_ns':patch.reference_stamp_ns,'epoch':patch.epoch,
            'source_region_sha256':patch.source_region_sha256,'local_support_error_m':patch.local_error_m,
            'tof_plane_residual_m':residual,'tof_plane_agreement_error_m':agreement_error,'foot_clearance_m':clearance,
            'foot_clearance_error_m':clearance_error,'plane_foot_clearance_m':plane_clearance,
            'body_goal_m':body_goal.tolist(),'local_goal_error_m':float(goal_error),'requested_sweep_radius_m':patch.requested_halfwidth_m,
            'global_translation_input_used':False,'historical_world_plane_used':False,
            'requires_root_native_controller_and_causal_global_certificate':True}


def existing_foot_calibrator_diagnostics(calibrator,now):
    """Read-only explanation for the legacy gauge-free offset calibrator."""
    samples=list(calibrator.samples);count=len(samples)
    result={'sample_count':count,'span_s':0.,'latest_age_s':None,'last_rejection':calibrator.last_rejection,
            'global_position_error_gate_used':False,'ready':False,'reason':None,
            'declared_range_error_m':float(calibrator.range_error),'declared_extrinsic_error_m':float(calibrator.extrinsic_error),
            'declared_error_sum_m':float(calibrator.range_error+calibrator.extrinsic_error),'admission_error_cap_m':.025,
            'minimum_samples':calibrator.min_samples,'minimum_span_s':calibrator.min_span,
            'offset_spread_cap_m':.012,'computed_offset_error_m':None,'observed_offset_max_deviation_m':None,
            'offset_spread_m':None,'diagnostic_only_median_offset_m':None,'diagnostic_median_is_calibration':False,
            'diagnostic_median_not_accepted':True,'numeric_admission_is_sensor_accuracy_qualification':False}
    if not samples:result['reason']=calibrator.last_rejection or 'no synchronized qualifying RangePose ingested';return result
    span=samples[-1][0]-samples[0][0];age=now-samples[-1][0]
    offsets=np.array([x[1] for x in samples]);center=float(np.median(offsets))
    error=calibrator.range_error+calibrator.extrinsic_error+float(np.max(abs(offsets-center)))
    result.update(span_s=float(span),latest_age_s=float(age),offset_spread_m=float(np.ptp(offsets)),computed_offset_error_m=float(error),
                  observed_offset_max_deviation_m=float(np.max(abs(offsets-center))),diagnostic_only_median_offset_m=center)
    if count<calibrator.min_samples:reason='insufficient continuous samples'
    elif span<calibrator.min_span:reason='continuous sample span too short'
    elif not -1e-9<=age<=.15:reason='latest accepted range window stale/future'
    elif np.ptp(offsets)>.012:reason='foot offset spread exceeds 12 mm'
    elif error>.025:reason='range/extrinsic/spread error exceeds 25 mm'
    else:reason=None
    result['ready']=reason is None;result['diagnostic_median_not_accepted']=reason is not None;result['reason']=reason;return result
