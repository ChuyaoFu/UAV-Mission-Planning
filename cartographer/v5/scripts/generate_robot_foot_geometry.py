#!/usr/bin/env python3
"""ROBOT SDF only. Never reads course/world/trajectory data. SITL definition only."""
import argparse,hashlib,json,math,itertools,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def transform(element):
    pose=element.find('pose');a=[0.]*6 if pose is None else [float(v) for v in pose.text.split()]
    if len(a)!=6 or (pose is not None and pose.get('relative_to') not in (None,'base_link')):raise ValueError('Unsupported ROBOT frame transform')
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',a[3:]).as_matrix();T[:3,3]=a[:3];return T


def generate(path,point_error_m=.002,tilt_limit_deg=5.):
    if not math.isfinite(point_error_m) or not 0<point_error_m<=.01 or not math.isfinite(tilt_limit_deg) or not 0<tilt_limit_deg<=5:raise ValueError('Explicit positive robot geometry bound and <=5deg cone required')
    data=Path(path).read_bytes();root=ET.fromstring(data);model=root.find('./model')
    if model is None or model.find("link[@name='base_link']") is None or model.findtext('static','false')=='true':raise ValueError('Actual non-static ROBOT model with base_link required; no course/world input')
    if not np.allclose(transform(model),np.eye(4)) or not np.allclose(transform(model.find("link[@name='base_link']")),np.eye(4)):raise ValueError('Explicit BODY/model calibration needed for nonidentity origin')
    theta=math.radians(tilt_limit_deg);feet=[];other=[];skids=[]
    for link in model.findall('link'):
        TL=transform(link)
        for collision in link.findall('collision'):
            TC=TL@transform(collision);geometry=collision.find('geometry');box=geometry.find('box');cylinder=geometry.find('cylinder')
            is_foot=link.get('name')=='base_link' and collision.get('name','').startswith('skid_')
            if box is not None:
                size=np.array([float(v) for v in box.findtext('size').split()]);vertices=np.array(list(itertools.product(*[[-v/2,v/2] for v in size])))
            elif cylinder is not None:
                radius=float(cylinder.findtext('radius'));length=float(cylinder.findtext('length'))
                # Bounding prism contains the whole cylinder and is conservative for other-part exclusion.
                vertices=np.array(list(itertools.product([-radius,radius],[-radius,radius],[-length/2,length/2])))
            else:raise ValueError('Every robot collision must be explicitly bounded; unknown shape refused')
            world=vertices@TC[:3,:3].T+TC[:3,3]
            if is_foot:
                if box is None or not np.allclose(TC[:3,:3],[ [1,0,0],[0,1,0],[0,0,1] ]):raise ValueError('Minimal generator requires axis-aligned skid boxes')
                feet.extend(world[np.isclose(world[:,2],world[:,2].min())].tolist());skids.append(collision.get('name'))
            else:
                bounds=[]
                for v in world:
                    a=-v[2];rho=float(np.linalg.norm(v[:2]));t=min(theta,math.atan2(rho,a))
                    bounds.append(float(a*math.cos(t)+rho*math.sin(t)))
                other.append({'collision':link.get('name')+'::'+collision.get('name'),'downward_support_upper_m':max(bounds)})
    if len(skids)!=2:raise ValueError('Exactly two actual skid collision boxes required')
    points=sorted(set(tuple(p) for p in feet))
    payload=dict(schema='robot-mechanical-foot-v1',body_frame='base_link',source_scope='sitl-robot-model-definition',
        source_model_sha256=hashlib.sha256(data).hexdigest(),source_robot_model_name=model.get('name'),
        point_error_metric='euclidean-per-contact-point-bound',point_error_m=point_error_m,contact_tilt_limit_rad=theta,
        contact_set_complete=True,other_collision_geometry_checked=True,foot_collision_names=skids,
        foot_contact_points_body_m=[list(p) for p in points],other_collision_downward_support_upper_m=max(p['downward_support_upper_m'] for p in other),
        other_collision_support_audit=other,hardware_use_forbidden=True)
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(payload=payload,calibration_digest=digest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot-model',required=True);p.add_argument('--output',required=True);p.add_argument('--point-error-m',type=float,default=.002);a=p.parse_args()
    result=generate(a.robot_model,a.point_error_m);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result['calibration_digest'])

if __name__=='__main__':main()
