#!/usr/bin/python3
"""Read-only processing budget and observation audit of an ORIGINAL MCAP.
No ROS publishers, motor commands, truth subscriptions or CAD dependencies.
"""
import sys,json
from pathlib import Path
import numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag=Path(sys.argv[1]);output=Path(sys.argv[2]);r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));wanted=['/stereo/health','/carto/depth_health','/mapping/health','/carto/localization_health','/fmu/in/trajectory_setpoint'];types={t.name:get_message(t.type) for t in r.get_all_topics_and_types() if t.name in wanted};r.set_filter(rosbag2_py.StorageFilter(topics=list(types)));rows={t:[] for t in types}
while r.has_next():
 topic,data,ts=r.read_next();m=deserialize_message(data,types[topic])
 if hasattr(m,'data'):
  try:
   value=json.loads(m.data);value['_recorded_age_s']=ts*1e-9-value.get('stamp',ts*1e-9);rows[topic].append(value)
  except (TypeError,json.JSONDecodeError):pass
 else:rows[topic].append({'stamp':m.timestamp*1e-6,'velocity':list(m.velocity)})
def stats(v):return {'n':len(v),'median':float(np.median(v)),'p95':float(np.percentile(v,95)),'max':float(np.max(v))} if len(v) else None
h=rows.get('/stereo/health',[]);mapping=rows.get('/mapping/health',[]);health=rows.get('/carto/localization_health',[]);vel=rows.get('/fmu/in/trajectory_setpoint',[]);out={'stereo_compute_ms':stats([x['compute_ms'] for x in h if 'compute_ms' in x]),'stereo_end_to_end_age_s':stats([x['_recorded_age_s'] for x in h]),'mapping_publish_age_s':stats([x['_recorded_age_s'] for x in mapping]),'stereo_period_s':stats(np.diff([x['stamp'] for x in h])),'mapping_structure_points':stats([x['structure_points'] for x in mapping]),'mapping_usable_fraction':float(np.mean([x['usable'] for x in mapping])) if mapping else None,'exposure_pose_time_error_s':stats([abs(x['pose_time_error_s']) for x in mapping if 'pose_time_error_s' in x]),'accepted_visual_features':stats([x['accepted_visual_update_features'] for x in health]),'health_unobservable_fraction':float(np.mean([not x['valid'] for x in health])) if health else None,'height_correction_m':stats([x['carto_height_correction_m'] for x in health if 'carto_height_correction_m' in x]),'max_command_speed_mps':max([float(np.linalg.norm(x['velocity'])) for x in vel if np.isfinite(x['velocity']).all()] or [0]),'hardware_timing_validated':False,'note':'Desktop measurements only. Re-run on target Orin with real exposure timestamps, actual rectified calibration, vibration/rolling-shutter and low light. Ages use original MCAP reception timestamps minus exposure stamps; they include recorder scheduling. A ROS simulation-clock read inside a blocking image callback can be stale, so that self-reported field is not used.'};output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
