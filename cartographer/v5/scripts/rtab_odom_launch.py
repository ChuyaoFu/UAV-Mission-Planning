#!/usr/bin/python3
"""Optional independent measured stereo/IMU odometry, bounded feature map."""
import os
from launch import LaunchDescription,LaunchService
from launch_ros.actions import Node
p={'use_sim_time':os.environ.get('USE_SIM_TIME','1')=='1','frame_id':'vio_imu','odom_frame_id':'rtab_odom','publish_tf':False,'approx_sync':False,'wait_imu_to_init':True,'qos':2,'qos_camera_info':1,'qos_imu':2,'topic_queue_size':2,'sync_queue_size':3,'imu_queue_size':300,'Odom/Strategy':'0','Odom/GuessSmoothingDelay':'0.5','Odom/ImuGravity':'0','Vis/FeatureType':'6','Vis/MaxFeatures':'600','Vis/MinInliers':'12','GFTT/MinDistance':'5','Vis/CorNNDR':'0.7','Vis/CorGuessWinSize':'30','OdomF2M/MaxSize':'2000','Odom/ResetCountdown':'0','publish_null_when_lost':True,'publish_compressed_sensor_data':False,'publish_sensor_data':False}
s=LaunchService();s.include_launch_description(LaunchDescription([Node(package='rtabmap_odom',executable='stereo_odometry',name='measured_stereo_odometry',parameters=[p],remappings=[('left/image_rect','/rtab/left/image_rect'),('right/image_rect','/rtab/right/image_rect'),('left/camera_info','/rtab/left/camera_info'),('right/camera_info','/rtab/right/camera_info'),('imu','/vio/imu'),('odom','/rtab/odom'),('odom_info','/rtab/odom_info')],output='screen')]));s.run()
