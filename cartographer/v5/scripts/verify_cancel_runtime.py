#!/usr/bin/python3
"""Isolated ROS signal regression; creates no flight/arming commands."""
import os,signal,json,tempfile
from types import SimpleNamespace
import rclpy
from rclpy.signals import SignalHandlerOptions
from px4_msgs.msg import VehicleStatus
from navigate import Navigator,install_stop_handlers
assert os.environ.get('ROS_DOMAIN_ID')=='43'
rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
with tempfile.TemporaryDirectory() as directory:
 n=Navigator(directory+'/cancel.json');n.status=SimpleNamespace(arming_state=VehicleStatus.ARMING_STATE_DISARMED);install_stop_handlers(n)
 try:
  for value in [signal.SIGINT,signal.SIGTERM]:
   n.cancelled=False;os.kill(os.getpid(),value);assert rclpy.ok(),'Signal shut ROS down before landing cleanup'
   try:n.check()
   except RuntimeError as error:assert 'controlled landing requested' in str(error)
   else:raise AssertionError('Cancellation ignored')
  n.finish();result=json.load(open(n.output));assert result['disarmed'] and not result['controller_complete'];assert rclpy.ok()
  print(json.dumps({'passed':True,'sigint_sigterm_cooperative':True,'context_alive_through_cleanup':True,'arming_commands_sent':0},indent=2))
 finally:n.destroy_node();rclpy.shutdown()
