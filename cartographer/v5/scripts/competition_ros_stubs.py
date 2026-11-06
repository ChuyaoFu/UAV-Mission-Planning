"""EXPLICIT UNIT-TEST ONLY ROS/PX4 stubs; never imported by production adapter.

No DDS participant, sockets, simulator, PX4 process, or rclpy executor is created.
Tests still import and exercise the copied REAL navigate.Navigator methods.
"""
import sys,types,copy
from types import SimpleNamespace as NS


def message_class(name,defaults=None,constants=None):
    class Message:
        def __init__(self,**kwargs):
            for key,value in (defaults or {}).items():setattr(self,key,value() if callable(value) else copy.deepcopy(value))
            self.__dict__.update(kwargs)
    Message.__name__=name
    for key,value in (constants or {}).items():setattr(Message,key,value)
    return Message


def install():
    if 'rclpy' in sys.modules and not getattr(sys.modules['rclpy'],'EXPLICIT_TEST_STUB',False):
        raise RuntimeError('Do not replace a real imported rclpy process; run these cases in their isolated subprocess')
    stamp=lambda:NS(sec=0,nanosec=0)
    header=lambda:NS(stamp=stamp(),frame_id='world')
    point=lambda:NS(x=0.,y=0.,z=0.)
    quat=lambda:NS(x=0.,y=0.,z=0.,w=1.)
    pose=lambda:NS(position=point(),orientation=quat())
    Header=message_class('Header',{'stamp':stamp,'frame_id':'world'})
    PoseStamped=message_class('PoseStamped',{'header':header,'pose':pose})
    Odometry=message_class('Odometry',{'header':header,'child_frame_id':'base_link',
        'pose':lambda:NS(pose=pose(),covariance=[0.]*36),
        'twist':lambda:NS(twist=NS(linear=point(),angular=point()),covariance=[0.]*36)})
    VehicleStatus=message_class('VehicleStatus',{'timestamp':1,'arming_state':1,'nav_state':14,'pre_flight_checks_pass':True},
        {'ARMING_STATE_DISARMED':1,'ARMING_STATE_ARMED':2,'NAVIGATION_STATE_OFFBOARD':14,
         'NAVIGATION_STATE_AUTO_LAND':18,'NAVIGATION_STATE_AUTO_LOITER':4})
    VehicleLocalPosition=message_class('VehicleLocalPosition',{'timestamp':1,'xy_valid':True,'z_valid':True,'x':0.,'y':0.,'z':0.,'heading':0.})
    VehicleLandDetected=message_class('VehicleLandDetected',{'timestamp':1,'landed':True,'freefall':False,'at_rest':True})
    VehicleCommand=message_class('VehicleCommand',{}, {'VEHICLE_CMD_DO_SET_MODE':176,'VEHICLE_CMD_COMPONENT_ARM_DISARM':400,'VEHICLE_CMD_NAV_LAND':21})
    VehicleCommandAck=message_class('VehicleCommandAck',{'timestamp':1,'command':0,'result':0,'target_system':250},
        {'VEHICLE_CMD_RESULT_ACCEPTED':0,'VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED':1,
         'VEHICLE_CMD_RESULT_DENIED':2,'VEHICLE_CMD_RESULT_UNSUPPORTED':3,'VEHICLE_CMD_RESULT_FAILED':4,
         'VEHICLE_CMD_RESULT_IN_PROGRESS':5,'VEHICLE_CMD_RESULT_CANCELLED':6})
    Image=message_class('Image',{'header':header,'encoding':'mono8','width':0,'height':0,'step':0,'data':b'','is_bigendian':False})
    Range=message_class('Range',{'header':header,'range':.1,'min_range':.05,'max_range':4.})
    class Publisher:
        def __init__(self,topic):self.topic=topic;self.messages=[]
        def publish(self,message):self.messages.append(copy.deepcopy(message))
    class Node:
        def __init__(self,*args,**kwargs):self.test_time=0.;self.publishers={};self.subscriptions={};self.logs=[]
        def get_clock(self):
            node=self
            class Clock:
                def now(self):
                    ns=int(round(node.test_time*1e9))
                    return NS(nanoseconds=ns,to_msg=lambda:NS(sec=ns//1000000000,nanosec=ns%1000000000))
            return Clock()
        def create_publisher(self,type,topic,qos):
            p=Publisher(topic);self.publishers[topic]=p;return p
        def create_subscription(self,type,topic,callback,qos):
            self.subscriptions.setdefault(topic,[]).append(callback);return NS(topic=topic,callback=callback)
        def get_logger(self):return NS(info=lambda m:self.logs.append(m),warning=lambda m:self.logs.append(m),error=lambda m:self.logs.append(m))
        def destroy_node(self):pass
    modules={}
    def module(name,**attrs):
        m=types.ModuleType(name);m.__dict__.update(attrs);modules[name]=m;return m
    r=module('rclpy',EXPLICIT_TEST_STUB=True,spin_once=lambda *a,**k:None,ok=lambda:True,init=lambda **kw:None,shutdown=lambda:None)
    module('rclpy.node',Node=Node);module('rclpy.parameter',Parameter=message_class('Parameter'))
    # Parameter accepts a positional name in the actual Navigator constructor.
    class Parameter:
        def __init__(self,*args,**kwargs):pass
    modules['rclpy.parameter'].Parameter=Parameter
    module('rclpy.qos',qos_profile_sensor_data=object(),QoSProfile=message_class('QoSProfile'),DurabilityPolicy=NS(TRANSIENT_LOCAL=1),ReliabilityPolicy=NS(RELIABLE=1,BEST_EFFORT=0))
    module('rclpy.signals',SignalHandlerOptions=NS(NO=0))
    for package in ('std_msgs','nav_msgs','sensor_msgs','geometry_msgs','px4_msgs','quadrotor_msgs'):module(package)
    module('std_msgs.msg',Bool=message_class('Bool',{'data':False}),String=message_class('String',{'data':''}),
           Int8MultiArray=message_class('Int8MultiArray',{'data':list}),Float64MultiArray=message_class('Float64MultiArray',{'data':list}))
    module('nav_msgs.msg',Odometry=Odometry)
    module('sensor_msgs.msg',Image=Image,Range=Range)
    module('geometry_msgs.msg',PoseStamped=PoseStamped)
    module('px4_msgs.msg',VehicleStatus=VehicleStatus,VehicleLocalPosition=VehicleLocalPosition,VehicleLandDetected=VehicleLandDetected,
           VehicleCommand=VehicleCommand,VehicleCommandAck=VehicleCommandAck,
           OffboardControlMode=message_class('OffboardControlMode'),TrajectorySetpoint=message_class('TrajectorySetpoint'))
    module('quadrotor_msgs.msg',PositionCommand=message_class('PositionCommand',{'header':header,'trajectory_id':1,'position':point,'velocity':point}))
    sys.modules.update(modules)
    return NS(Header=Header,PoseStamped=PoseStamped,Odometry=Odometry,Image=Image,Range=Range,
              VehicleStatus=VehicleStatus,VehicleLocalPosition=VehicleLocalPosition,VehicleLandDetected=VehicleLandDetected,
              VehicleCommand=VehicleCommand,VehicleCommandAck=VehicleCommandAck)
