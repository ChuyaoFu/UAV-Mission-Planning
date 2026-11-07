"""Portable stopping-distance and velocity-slew limits; no simulator inputs."""
import math
import numpy as np

def stopping_distance(speed,latency,deceleration=1.,uncertainty=.03):
 if min(speed,latency,uncertainty)<0 or deceleration<=0:raise ValueError('Invalid motion bound')
 return speed*latency+speed*speed/(2*deceleration)+uncertainty

def safe_speed(clearance,latency,deceleration=1.,uncertainty=.03):
 available=max(0.,clearance-uncertainty)
 return max(0.,math.sqrt((deceleration*latency)**2+2*deceleration*available)-deceleration*latency)

def slew_velocity(previous,desired,dt,acceleration=1.):
 delta=np.asarray(desired)-previous;length=np.linalg.norm(delta)
 return previous+delta*min(1.,acceleration*max(0.,dt)/max(length,1e-9))


def visibility_speed_factor(angle,half_fov=.7):
 """Keep the velocity direction inside the measured forward stereo cone."""
 stop=max(.1,half_fov-.15);full=min(.15,stop*.5)
 return float(np.clip((stop-abs(angle))/max(stop-full,1e-6),0.,1.))
