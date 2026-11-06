"""Bounded native-FCU yaw requests; independent of the VIO/map gauge."""
import math


def wrap(value):return math.atan2(math.sin(value),math.cos(value))


class YawRequestLimiter:
    def __init__(self,rate_rad_s,lead_rad):
        if not (0<rate_rad_s<=math.pi and 0<lead_rad<math.pi/2):raise ValueError('Invalid yaw limits')
        self.rate=rate_rad_s;self.lead=lead_rad;self.value=None;self.stamp=None

    def update(self,desired,actual,stamp):
        if not all(math.isfinite(v) for v in [desired,actual,stamp]):raise ValueError('Nonfinite yaw input')
        if self.value is None or stamp<self.stamp:
            self.value=wrap(actual);self.stamp=stamp
        dt=min(.2,max(0.,stamp-self.stamp));self.stamp=stamp
        # If native heading changes discontinuously, follow its coordinate
        # frame near the actual attitude; never retain a large stale yaw error.
        self.value=wrap(actual+max(-self.lead,min(self.lead,wrap(self.value-actual))))
        step=max(-self.rate*dt,min(self.rate*dt,wrap(desired-self.value)))
        value=wrap(self.value+step)
        self.value=wrap(actual+max(-self.lead,min(self.lead,wrap(value-actual))))
        return self.value
