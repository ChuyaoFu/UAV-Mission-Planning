"""Bounded static conditional-map support, distinct from control reaction time.

Original exposure stamps remain unchanged. Native map support is withdrawn
early enough that the preceding safe stopping tube has time left to expire.
This policy assumes a static scene and the explicitly bounded motion model.
"""
import math


def policy(configuration):
    legacy = configuration.get('inferred_geometry_max_age_s', .6)
    if not configuration.get('conditional_static_memory', False):
        return dict(enabled=False, lifetime_s=legacy, withdraw_before_expiry_s=0.,
                    reaction_s=max(configuration['max_observation_age_s'], legacy))
    lifetime = float(configuration.get('conditional_support_lifetime_s', 10.))
    reaction = float(configuration.get('conditional_control_latency_s', .35))
    speed = float(configuration.get('conditional_braking_speed_bound_mps', 1.))
    acceleration = float(configuration['max_acceleration_mps2'])
    observation = float(configuration['max_observation_age_s'])
    input_age = float(configuration.get('inferred_geometry_input_max_age_s', .25))
    values = [lifetime, reaction, speed, acceleration, observation, input_age]
    if not all(math.isfinite(v) for v in values) or not 1. < lifetime <= 10. or not .05 <= reaction <= .5 or not 0 < acceleration or not configuration['max_speed_mps'] <= speed <= 2.:
        raise ValueError('Invalid bounded static conditional-memory policy')
    reserve = observation + reaction + speed / acceleration
    if not 0 < input_age <= .5 or lifetime <= reserve + input_age:
        raise ValueError('Support lifetime cannot cover delivery and complete braking reserve')
    return dict(enabled=True, lifetime_s=lifetime, withdraw_before_expiry_s=reserve,
                reaction_s=reaction, braking_speed_bound_mps=speed,
                raw_and_map_age_bound_s=observation,
                assumption='static scene; declared motion/latency bounds; original stamps retained')
