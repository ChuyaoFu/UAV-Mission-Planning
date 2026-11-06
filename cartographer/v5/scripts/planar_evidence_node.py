#!/usr/bin/env python3
"""Optional ROS2 adapter. No pose/CAD input and no inferred occupied endpoints.

Install beside planar_stereo.py, planar_evidence_runtime.py, camera_geometry.py
and self_mask.py. Run in a separate process with one BLAS/OpenCV CPU thread.
"""
import os
# Before importing numpy: bound BLAS pools as well as OpenCV's worker pool.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import copy
import json
import time
import cv2
import numpy as np
from planar_stereo import PlanarStereo
from planar_evidence_runtime import ExactTripletJoin, LatestOnlyWorker, pack_evidence


def main():
    # Keeping ROS imports here also permits offline import/contract tests.
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    from camera_geometry import load_rig
    from self_mask import self_masks

    class PlanarEvidenceNode(Node):
        def __init__(self):
            super().__init__('planar_free_ray_evidence', parameter_overrides=[Parameter('use_sim_time', value=os.environ.get('USE_SIM_TIME', '1') == '1')])
            self.enabled = bool(self.declare_parameter('enable_inferred_free', False).value)
            self.max_age_s = float(self.declare_parameter('max_age_s', .25).value)
            self.compute_budget_s = float(self.declare_parameter('compute_budget_s', .15).value)
            self.max_hz = float(self.declare_parameter('max_hz', 5.).value)
            if not (0 < self.max_age_s <= .4 and 0 < self.compute_budget_s <= .2 and 0 < self.max_hz <= 5.):
                raise ValueError('Age <=.4s, soft compute budget <=.2s, rate <=5Hz required')
            self.requested_enabled = self.enabled
            cv2.setNumThreads(1)
            self.resource_blocked = self.enabled and cv2.getNumThreads() != 1
            if self.resource_blocked:
                self.enabled = False
                self.get_logger().error('OpenCV did not honor the single-thread limit; inference disabled, measured channel continues')
            self.bridge = CvBridge()
            self.K, _, self.baseline, self.resolution = load_rig()
            if self.resolution != (1280, 800):
                raise ValueError('This adapter accepts only native 1280x800')
            self.masks = self_masks()
            if len(self.masks) != 2 or any(np.asarray(mask).shape != (800, 1280) or not np.isfinite(mask).all() for mask in self.masks):
                raise ValueError('Both native self masks required')
            self.model = PlanarStereo()
            self.joiner = ExactTripletJoin(capacity=6)
            self.epoch = 0
            self.last_submit_stamp = -1
            self.last_submit_mono = float('-inf')
            self.last_publish_stamp = -1
            self.last_inference_stamp = -1
            self.pending_raw = None
            self.last_clock = self.get_clock().now().nanoseconds
            self.last_health = float('-inf')
            self.counts = dict(published=0, stale=0, invalid=0, budget_exceeded=0, clock_resets=0, model_deadline_exhausted=0, measured_only_fallback=0)
            self.last_compute_s = 0.
            self.last_input_age_s = None
            self.last_publish_age_s = None
            self.last_free_fraction = 0.
            self.worker = LatestOnlyWorker(self.compute)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(Image, '/nav/planar_evidence', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
            self.health = self.create_publisher(String, '/nav/planar_evidence_health', 1)
            topics = ('/vio/left/image_raw', '/vio/right/image_raw', '/stereo/depth/image')
            self.subs = [self.create_subscription(Image, topic, lambda msg, i=i: self.receive(i, msg), qos) for i, topic in enumerate(topics)]
            self.timer = self.create_timer(.01, self.publish_ready)

        def clock_ns(self):
            now = self.get_clock().now().nanoseconds
            if now < self.last_clock:
                self.epoch += 1
                self.joiner.reset()
                self.last_submit_stamp = self.last_publish_stamp = -1
                self.last_submit_mono = float('-inf')
                self.last_inference_stamp = -1
                self.pending_raw = None
                self.counts['clock_resets'] += 1
            self.last_clock = now
            return now

        def receive(self, stream, msg):
            now = self.clock_ns()
            key = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            if msg.width != 1280 or msg.height != 800 or msg.encoding != ('mono8' if stream < 2 else '32FC1'):
                self.counts['invalid'] += 1
                return
            if stream == 2:
                self.last_input_age_s = (now - key) * 1e-9
            if stream == 2 and msg.header.frame_id != 'left_optical':
                self.counts['invalid'] += 1
                return
            if not (-.02 <= (now - key) * 1e-9 <= self.max_age_s):
                self.counts['stale'] += 1
                return
            mono = time.monotonic()
            if stream == 2 and key > self.last_submit_stamp and mono - self.last_submit_mono >= 1. / self.max_hz:
                # The raw output clock is independent of stereo matching and the
                # model worker. One raw exposure can wait only to its deadline.
                admission_epoch = self.epoch
                if self.pending_raw is not None:
                    self.emit_pending(None, now, force=True)
                now = self.clock_ns()
                if self.epoch != admission_epoch or not (-.02 <= (now - key) * 1e-9 <= self.max_age_s):
                    self.counts['stale'] += 1
                    return
                mono = time.monotonic()
                reserve = max(0., self.max_age_s - (now - key) * 1e-9 - .01)
                deadline = mono + min(self.compute_budget_s, reserve)
                self.pending_raw = (self.epoch, key, mono, deadline, msg)
                self.last_submit_stamp, self.last_submit_mono = key, mono
            paired = self.joiner.push(stream, key, msg)
            if paired is None or not self.enabled or self.pending_raw is None:
                return
            key, triple = paired
            epoch, raw_key, submitted, deadline, raw = self.pending_raw
            if key != raw_key or key <= self.last_inference_stamp:
                return
            self.last_inference_stamp = key
            self.worker.submit((epoch, key, submitted, deadline, triple))

        def compute(self, item):
            epoch, key, submitted, deadline, (left, right, raw) = item
            measured = self.bridge.imgmsg_to_cv2(raw, desired_encoding='passthrough')
            if time.monotonic() >= deadline - .01:
                return pack_evidence(measured), True
            a = self.bridge.imgmsg_to_cv2(left, desired_encoding='passthrough')
            b = self.bridge.imgmsg_to_cv2(right, desired_encoding='passthrough')
            start = time.monotonic()
            model_deadline = min(deadline - .01, start + max(0., self.compute_budget_s - .01))
            result = self.model.compute(a, measured, self.K, self.baseline, self.masks[0], right=b, right_self_mask=self.masks[1], deadline_monotonic=model_deadline)
            elapsed = time.monotonic() - start
            evidence = pack_evidence(measured, result, enabled=elapsed <= self.compute_budget_s)
            return evidence, bool(result.get('budget_exhausted', False))

        def emit_pending(self, computed, now, force=False):
            if self.pending_raw is None:
                return
            epoch, key, submitted, deadline, raw = self.pending_raw
            mono = time.monotonic()
            if computed is None and self.enabled and not force and mono < deadline:
                return
            self.pending_raw = None
            age = (now - key) * 1e-9
            if epoch != self.epoch or key <= self.last_publish_stamp or not (-.02 <= age <= self.max_age_s) or mono - submitted > self.max_age_s:
                self.counts['stale'] += 1
                return
            try:
                if computed is None or mono >= deadline:
                    measured = self.bridge.imgmsg_to_cv2(raw, desired_encoding='passthrough')
                    evidence = pack_evidence(measured)
                    self.counts['measured_only_fallback'] += int(self.enabled)
                else:
                    evidence = computed
                msg = self.bridge.cv2_to_imgmsg(evidence, encoding='32FC2')
                msg.header = copy.deepcopy(raw.header)
            except Exception as exc:
                self.counts['invalid'] += 1
                self.get_logger().error(repr(exc))
                return
            publish_age = (self.clock_ns() - key) * 1e-9
            if epoch != self.epoch or not (-.02 <= publish_age <= self.max_age_s) or time.monotonic() - submitted > self.max_age_s:
                self.counts['stale'] += 1
                return
            self.pub.publish(msg)
            self.last_publish_age_s = publish_age
            self.last_free_fraction = float(np.isfinite(evidence[..., 1]).mean())
            self.last_publish_stamp = key
            self.counts['published'] += 1

        def publish_ready(self):
            now = self.clock_ns()
            ready = self.worker.take()
            computed, force = None, False
            if ready is not None:
                (epoch, key, submitted, deadline, triple), result, elapsed, error = ready
                self.last_compute_s = elapsed
                self.counts['budget_exceeded'] += int(elapsed > self.compute_budget_s)
                if error is None and result is not None:
                    self.counts['model_deadline_exhausted'] += int(result[1])
                matches_pending = self.pending_raw is not None and self.pending_raw[0] == epoch and self.pending_raw[1] == key
                if error is not None:
                    self.counts['invalid'] += 1
                    self.get_logger().error(error)
                    force = matches_pending
                elif epoch != self.epoch or key <= self.last_publish_stamp or not matches_pending:
                    self.counts['stale'] += 1
                else:
                    computed = result[0]
                    if elapsed > self.compute_budget_s:
                        computed[..., 1] = np.nan
            self.emit_pending(computed, now, force=force)
            mono = time.monotonic()
            if mono - self.last_health >= 1.:
                self.last_health = mono
                report = dict(self.counts, inferred_free_enabled=self.enabled, requested_inferred_free=self.requested_enabled, inference_resource_blocked=self.resource_blocked, raw_output_waits_for_stereo=False, exact_exposure_sync=True, requested_opencv_threads=1, reported_opencv_threads=cv2.getNumThreads(), native_resolution=[1280, 800], channels=['measured_axial_depth_m', 'inferred_free_only_lower_axial_depth_m'], last_compute_s=self.last_compute_s, last_input_age_s=self.last_input_age_s, last_publish_age_s=self.last_publish_age_s, last_free_fraction=self.last_free_fraction, max_hz=self.max_hz, max_age_s=self.max_age_s, soft_compute_budget_s=self.compute_budget_s, input_evicted=self.joiner.evicted, pending_superseded=self.worker.superseded, results_overwritten=self.worker.overwritten_results, worker_errors=self.worker.errors, occupied_endpoints='independent raw measured stream only; channel0 is provenance', target_hardware_qualified=False)
                self.health.publish(String(data=json.dumps(report)))

    rclpy.init()
    node = PlanarEvidenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.worker.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
