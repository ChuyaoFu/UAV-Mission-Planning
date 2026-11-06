"""Offline ROS-adapter transport smoke tests using explicit ROS/CvBridge stubs.

These test adapter control flow, not DDS, cv_bridge binaries or a ROS deployment.
"""
import copy
from types import SimpleNamespace, ModuleType
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import planar_evidence_node as adapter


class AdapterSmokeTests(unittest.TestCase):
    def run_adapter(self, exercise, enabled=False, model=None, budget=.15, reported_threads=1, advance_on_bridge_ns=0):
        modules = {}
        for name in ('rclpy', 'rclpy.node', 'rclpy.parameter', 'rclpy.qos', 'sensor_msgs', 'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg', 'cv_bridge', 'camera_geometry', 'self_mask'):
            modules[name] = ModuleType(name)
        now = [100050000000]
        class Publisher:
            def __init__(self): self.messages = []
            def publish(self, msg): self.messages.append(msg)
        class Node:
            def __init__(self, *args, **kwargs):
                self.test_pubs = {}
                self.test_now = now
            def declare_parameter(self, name, default):
                return SimpleNamespace(value={'enable_inferred_free': enabled, 'compute_budget_s': budget}.get(name, default))
            def get_clock(self): return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now[0]))
            def create_publisher(self, msg_type, topic, qos):
                pub = Publisher(); self.test_pubs[topic] = pub; return pub
            def create_subscription(self, *args): return object()
            def create_timer(self, *args): return object()
            def destroy_node(self): pass
            def get_logger(self): return SimpleNamespace(error=lambda text: None)
        class Bridge:
            def imgmsg_to_cv2(self, msg, **kwargs): return msg.array
            def cv2_to_imgmsg(self, array, encoding):
                now[0] += advance_on_bridge_ns
                return SimpleNamespace(array=array.copy(), encoding=encoding, width=array.shape[1], height=array.shape[0], step=array.shape[1]*8)
        modules['rclpy'].init = lambda: None
        modules['rclpy'].shutdown = lambda: None
        modules['rclpy'].spin = exercise
        modules['rclpy.node'].Node = Node
        modules['rclpy.parameter'].Parameter = lambda *args, **kwargs: object()
        modules['rclpy.qos'].QoSProfile = lambda **kwargs: object()
        modules['rclpy.qos'].ReliabilityPolicy = SimpleNamespace(BEST_EFFORT=0)
        modules['sensor_msgs.msg'].Image = object
        modules['std_msgs.msg'].String = lambda **kwargs: SimpleNamespace(**kwargs)
        modules['cv_bridge'].CvBridge = Bridge
        modules['camera_geometry'].load_rig = lambda: (np.array([760., 760., 640., 400.]), None, .075, (1280, 800))
        modules['self_mask'].self_masks = lambda: (np.zeros((800, 1280), bool), np.zeros((800, 1280), bool))
        class Model:
            def compute(self, *args, **kwargs): return model(*args, **kwargs)
        with patch.dict('sys.modules', modules), patch.object(adapter, 'PlanarStereo', Model), patch.object(adapter.cv2, 'getNumThreads', return_value=reported_threads):
            adapter.main()

    @staticmethod
    def messages(stamp=100000000000):
        result = []
        for stream in range(3):
            array = np.full((800, 1280), 170, np.uint8) if stream < 2 else np.full((800, 1280), np.nan, np.float32)
            if stream == 2: array[100, 200] = 1.
            result.append(SimpleNamespace(width=1280, height=800, encoding='mono8' if stream < 2 else '32FC1', header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp//1000000000, nanosec=stamp%1000000000), frame_id='right_optical' if stream == 1 else 'left_optical'), array=array))
        return result

    def wait_ready(self, node):
        deadline = time.monotonic() + 1.
        while time.monotonic() < deadline:
            with node.worker.condition:
                if node.worker.ready is not None: return
            time.sleep(.001)
        self.fail('adapter worker did not return')

    def test_default_disabled_wire_contract_does_not_wait_for_matching_right(self):
        def exercise(node):
            msgs = self.messages()
            wrong = copy.deepcopy(msgs[1]); wrong.header.stamp.nanosec += 1
            node.receive(0, msgs[0]); node.receive(1, wrong); node.receive(2, msgs[2])
            self.assertIsNone(node.worker.pending)
            node.publish_ready()
            node.receive(1, msgs[1]); node.publish_ready()
            published = node.test_pubs['/nav/planar_evidence'].messages
            self.assertEqual(len(published), 1)
            msg = published[0]
            self.assertEqual((msg.encoding, msg.width, msg.height, msg.step), ('32FC2', 1280, 800, 10240))
            self.assertEqual(msg.header, msgs[2].header)
            np.testing.assert_array_equal(msg.array[..., 0], msgs[2].array)
            self.assertTrue(np.isnan(msg.array[..., 1]).all())
        self.run_adapter(exercise)

    def test_enabled_wire_keeps_inference_in_free_only_channel(self):
        def model(left, measured, *args, **kwargs):
            return dict(inferred_depth=np.full_like(measured, 2.), free_ray_lower_bound=np.full_like(measured, 1.9), depth_interval_m=np.full_like(measured, .1), plane_id=np.zeros_like(measured, np.int16))
        def exercise(node):
            msgs = self.messages()
            for i, msg in enumerate(msgs): node.receive(i, msg)
            self.wait_ready(node); node.publish_ready()
            published = node.test_pubs['/nav/planar_evidence'].messages
            self.assertEqual(len(published), 1)
            np.testing.assert_array_equal(published[0].array[..., 0], msgs[2].array)
            self.assertTrue(np.isnan(published[0].array[100, 200, 1]))
            self.assertAlmostEqual(published[0].array[101, 201, 1], 1.9, places=6)
        self.run_adapter(exercise, enabled=True, model=model)

    def test_enabled_budget_exhaustion_publishes_measured_only(self):
        def model(left, measured, *args, **kwargs):
            time.sleep(.06)
            return dict(inferred_depth=np.full_like(measured, 2.), free_ray_lower_bound=np.full_like(measured, 1.9), depth_interval_m=np.full_like(measured, .1), plane_id=np.zeros_like(measured, np.int16))
        def exercise(node):
            msgs = self.messages()
            for i, msg in enumerate(msgs): node.receive(i, msg)
            self.wait_ready(node); node.publish_ready()
            published = node.test_pubs['/nav/planar_evidence'].messages
            self.assertEqual(len(published), 1)
            self.assertTrue(np.isnan(published[0].array[..., 1]).all())
            self.assertEqual(node.counts['budget_exceeded'], 1)
        self.run_adapter(exercise, enabled=True, model=model, budget=.04)

    def test_stale_future_wrong_encoding_and_frame_are_rejected(self):
        def exercise(node):
            for stamp in (99000000000, 101000000000):
                for i, msg in enumerate(self.messages(stamp)): node.receive(i, msg)
            msgs = self.messages(); msgs[0].encoding = 'rgb8'; msgs[2].header.frame_id = 'world'
            node.receive(0, msgs[0]); node.receive(2, msgs[2])
            node.publish_ready()
            self.assertFalse(node.test_pubs['/nav/planar_evidence'].messages)
            self.assertEqual(node.counts['stale'], 6)
            self.assertEqual(node.counts['invalid'], 2)
        self.run_adapter(exercise)

    def test_output_that_becomes_stale_during_serialization_is_dropped(self):
        def exercise(node):
            for i, msg in enumerate(self.messages()): node.receive(i, msg)
            node.publish_ready()
            self.assertFalse(node.test_pubs['/nav/planar_evidence'].messages)
            self.assertEqual(node.counts['stale'], 1)
        self.run_adapter(exercise, advance_on_bridge_ns=300000000)

    def test_backward_clock_drops_previous_epoch_result(self):
        started, release = threading.Event(), threading.Event()
        def model(left, measured, *args, **kwargs):
            started.set(); release.wait(1.)
            return dict(inferred_depth=np.full_like(measured, 2.), free_ray_lower_bound=np.full_like(measured, 1.9), depth_interval_m=np.full_like(measured, .1), plane_id=np.zeros_like(measured, np.int16))
        def exercise(node):
            try:
                for i, msg in enumerate(self.messages()): node.receive(i, msg)
                self.assertTrue(started.wait(1.))
                node.test_now[0] = 10000000000
                node.clock_ns()
                release.set(); self.wait_ready(node); node.publish_ready()
                self.assertFalse(node.test_pubs['/nav/planar_evidence'].messages)
                self.assertEqual(node.counts['clock_resets'], 1)
                self.assertEqual(node.counts['stale'], 1)
            finally: release.set()
        self.run_adapter(exercise, enabled=True, model=model)

    def test_clock_reset_during_old_raw_flush_does_not_poison_new_stamp_history(self):
        def exercise(node):
            node.receive(2, self.messages()[2])
            node.last_submit_mono -= .21
            node.test_now[0] = 100250000000
            # Flushing the previous raw frame observes a clock rollback during
            # conversion. The 100.2s exposure must not enter the new 10s epoch.
            node.receive(2, self.messages(100200000000)[2])
            self.assertEqual(node.last_submit_stamp, -1)
            self.assertIsNone(node.pending_raw)
            node.receive(2, self.messages(10200000000)[2])
            self.assertEqual(node.last_submit_stamp, 10200000000)
        self.run_adapter(exercise, enabled=True, advance_on_bridge_ns=-90000000000)

    def test_resource_failure_disables_only_inference_and_keeps_raw_output(self):
        def exercise(node):
            self.assertFalse(node.enabled)
            self.assertTrue(node.resource_blocked)
            node.receive(2, self.messages()[2]); node.publish_ready()
            published = node.test_pubs['/nav/planar_evidence'].messages
            self.assertEqual(len(published), 1)
            self.assertTrue(np.isnan(published[0].array[..., 1]).all())
        self.run_adapter(exercise, enabled=True, reported_threads=10)

    def test_model_exception_immediately_falls_back_to_measured_channel(self):
        def model(*args, **kwargs): raise RuntimeError('injected inference failure')
        def exercise(node):
            msgs = self.messages()
            for i, msg in enumerate(msgs): node.receive(i, msg)
            self.wait_ready(node); node.publish_ready()
            published = node.test_pubs['/nav/planar_evidence'].messages
            self.assertEqual(len(published), 1)
            np.testing.assert_array_equal(published[0].array[..., 0], msgs[2].array)
            self.assertTrue(np.isnan(published[0].array[..., 1]).all())
            self.assertEqual(node.worker.errors, 1)
        self.run_adapter(exercise, enabled=True, model=model)

    def test_stalled_worker_cannot_freeze_raw_channel_or_publish_late_result(self):
        started, release = threading.Event(), threading.Event()
        def model(left, measured, *args, **kwargs):
            started.set(); release.wait(1.5)
            return dict(inferred_depth=np.full_like(measured, 2.), free_ray_lower_bound=np.full_like(measured, 1.9), depth_interval_m=np.full_like(measured, .1), plane_id=np.zeros_like(measured, np.int16))
        def expire_pending(node):
            epoch, key, submitted, deadline, raw = node.pending_raw
            node.pending_raw = (epoch, key, submitted, time.monotonic() - .001, raw)
        def exercise(node):
            try:
                msgs = self.messages()
                for i, msg in enumerate(msgs): node.receive(i, msg)
                self.assertTrue(started.wait(1.))
                expire_pending(node); node.publish_ready()
                published = node.test_pubs['/nav/planar_evidence'].messages
                self.assertEqual(len(published), 1)
                self.assertFalse(release.is_set())
                time.sleep(.21)
                node.test_now[0] = 100250000000
                second = self.messages(100200000000)[2]
                node.receive(2, second)
                expire_pending(node); node.publish_ready()
                self.assertEqual(len(published), 2)
                self.assertEqual(published[-1].header, second.header)
                self.assertTrue(np.isnan(published[-1].array[..., 1]).all())
                release.set(); self.wait_ready(node); node.publish_ready()
                self.assertEqual(len(published), 2)
                self.assertGreaterEqual(node.counts['stale'], 1)
            finally: release.set()
        self.run_adapter(exercise, enabled=True, model=model)


if __name__ == '__main__':
    unittest.main()
