"""ROS-independent bounded transport and free-only planar evidence contract."""
from collections import OrderedDict
import threading
import time
import numpy as np


class ExactTripletJoin:
    """Exact exposure stamps; at most capacity samples per input stream.

    Streams are left mono, right mono, measured stereo depth. Never nearest-time
    pair independent exposures. A completed newer exposure retires older input.
    Call only from the executor thread.
    """
    def __init__(self, capacity=6):
        if not 1 <= capacity <= 6:
            raise ValueError('Input capacity must be in [1, 6]')
        self.capacity = capacity
        self.reset()
        self.evicted = 0

    def reset(self):
        self.buffers = [OrderedDict() for _ in range(3)]
        self.last_emitted = -1

    def push(self, stream, stamp_ns, value):
        if stream not in (0, 1, 2) or stamp_ns < 0:
            raise ValueError('Invalid stream or exposure stamp')
        if stamp_ns <= self.last_emitted:
            return None
        target = self.buffers[stream]
        target[stamp_ns] = value
        while len(target) > self.capacity:
            del target[min(target)]
            self.evicted += 1
        if not all(stamp_ns in buf for buf in self.buffers):
            return None
        result = tuple(buf[stamp_ns] for buf in self.buffers)
        self.last_emitted = stamp_ns
        for buf in self.buffers:
            for old in [key for key in buf if key <= stamp_ns]:
                del buf[old]
        return stamp_ns, result


class LatestOnlyWorker:
    """One active + one pending job + one ready result; no executor work.

    Superseded pending jobs/results are discarded. In-flight work is bounded by
    the model policy, not preempted. The caller must gate age and compute budget.
    """
    def __init__(self, handler):
        self.handler = handler
        self.condition = threading.Condition()
        self.pending = None
        self.ready = None
        self.closed = False
        self.superseded = 0
        self.overwritten_results = 0
        self.errors = 0
        self.thread = threading.Thread(target=self._run, name='planar-evidence', daemon=True)
        self.thread.start()

    def submit(self, item):
        with self.condition:
            if self.closed:
                return False
            self.superseded += int(self.pending is not None)
            self.pending = item
            self.condition.notify()
            return True

    def _run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                item, self.pending = self.pending, None
            start = time.monotonic()
            try:
                result = self.handler(item)
                error = None
            except Exception as exc:
                result, error = None, repr(exc)
            elapsed = time.monotonic() - start
            with self.condition:
                if self.closed:
                    return
                self.errors += int(error is not None)
                self.overwritten_results += int(self.ready is not None)
                self.ready = (item, result, elapsed, error)

    def take(self):
        with self.condition:
            ready, self.ready = self.ready, None
            return ready

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = self.ready = None
            self.condition.notify_all()
        self.thread.join(timeout=2.)


def pack_evidence(measured, result=None, enabled=False):
    """32FC2: [measured axial Z, inferred free-only lower axial Z], meters.

    Channel 0 is an exact float32 copy, including NaN. Channel 1 is finite only
    for unknown raw pixels and a self-consistent bounded inference result. Never
    concatenate inferred surfaces into channel 0 or publish a point-cloud hit.
    """
    measured = np.asarray(measured)
    if measured.ndim != 2 or measured.dtype != np.float32:
        raise ValueError('Expected 32FC1 measured depth')
    evidence = np.full(measured.shape + (2,), np.nan, np.float32)
    evidence[..., 0] = measured
    if not enabled or result is None:
        return evidence
    lower = np.asarray(result['free_ray_lower_bound'])
    surface = np.asarray(result['inferred_depth'])
    interval = np.asarray(result['depth_interval_m'])
    plane_id = np.asarray(result['plane_id'])
    if any(a.shape != measured.shape for a in (lower, surface, interval, plane_id)):
        raise ValueError('Evidence grids must match measured native grid')
    accepted = np.isnan(measured) & np.isfinite(lower) & np.isfinite(surface) & np.isfinite(interval)
    accepted &= (plane_id >= 0) & (lower > .25) & (surface < 8.) & (surface > lower)
    accepted &= (interval >= 0) & (interval <= .12 * surface)
    accepted &= np.isclose(interval, surface - lower, atol=1e-5, rtol=1e-5)
    evidence[..., 1][accepted] = lower[accepted]
    return evidence
