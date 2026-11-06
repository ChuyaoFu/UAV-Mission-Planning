# Minimal global-height freeze contract

Root has already implemented this API. The adapter consumes it; the default integration patch does not duplicate or replace fusion/height-observer changes.

- Input: `/nav/freeze_global_height`, `std_msgs/Bool`; true latches for the process lifetime. False does nothing.
- `HeightObserver.freeze()` sets its frozen state, drops pending correction data and clears innovation reacquisition history. It retains the current accumulated correction and the existing VIO gauge.
- `HeightObserver.odometry()` continues accepted timestamp/history handling and returns the fixed correction while frozen. It does not integrate new ToF innovations into global z.
- Fusion continues publishing continuous VIO/body/EV data plus that fixed offset. Disarm/rearm must not reset it or the task's own launch coordinate frame.
- `/carto/localization_health` reports `global_height_frozen`. The adapter requires a fresh true acknowledgement before dock approach.
- Range sensing itself remains active for local `surface_relative_observation`, foot clearance, landing-plane agreement and native LAND monitoring.

Equivalent minimal source changes for an older pre-latch implementation are:

```python
# HeightObserver.__init__
self.frozen = False

# New method
def freeze(self):
    self.frozen = True
    self.pending = None
    self.large_innovations.clear()

# HeightObserver.odometry, after preserving the new accepted history sample:
if self.frozen:
    return self.correction

# Fusion.__init__
self.create_subscription(Bool, '/nav/freeze_global_height',
                         lambda m: self.height_observer.freeze() if m.data else None, 10)

# Existing health JSON:
'global_height_frozen': self.height_observer.frozen
```

Do not add an automatic timeout/unfreeze, reset the correction to zero, or replace the original ground anchor with a raised platform. Root owns actual ROS validation and any required pipeline-specific semantic policy.
