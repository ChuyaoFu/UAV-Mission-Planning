# Validation and metric interpretation

The latest downloaded evidence was obtained during the November 1 simulation campaign. This repository was assembled for publication on November 4. The Ubuntu connection was subsequently lost; no additional successful repeat is inferred from the interrupted attempt to start episode fast-12.

## Full-course flights

- **fast-09:** 61.868 s, zero contacts, ordered engineering regions, complete platform landing and disarm. Its controller was an earlier configuration. Confirmed-arm-to-disarm was 64.940 s.
- **fast-11:** 71.596 s, zero contacts, same required regions and window route, complete platform landing/disarm. Includes subsequent forward-visibility and height-recovery work.
- **fast-04:** 76.272 s, zero contacts, full window route. This intermediate mapping build had optimized-node RMSE about 5.2 cm and reference-surface P95 about 11.9 cm.
- **Gate B / fast-10:** 59.328 s elapsed, one contact with the C2b chicane wall, 5 s penalty, penalized time 64.328 s. Do not claim a clean sub-minute result from it.
- Attempts fast-02/03/05/06/07 included early landings or arena-boundary violations. Controller phase completion alone is insufficient; the independent judge determines the engineering result.

These are engineering checks against preliminary CAD annotations, not official referee certification. User-supplied rule summaries treated Gate B or the window as route options; current official event rules and timing semantics were not independently authenticated. Sixty seconds is an optimization target, not a newly verified official rule.

## SLAM geometry

Actual observed maps are sparse. Blank wall areas in a passive stereo reconstruction can mean insufficient texture/observability, not a scaled copy of the CAD. CAD geometry is never filled into the SLAM result.

All corrected trajectory comparisons use a single initialization gauge, without scale fitting or best-fit trajectory alignment. Earlier v3 reporting aligned the first structure-bearing node inside the recorder interval; this could hide pre-existing error. With the actual initialized node retained, slamfix-05 optimized-node RMSE is approximately **9.2 cm**, rather than the earlier 6.8 cm figure. The prior archived outputs are preserved locally and this publication documents the correction.

The latest downloaded fast-11 optimized-node RMSE was approximately **9.9 cm**. Its sparse-map surface error was median **4.4 cm**, P95 **14.4 cm**, with roughly **2.5%** of points farther than 20 cm from the independent CAD voxel surface. These do not establish complete wall coverage or zero drift. Scalar metrics and the full online pose statistics retained from downloaded runs are in [episode-metrics.json](episode-metrics.json).

## Controlled recorded-input checks

- Sampled same-recording stereo audit: reference-surface P95 of sensor points changed from about 11.1 cm to 8.8 cm; sampled outliers beyond 20 cm changed from about 1.16% to 0. Point count decreased, so this is not a completeness claim.
- Same original recording, height observer: previous-observer vertical RMSE about 8.4 cm versus 1.9 cm after floor reacquisition; final vertical error changed from about -13.7 cm to -0.24 cm. Only real passive stereo floor support and the existing single downward ToF were inputs. Truth was read afterwards for evaluation.
- Isolated fast-11 mapping replays compared odometry translation weights 1e6, 1e5 and 1e4 with identical recorded sensor messages and measured floor startup. Optimized-node RMSE was about 9.2, 11.0 and 21.5 cm, respectively. The weaker priors were rejected; active `oak_3d.lua` retains 1e6. Initial replay runs with multiple mapper instances were explicitly invalidated and excluded.

Loop closure remains enabled with the v3 post-refinement innovation and surface-normal observability checks. Recent flights had no accepted cross-submap loop constraints. Prior v3 accepted overlaps had separate physical-overlap checks; that does not establish arbitrary long-distance loop recall.

## Tests and hardware limits

Core tests cover stereo displacement/occlusion, calibration round trips, delayed height correction and floor reacquisition, small obstacle-top rejection, path progress, collision-checked smoothing, stopping bounds and perception backlog coalescing. The synthetic judge has ten acceptance checks. C++ logs in `validation/` are preserved prior regression evidence.

Desktop measurements are not Orin performance results. For fast-11, recorded stereo arrival age was approximately 72 ms median / 92 ms P95, and mapping publication age about 80 / 100 ms. These use original MCAP reception timestamps; reading a simulation clock inside a blocking callback can underestimate age and is not the assessment method.

The model is a seven-inch approximation with original flat course materials, ideal pinhole images, OpenCV passive stereo and a bounded downward-ToF proxy. Real vibration, blur, dark-tunnel operation, reflectance failures and actuator/braking limits remain to be validated. No real-airframe deployment or current competition compliance is claimed.
