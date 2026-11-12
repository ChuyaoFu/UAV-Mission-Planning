# Venue replica (Gazebo Harmonic) — offline perception test set

Approximate replica of the DroneHacks 2026 venue (photos of 2026-11-06), rendered
headless along a flight route as an OAK-D-like stereo pair (`OFFICIAL_K`, 640x400,
horizontal, 7.5 cm baseline). **All dimensions are photo estimates.** Full write-up,
differences to the real venue, stereo parameters and evaluations:
`outputs/venue-replica/REPORT.md`.

Per-frame npz (`outputs/venue-replica/frames/`), `common/geometry.py` `Frame` conventions:

| field | meaning |
|---|---|
| `image`, `image_right` | uint8 400x640 mono left / right eye, with sensor model (gain wobble, read+shot noise, blur) |
| `depth` | **primary**: OAK-D-like stereo depth (SGBM on the noisy pair), metres along optical z, NaN = no return |
| `depth_ideal` | truth without cage net / tunnel black mesh |
| `depth_full` | truth of every surface incl. net + mesh |
| `surface` | uint8 per-pixel surface type (`index.json` `surface_names`) |
| `rgb`, `gain`, `K`, `R_nc`, `cam_ned`, `body_ned`, `heading`, `stamp`, `labels` | clean colour left image, sensor gain, pose, ground truth (JSON, offline only) |

| file | role |
|---|---|
| `layout.py` | single source of truth: estimated geometry, NED conventions, ground-truth labels, pose collision check |
| `make_textures.py` | photo crops → de-netted textures (`assets/textures/`), floor print, procedural hall/boards |
| `build_world.py` | meshes (`world/meshes/*.obj`) + `world/venue_full.sdf` (left/right RGB + depth) + `world/venue_depth.sdf` (no net / tunnel mesh) |
| `poses.py` | camera poses → `world/poses.json` (route at 0.4/0.8/1.2 m every 0.25 m + stage close-ups + photo-compare views) |
| `render_gz.py` | runs **inside** the container: moves the camera via `set_pose`, waits for `pose/info` + a newer frame, saves raw npz |
| `run_render.sh` | one-shot `docker run --rm --gpus all` (no host network, only the work dir mounted; forces NVIDIA EGL, chowns outputs back); `PASSES`, resumable |
| `stereo_depth.py` | sensor model + OpenCV StereoSGBM (96 disparities, LR check, 1/8 px subpixel, texture confidence, 5x5 median); all parameters in one place |
| `surfaces.py` | per-pixel surface type from `depth_ideal` / `depth_full` + layout geometry |
| `package_dataset.py` | raw → `frames/*.npz` (stereo depth computed here) + `index.json` + `qa/*.jpg` label overlays |
| `stereo_stats.py` | per-surface hole fraction and stereo error → `stereo_stats.json` |
| `make_overview.py` | `overview.png`, `compare_photos.png` |
| `check_dataset.py` | consistency checks of the packaged set against its labels |
| `dataset.py` | `load_frame(path, depth_key='depth'|'depth_ideal'|'depth_full') -> (Frame, labels)`, `iter_frames(root, group)` |
| `eval_perception.py` | runs perception/{floor,tunnel,window}.py unchanged (ideal vs stereo depth x prior/none) → `outputs/venue-replica/eval/` |

Reproduce (Mac side uses the repo `.venv`; photos `1..7.webp`; if Tailscale is flaky use the LAN alias `uav-ubuntu`):

```bash
cd cartographer/official/replica
../../../.venv/bin/python make_textures.py PHOTO_DIR
../../../.venv/bin/python build_world.py && ../../../.venv/bin/python poses.py
rsync -a --delete --exclude __pycache__ ./ uav-ubuntu-ts:/home/fcy/claude-work/venue-replica/replica/
ssh uav-ubuntu-ts 'cd /home/fcy/claude-work/venue-replica && OUT=out bash replica/run_render.sh'
rsync -a uav-ubuntu-ts:/home/fcy/claude-work/venue-replica/out/ RAW_DIR/
../../../.venv/bin/python package_dataset.py --raw RAW_DIR
../../../.venv/bin/python check_dataset.py && ../../../.venv/bin/python stereo_stats.py
../../../.venv/bin/python make_overview.py PHOTO_DIR && ../../../.venv/bin/python eval_perception.py
```

Labels, `depth_ideal`, `depth_full` and `surface` are ground truth for offline evaluation
only — flight code must never import `layout.py`, `dataset.py` or read them.
