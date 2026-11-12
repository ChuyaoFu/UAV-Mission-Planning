# Survey walk-through → course prior / 手持走场 → 航线先验

One hand-carried walk with the real drone replaces the tape measure. `tools/survey_to_prior.py` turns the
recording into `course_prior.<name>.json` (START frame) plus a top-down PNG to check.
用真机手持走一遍场地代替卷尺。`tools/survey_to_prior.py` 把录包变成 `course_prior.<name>.json`（START 坐标系）和一张俯视检查图。

## Before / 准备

- **Props off, disarmed.** Drone powered, VIO healthy (`/estimator/healthy` true).
  **拆桨，不解锁。** 飞机上电，VIO 正常（`/estimator/healthy` 为 true）。
- **Mark the start pad**: tape the drone's footprint and a nose arrow. The flight must start from the same
  marks with the same heading: the walk's first pose *is* the START frame (2° heading error = 17 cm at 5 m).
  **在起飞垫上贴出机身轮廓和机头箭头。** 正式飞行必须放在同一位置、同一朝向：走场的第一个位姿就是 START 坐标系（朝向差 2°，5 m 外差 17 cm）。
- Record (any flight bag works too) / 录包（飞行包也能用）：

```bash
ros2 bag record -s mcap -o survey_venue --topics /stereo/left/image /stereo/left/camera_info /camera/depth/image_raw /camera/depth/camera_info /tf_static /fmu/out/vehicle_local_position_v1 /fmu/out/vehicle_attitude /fmu/out/distance_sensor /fmu/out/vehicle_status_v1
```

## Walk / 走场

Walk slowly (~0.5 m/s), keep the drone level, no fast spins, don't cover the camera. At every **stop** hold
still **≥ 3 s** with the nose toward the target. Don't pause ≥ 2 s anywhere else.
慢走（约 0.5 m/s），机身保持水平，不要快速转身，不要挡住相机。每个**停点**静止 **≥ 3 秒**，机头对准目标。其他地方不要停 ≥ 2 秒。

| # | Stop / 停点 | Where / 位置 | Gives / 得到 |
|---|---|---|---|
| 1 | Start / 起点 | On the pad marks, 5 s / 放在垫子标记上，5 秒 | START frame |
| – | Corridor / 通道 | Carry at flight height (~0.8 m) along the line it should fly / 按想飞的路线、飞行高度（约 0.8 m）走 | corridor, cruise height |
| 2 | Tunnel / 隧道 | ~1.8 m before the first hoop (≥ 1.2 m if the net is closer), on the axis, camera at hoop-centre height / 第一个圈前约 1.8 m（网太近时 ≥ 1.2 m），在轴线上，相机与圈心同高 | entry, axis, radius, length |
| 3 | Window / 窗 | ~1.5 m in front, square on, camera at window-centre height / 正前方约 1.5 m，正对，相机与窗中心同高 | centre, direction, size |
| 4 | Rock view / 看石头 | ~1.6 m from the white rock, camera ~0.5 m above its top / 离白石约 1.6 m，相机比石顶高约 0.5 m | rock centre |
| 5 | Rock top / 石顶 | Set the drone down on the middle of the rock top / 把飞机放在石顶中间 | top height (pose + ToF) |
| 6 | Wall, rock side / 墙·石头侧 | 0.8–1.0 m from the wall where it should cross, nose at the wall / 在要越墙的位置，离墙 0.8–1.0 m，机头对墙 | cross_from |
| 7 | Wall top / 墙顶 | Rest the skids on the top edge / 起落架搭在墙顶边上 | wall top_up |
| 8 | Wall, home side / 墙·回家侧 | 0.8–1.0 m on the other side, nose at the wall / 另一侧 0.8–1.0 m，机头对墙 | cross_to |
| 9–12 | Nets / 网 | Front, back, left, right: **nose first**, touch gently (don't push), chest height / 前后左右四面网：**机头朝网**，轻碰不推，胸口高度 | fence sides |
| 13 | End / 终点 | Back on the pad marks, same heading, 5 s / 回到垫子标记，同朝向，5 秒 | drift check |

Order is free, except: start first, end last, and wall stops 6-7-8 back to back.
顺序随意，但起点最先、终点最后，墙的 6-7-8 三个停点要连续。

## Make the prior / 生成先验

```bash
python3 tools/survey_to_prior.py survey_venue --name venue --out-dir flights/survey --list-stops
```

```bash
python3 tools/survey_to_prior.py survey_venue --name venue --out-dir flights/survey
```

- Open `course_prior.venue.png`: path, stops (index + label), detections, prior. Read every WARNING.
  打开 `course_prior.venue.png` 核对路径、停点（编号 + 标签）、检测点和先验；每条 WARNING 都要看。
- A stop labelled wrong or `?` → write a stops file (index label per line) and pass `--stops stops.txt`;
  labels: `start end tunnel window rock_view rock_top fence_front fence_back fence_left fence_right wall_from wall_top wall_to skip`.
  停点标错或显示 `?`：写一个停点文件（每行“编号 标签”），加 `--stops stops.txt`。
- Missing fields are left empty (the prior then does not load). Fill them from a tape/eyeball prior with
  `--fallback other.json` (marked FALLBACK in `_comment`), or redo that stop.
  测不到的字段留空（先验此时无法加载）。可以用 `--fallback other.json` 从卷尺/目测先验补（`_comment` 里标 FALLBACK），或者补走那个停点。
- Measured extras help: `--window-size W H`, `--tunnel-length L`, `--n-hoops N`, `--tunnel-radius R`.
  有实测值就加上：`--window-size W H`、`--tunnel-length L`、`--n-hoops N`、`--tunnel-radius R`。
- Fly with it / 用它飞：`python3 tools/flight_run.py --real --name ... --prior <path>/course_prior.venue.json`.

Assumptions you can change / 可改的假设：body centre 0.20 m from the net when the nose touches it
(`--touch-offset`); base_link 0.05 m above whatever it rests on (`--rest-height`, the node uses 0.05);
`fence.up_max` is not measured (`--up-max`, default 2.8).
机头碰网时机身中心离网 0.20 m（`--touch-offset`）；base_link 比所放表面高 0.05 m（`--rest-height`，节点也用 0.05）；`fence.up_max` 走场测不到（`--up-max`，默认 2.8）。
