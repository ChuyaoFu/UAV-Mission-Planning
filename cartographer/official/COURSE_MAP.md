# Official course map (from the organizers' top-view diagram the user shared, 2026-11-07)

Transcribed by reading the diagram's 1-ft grid, accurate to about ±0.3 ft (±0.1 m). The map is a coarse layout,
not CAD: use it as the course prior and refine everything with our own sensors. All walls are 68 in (1.73 m)
tall. Arena 20 × 20 ft (6.10 m). Our first-corridor measurement in the official sim (own stereo depth) agrees
with this map within 0.1–0.2 m.

## World frame used here: x = east, y = north, feet, origin at the SW corner (multiply by 0.3048 for m)

| element | geometry (ft) | notes |
|---|---|---|
| arena | x 0–20, y 0–20 | net cage |
| Launch | (1.74, 13.0) | drone faces north (+y) at takeoff |
| barrier (orange) | y = 11.5, x 0 → 15.2 | separates north (corridor, maze, tunnel area) from south (landing, chicane); flown over on the return |
| first corridor | x 0–3.7, from the launch north to y ≈ 18.5, then turn east | right wall W1: x = 3.7, y 11.6 → 16.9; checkpoint 1 at y = 18.5 |
| maze wall W2 | x = 7.35, y 14.8 → 20 | hangs from the north wall |
| maze wall W3 | x = 11.0, y 11.6 → 16.9 | Gate A = gap x = 11, y 16.9 → 20 (checkpoint 2) |
| floor path line (red) | (1.74,13.0) → (1.74,18.5) → (5.5,18.5) → (5.5,13.2) → (9.2,13.2) → (9.2,18.5) → (11.0,18.5) | serpentine: north, east, south (lane x 3.7–7.35), U-turn around the W2 south end (y 14.8), north (lane x 7.35–11), east through Gate A |
| tunnel (dark inside) | box x 15.2–20, y 9.0–14.0; enter at the NORTH face y = 14.0, exit at the SOUTH face y = 9.0 (checkpoint 3) | centre line x ≈ 17.6; length 5 ft (1.52 m) |
| chicane wall C1 | (14.9, 11.2) → (13.2, 6.3) | diagonal |
| chicane wall C2 + window | (10.5, 8.1) → (7.5, 0.0); window section ≈ (7.8, 1.0)–(9.0, 4.0), centre ≈ (8.4, 2.5) | crossing direction WNW (normal (-0.937, 0.347)) |
| Gate B | x = 10.5, y 8.1 → 11.5 (gap between the C2 north end and the barrier) | checkpoint 4 = Gate B OR the window (same 7 pts), "without landing" |
| short wall W4 | x = 6.8, y 8.0 → 11.3 | between the Gate B side and the landing area |
| landing pad (5) | centre (3.3, 7.35); outer ring radius ≈ 2.55 ft (0.78 m) | "land and disarm with the whole drone inside the pad's outer ring" |
| boulders (Ø 18 in = 0.46 m) | (1.6,1.8), (5.6,2.2), (12.6,2.3), (16.6,1.9), (18.3,4.9) | real obstacles |
| return bonus | landing → over the barrier → launch | dashed arc in the diagram |

Points: corridor 4, maze 7, tunnel 10, chicane 7, landing 7, plus the return bonus.

## Start frame (mission prior): forward f = north, left l = west, origin = launch
f = (y − 13.0) · 0.3048, l = −(x − 1.74) · 0.3048. See `mission/course_prior.venue_map.json`.
