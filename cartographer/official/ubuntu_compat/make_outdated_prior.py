#!/usr/bin/env python3
"""OUTDATED maze prior: the maze was changed after the prior was made. The world's own blind prior, but its maze part
(corridor waypoints + the generated maze walls D*) replaced by ANOTHER maze's (the donor's), converted into this
world's start frame (the two mazes have different start points). Everything outside the maze (barrier, W3, C1, C2,
W4, tunnel, chicane, gates, window, pad, boulders, fence) stays this world's.

  python3 make_outdated_prior.py WORLD DONOR      (in the container)
      -> ubuntu_compat/course_prior.maze<WORLD>.outdated.json   (fly: run_blind.sh ... --mazes "WORLD" --kind outdated)
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FT = 0.3048


def start_ab(s):
    x, y = json.loads(Path(f'/compat/maze/m{s}/replica/maze.json').read_text())['start_ft']
    return y * FT, x * FT                                        # cage (a north, b east) of the launch


def main():
    w, d = int(sys.argv[1]), int(sys.argv[2])
    P = json.loads((HERE / f'course_prior.maze{w}.blind.json').read_text())
    D = json.loads((HERE / f'course_prior.maze{d}.blind.json').read_text())
    (wa, wb), (da, db) = start_ab(w), start_ab(d)

    def conv(q):                                                 # donor start frame -> cage -> world start frame
        a, b = q['forward'] + da, -q['left'] + db
        return dict(q, forward=round(a - wa, 3), left=round(-(b - wb), 3))
    P['corridor'] = [conv(c) for c in D['corridor']]
    P['walls'] = [x for x in P.get('walls', []) if not x['name'].startswith('D')] + \
                 [dict(x, **{'from': conv(x['from']), 'to': conv(x['to'])}) for x in D.get('walls', []) if x['name'].startswith('D')]
    P['_scale'] = 'maze-outdated'; P['_seed'] = w
    P['_comment'] = (f'OUTDATED prior: world maze {w}, maze part (corridor + D* walls) from maze {d} (converted to this start frame; '
                     f'donor start offset {round(da - wa, 3)} m north, {round(db - wb, 3)} m east). ' + P.get('_comment', '')[:100])
    out = HERE / f'course_prior.maze{w}.outdated.json'
    out.write_text(json.dumps(P, indent=1) + '\n')
    print(out.name, 'from donor', d, '| corridor', len(P['corridor']), 'D walls', sum(1 for x in P['walls'] if x['name'].startswith('D')))


if __name__ == '__main__':
    main()
