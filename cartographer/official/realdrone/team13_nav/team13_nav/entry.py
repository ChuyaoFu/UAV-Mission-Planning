"""Console entry points. The algorithm code is the vendored official stack in <package src>/official
(copied there by cartographer/official/realdrone/deploy.sh; COLCON_IGNOREd), run unchanged."""
import os
import sys
from pathlib import Path

CANDIDATES = (os.environ.get('TEAM13_OFFICIAL'), str(Path(__file__).resolve().parents[1] / 'official'),
              '/root/team_ws/src/team13_nav/official')


def official_root():
    for c in CANDIDATES:
        if c and (Path(c) / 'node' / 'mission_node.py').is_file():
            root = str(Path(c).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return Path(root)
    raise SystemExit('team13_nav: vendored official/ tree not found (run realdrone/deploy.sh or set TEAM13_OFFICIAL)')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:  # noqa: BLE001 - a full disk must not stop the flight loop
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass


def _tee_to(logdir, name):
    if not logdir:
        return
    try:
        Path(logdir).mkdir(parents=True, exist_ok=True)
        f = open(Path(logdir) / name, 'a', buffering=1)
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)
    except OSError as e:
        print(f'team13_nav: cannot log to {logdir}: {e}', file=sys.stderr)


def mission():
    root = official_root()
    _tee_to(os.environ.get('TEAM13_LOG_DIR'), 'mission.log')
    os.chdir(root)
    from node import mission_node
    mission_node.main()


def stereo():
    official_root()
    _tee_to(os.environ.get('TEAM13_LOG_DIR'), 'stereo_depth.log')
    from realdrone import stereo_depth_node
    stereo_depth_node.main()


def bag_topics(images):
    official_root()
    from realdrone.skybeta_io import BAG_TOPICS, BAG_IMAGE_TOPICS
    return BAG_TOPICS + (BAG_IMAGE_TOPICS if images else [])
