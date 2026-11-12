"""Read a rosbag2 (mcap) without assuming a ROS install: used by tools/survey_to_prior.py.

Readers, tried in this order by open_bag(reader='auto'):
  rosbag2  rosbag2_py + rclpy deserialisation (the drone / any ROS 2 install with px4_msgs);
  mcap     the `mcap` pip package + the schema-driven CDR decoder below;
  builtin  a ~60-line mcap parser in this file (zstd chunks need the `zstandard` module, lz4 `lz4`).
The mcap/builtin paths decode every message from the ros2msg schema stored in the bag itself,
so they do not need px4_msgs and follow whatever px4_msgs version recorded it.

Every reader yields (topic, log_time_s, msg); msg is a ROS message object (rosbag2) or a dict
(mcap/builtin; uint8[] fields are zero-copy memoryviews). Use field(msg, name) for both.
Python 3.9+, numpy only.
"""
import mmap
import os
import re
import struct
from pathlib import Path

import numpy as np

MCAP_MAGIC = b'\x89MCAP0\r\n'


def field(m, name, default=None):
    """Field access for both ROS message objects and decoded dicts."""
    if isinstance(m, dict):
        return m.get(name, default)
    return getattr(m, name, default)


def mcap_files(path):
    """A bag directory (metadata.yaml + *.mcap) or one .mcap file -> sorted list of .mcap files."""
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(p.glob('*.mcap'), key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f.name)])
    if not files:
        raise FileNotFoundError('no .mcap file in %s' % p)
    return files


# ----------------------------------------------------------------------------------- CDR
_PRIM = {'bool': ('?', 1), 'byte': ('B', 1), 'char': ('B', 1), 'uint8': ('B', 1), 'int8': ('b', 1),
         'int16': ('h', 2), 'uint16': ('H', 2), 'int32': ('i', 4), 'uint32': ('I', 4),
         'int64': ('q', 8), 'uint64': ('Q', 8), 'float32': ('f', 4), 'float64': ('d', 8),
         'float': ('f', 4), 'double': ('d', 8)}
_NP = {'?': np.bool_, 'B': np.uint8, 'b': np.int8, 'h': np.int16, 'H': np.uint16, 'i': np.int32,
       'I': np.uint32, 'q': np.int64, 'Q': np.uint64, 'f': np.float32, 'd': np.float64}
_ALIAS = {'Header': 'std_msgs/Header', 'time': 'builtin_interfaces/Time', 'duration': 'builtin_interfaces/Duration'}


def _norm_type(t, pkg):
    t = _ALIAS.get(t, t)
    if '/' not in t:
        return '%s/%s' % (pkg, t)
    parts = t.split('/')
    return '%s/%s' % (parts[0], parts[-1])            # pkg/msg/Type -> pkg/Type


class CdrDecoder:
    """ROS 2 ros2msg schema text -> decoder for little/big endian plain CDR (XCDR1)."""

    def __init__(self, type_name, schema_text):
        self.defs = {}
        chunks = re.split(r'^=+\s*$', schema_text, flags=re.M)
        root = _norm_type(type_name, type_name.split('/')[0])
        self._parse(root, chunks[0])
        for c in chunks[1:]:
            c = c.strip('\n')
            first, _, body = c.partition('\n')
            m = re.match(r'\s*MSG:\s*(\S+)', first)
            if m:
                self._parse(_norm_type(m.group(1), m.group(1).split('/')[0]), body)
        self.root = root

    def _parse(self, name, text):
        pkg = name.split('/')[0]
        fields = []
        for line in text.splitlines():
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            toks = line.split(None, 1)
            if len(toks) < 2:
                continue
            typ, rest = toks
            if '=' in rest.split()[0] or (len(rest.split()) > 1 and rest.split()[1].startswith('=')):
                continue                                                  # constant
            fname = rest.split()[0]
            arr = None                                                    # None | -1 (sequence) | N (fixed)
            m = re.match(r'^(.*?)\[(.*)\]$', typ)
            if m:
                typ, inner = m.group(1), m.group(2)
                arr = int(inner) if inner.isdigit() else -1               # [], [<=N] -> sequence
            typ = re.sub(r'<=\d+$', '', typ)                              # string<=N
            if typ not in _PRIM and typ not in ('string', 'wstring'):
                typ = _norm_type(typ, pkg)
            fields.append((fname, typ, arr))
        self.defs[name] = fields

    def decode(self, data):
        b = memoryview(data)
        if len(b) < 4:
            raise ValueError('CDR payload too short')
        e = '<' if b[1] & 1 else '>'
        st = [4]                                                          # offset; alignment origin = 4
        return self._msg(self.root, b, st, e)

    def _align(self, st, n):
        st[0] += (-(st[0] - 4)) % n

    def _prim(self, typ, b, st, e):
        fmt, n = _PRIM[typ]
        self._align(st, n)
        v = struct.unpack_from(e + fmt, b, st[0])[0]
        st[0] += n
        return v

    def _string(self, b, st, e):
        self._align(st, 4)
        n = struct.unpack_from(e + 'I', b, st[0])[0]
        st[0] += 4
        s = bytes(b[st[0]:st[0] + n])
        st[0] += n
        return s.rstrip(b'\x00').decode('utf-8', 'replace')

    def _msg(self, name, b, st, e):
        try:
            fields = self.defs[name]
        except KeyError:
            raise ValueError('schema has no definition for %s' % name)
        out = {}
        for fname, typ, arr in fields:
            if arr is None:
                if typ in _PRIM:
                    out[fname] = self._prim(typ, b, st, e)
                elif typ == 'string':
                    out[fname] = self._string(b, st, e)
                elif typ == 'wstring':
                    raise ValueError('wstring not supported')
                else:
                    out[fname] = self._msg(typ, b, st, e)
                continue
            if arr < 0:
                self._align(st, 4)
                n = struct.unpack_from(e + 'I', b, st[0])[0]
                st[0] += 4
            else:
                n = arr
            if typ in _PRIM:
                fmt, sz = _PRIM[typ]
                if n:
                    self._align(st, sz)
                raw = b[st[0]:st[0] + n * sz]
                st[0] += n * sz
                if sz == 1 and fmt in 'Bb' and arr < 0:
                    out[fname] = raw                                       # zero-copy byte blobs (images)
                else:
                    dt = np.dtype(_NP[fmt]).newbyteorder(e)
                    out[fname] = np.frombuffer(raw, dt).astype(_NP[fmt]).tolist() if n <= 64 else \
                        np.frombuffer(raw, dt).astype(_NP[fmt])
            elif typ == 'string':
                out[fname] = [self._string(b, st, e) for _ in range(n)]
            else:
                out[fname] = [self._msg(typ, b, st, e) for _ in range(n)]
        return out


# ------------------------------------------------------------------------------ mcap readers
def _decompress(comp, data, size):
    if comp == '':
        return data
    if comp == 'zstd':
        try:
            import zstandard
        except ImportError as e:
            raise RuntimeError('bag chunks are zstd-compressed: pip install zstandard (or use rosbag2_py)') from e
        return zstandard.ZstdDecompressor().decompress(bytes(data), max_output_size=size)
    if comp == 'lz4':
        try:
            import lz4.frame
        except ImportError as e:
            raise RuntimeError('bag chunks are lz4-compressed: pip install lz4 (or use rosbag2_py)') from e
        return lz4.frame.decompress(bytes(data))
    raise RuntimeError('unknown mcap compression %r' % comp)


def _iter_builtin(path, topics):
    """Minimal MCAP v0 reader: (topic, log_time_ns, schema_name, schema_text, data)."""
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        mv = memoryview(mm)
        if bytes(mv[:8]) != MCAP_MAGIC:
            raise ValueError('%s is not an mcap file' % path)
        schemas, channels = {}, {}

        def s_(b, o):
            n = struct.unpack_from('<I', b, o)[0]
            return bytes(b[o + 4:o + 4 + n]).decode(), o + 4 + n

        def records(b):
            o, L = 0, len(b)
            while o + 9 <= L:
                op = b[o]
                n = struct.unpack_from('<Q', b, o + 1)[0]
                yield op, b[o + 9:o + 9 + n]
                o += 9 + n

        def handle(op, body):
            if op == 0x03:                                                # Schema
                sid = struct.unpack_from('<H', body, 0)[0]
                name, o = s_(body, 2)
                enc, o = s_(body, o)
                n = struct.unpack_from('<I', body, o)[0]
                schemas[sid] = (name, enc, bytes(body[o + 4:o + 4 + n]).decode('utf-8', 'replace'))
            elif op == 0x04:                                              # Channel
                cid, sid = struct.unpack_from('<HH', body, 0)
                topic, o = s_(body, 4)
                channels[cid] = (topic, sid)
            elif op == 0x05:                                              # Message
                cid, _seq, log_t, _pub = struct.unpack_from('<HIQQ', body, 0)
                ch = channels.get(cid)
                if ch is None or (topics is not None and ch[0] not in topics):
                    return None
                sc = schemas.get(ch[1], ('', '', ''))
                return ch[0], log_t, sc[0], sc[2], body[22:]
            return None

        b = mv[8:]
        for op, body in records(b):
            if op == 0x06:                                                # Chunk
                _t0, _t1, usize, _crc = struct.unpack_from('<QQQI', body, 0)
                comp, o = s_(body, 28)
                n = struct.unpack_from('<Q', body, o)[0]
                inner = memoryview(_decompress(comp, body[o + 8:o + 8 + n], usize))
                for op2, body2 in records(inner):
                    r = handle(op2, body2)
                    if r is not None:
                        yield r
            elif op in (0x0F, 0x02):                                      # DataEnd / Footer
                break
            else:
                r = handle(op, body)
                if r is not None:
                    yield r
    finally:
        try:
            mm.close()
        except BufferError:
            pass                                                          # a caller still holds a view


def _iter_mcap_pkg(path, topics):
    from mcap.reader import make_reader
    with open(path, 'rb') as f:
        r = make_reader(f)
        for sc, ch, m in r.iter_messages(topics=None if topics is None else list(topics)):
            yield ch.topic, m.log_time, sc.name if sc else '', sc.data.decode('utf-8', 'replace') if sc else '', m.data


class _SchemaDecoded:
    def __init__(self, raw_iter):
        self.raw = raw_iter
        self.dec = {}

    def __iter__(self):
        for topic, log_t, sname, stext, data in self.raw:
            d = self.dec.get(sname)
            if d is None:
                d = self.dec[sname] = CdrDecoder(sname, stext)
            yield topic, log_t * 1e-9, d.decode(data)


def _rosbag2_available():
    try:
        import rosbag2_py  # noqa: F401
        from rclpy.serialization import deserialize_message  # noqa: F401
        from rosidl_runtime_py.utilities import get_message  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return False


def _iter_rosbag2(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    p = Path(path)
    uri = str(p if p.is_dir() else p)
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=uri, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = [t for t in types if topics is None or t in topics]
    classes = {t: get_message(types[t]) for t in want}                   # raises if px4_msgs is missing
    r.set_filter(rosbag2_py.StorageFilter(topics=want))
    while r.has_next():
        topic, data, t = r.read_next()
        yield topic, t * 1e-9, deserialize_message(data, classes[topic])


def open_bag(path, topics=None, reader='auto'):
    """Iterator over (topic, log_time_s, msg) for `topics` (set or None = all), in file order.
    reader: auto | rosbag2 | mcap | builtin (auto = resolve_reader(path, topics))."""
    topics = None if topics is None else set(topics)
    if reader == 'auto':
        reader = resolve_reader('auto', path, topics)
    if reader == 'rosbag2':
        return _iter_rosbag2(path, topics)
    files = mcap_files(path)
    if reader == 'mcap':
        def raw():
            for f in files:
                yield from _iter_mcap_pkg(str(f), topics)
    elif reader == 'builtin':
        def raw():
            for f in files:
                yield from _iter_builtin(str(f), topics)
    else:
        raise ValueError('reader must be auto|rosbag2|mcap|builtin')
    return iter(_SchemaDecoded(raw()))


def _rosbag2_usable(path, topics):
    """rosbag2_py can open this bag (mcap storage plugin) and every wanted message type imports (px4_msgs)."""
    try:
        import rosbag2_py
        from rosidl_runtime_py.utilities import get_message
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
        for t in r.get_all_topics_and_types():
            if topics is None or t.name in topics:
                get_message(t.type)
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_reader(reader='auto', path=None, topics=None):
    """Name of the reader open_bag(reader) will use. auto: rosbag2 if it can read this bag (plugin + message
    packages), else the mcap package, else the built-in parser."""
    if reader != 'auto':
        return reader
    if _rosbag2_available() and (path is None or _rosbag2_usable(path, topics)):
        return 'rosbag2'
    try:
        import mcap.reader  # noqa: F401
        return 'mcap'
    except ImportError:
        return 'builtin'


# --------------------------------------------------------------------------- message helpers
def stamp_s(m):
    h = field(m, 'header')
    st = field(h, 'stamp')
    return float(field(st, 'sec')) + float(field(st, 'nanosec')) * 1e-9


def image_array(m):
    """sensor_msgs/Image -> (H, W) uint8 grey or float32 depth (metres), or None for other encodings."""
    enc = str(field(m, 'encoding'))
    h, w, step = int(field(m, 'height')), int(field(m, 'width')), int(field(m, 'step'))
    data = field(m, 'data')
    buf = np.frombuffer(data, np.uint8)
    big = bool(field(m, 'is_bigendian', 0))
    if enc in ('mono8', '8UC1'):
        return buf.reshape(h, step)[:, :w].copy()
    if enc in ('rgb8', 'bgr8'):
        import cv2
        im = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY if enc == 'rgb8' else cv2.COLOR_BGR2GRAY)
    if enc == '32FC1':
        return np.frombuffer(buf.reshape(h, step)[:, :w * 4].tobytes(), np.dtype('>f4' if big else '<f4')).reshape(h, w).astype(np.float32)
    if enc in ('16UC1', 'mono16'):
        d = np.frombuffer(buf.reshape(h, step)[:, :w * 2].tobytes(), np.dtype('>u2' if big else '<u2')).reshape(h, w)
        return d.astype(np.float32) / 1000.0
    return None


def quat_xyzw_to_mat(x, y, z, w):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class TfTree:
    """Static transforms parent <- child; lookup('base_link', 'oak_left') = (R, t) mapping child coords
    to parent coords, composed along the tree."""

    def __init__(self):
        self.edges = {}                                                   # child -> (parent, R, t)

    def add_msg(self, m):
        for tr in field(m, 'transforms') or []:
            par = str(field(field(tr, 'header'), 'frame_id')).lstrip('/')
            ch = str(field(tr, 'child_frame_id')).lstrip('/')
            T, Q = field(field(tr, 'transform'), 'translation'), field(field(tr, 'transform'), 'rotation')
            R = quat_xyzw_to_mat(field(Q, 'x'), field(Q, 'y'), field(Q, 'z'), field(Q, 'w'))
            self.edges[ch] = (par, R, np.array([field(T, 'x'), field(T, 'y'), field(T, 'z')], float))

    def _to_root(self, frame):
        R, t, chain = np.eye(3), np.zeros(3), [frame]
        while frame in self.edges and len(chain) < 32:
            par, Rp, tp = self.edges[frame]
            R, t = Rp @ R, Rp @ t + tp
            frame = par
            chain.append(frame)
        return chain, R, t

    def lookup(self, target, source):
        target, source = target.lstrip('/'), source.lstrip('/')
        if target == source:
            return np.eye(3), np.zeros(3)
        cs, Rs, ts = self._to_root(source)                                # source -> root_s
        ct, Rt, tt = self._to_root(target)                                # target -> root_t
        if cs[-1] != ct[-1]:
            return None
        # p_root = Rs p_s + ts = Rt p_t + tt  ->  p_t = Rt^T (Rs p_s + ts - tt)
        return Rt.T @ Rs, Rt.T @ (ts - tt)


def file_size_mb(path):
    return sum(os.path.getsize(f) for f in mcap_files(path)) / 1e6
