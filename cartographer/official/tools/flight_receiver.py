#!/usr/bin/env python3
"""Local receiver for flight records pushed from the official lab browser page.

The lab companion has no laptop SSH for portal accounts, so the browser page
(which can read team files via code-server) PUTs each file to this server on
127.0.0.1. Files land under --root (default: <repo>/outputs/flights/official).

  python3 cartographer/official/tools/flight_receiver.py            # serve on 127.0.0.1:8765
  PUT /upload?path=<flight>/<file>   body = raw bytes  -> writes root/<flight>/<file>
  GET /ls?path=<flight>                                -> JSON list of files with sizes

Only binds to loopback; paths are confined to --root.
"""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[3]


def make_handler(root: Path):
    class H(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin') or '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.send_header('Access-Control-Allow-Private-Network', 'true')

        def _target(self):
            q = parse_qs(urlparse(self.path).query)
            rel = (q.get('path') or [''])[0].lstrip('/')
            p = (root / rel).resolve()
            if root.resolve() not in (p, *p.parents):
                raise ValueError('path escapes root')
            return p

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self._cors()
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204); self._cors(); self.send_header('Content-Length', '0'); self.end_headers()

        def do_PUT(self):
            try:
                p = self._target()
                n = int(self.headers.get('Content-Length', '0'))
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + '.part')
                with open(tmp, 'wb') as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        f.write(chunk); left -= len(chunk)
                if left:
                    tmp.unlink(missing_ok=True); return self._reply(400, dict(error='short body', missing=left))
                os.replace(tmp, p)
                print(f'saved {p.relative_to(root)} ({n} B)', flush=True)
                self._reply(200, dict(saved=str(p.relative_to(root)), bytes=n))
            except Exception as e:  # noqa: BLE001
                self._reply(400, dict(error=str(e)))

        def do_GET(self):
            try:
                p = self._target()
                files = [] if not p.exists() else [
                    dict(path=str(f.relative_to(root)), bytes=f.stat().st_size) for f in sorted(p.rglob('*')) if f.is_file()]
                self._reply(200, dict(files=files))
            except Exception as e:  # noqa: BLE001
                self._reply(400, dict(error=str(e)))

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--root', default=str(REPO / 'outputs' / 'flights' / 'official'))
    a = ap.parse_args()
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(root))
    print(f'flight receiver on http://127.0.0.1:{a.port} -> {root}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
