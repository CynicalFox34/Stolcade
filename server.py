#!/usr/bin/env python3
"""
Stolcade local server.
Serves board-game/ and exposes /api/log for the Engine & Data page.

Run:  python3 server.py
Open: http://localhost:8080
"""
import http.server, os, json, subprocess, random, string
from urllib.parse import urlparse, parse_qs

# ── LAN multiplayer sessions ────────────────────────────────
mp_games = {}

def _mp_code():
    chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'  # no confusing chars
    while True:
        code = ''.join(random.choices(chars, k=4))
        if code not in mp_games:
            return code

def _mp_cleanup():
    import time as _t
    now = _t.time()
    for c in list(mp_games.keys()):
        if now - mp_games[c]['created'] > 7200:
            del mp_games[c]

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BOARD_GAME_DIR  = os.path.join(BASE_DIR, 'board-game')
LOG_FILE        = os.path.join(BASE_DIR, 'engine', 'training.log')
EVAL_GAMES_FILE = os.path.join(BASE_DIR, 'engine', 'eval_games.json')

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BOARD_GAME_DIR, **kwargs)

    def do_GET(self):
        _parsed = urlparse(self.path)
        if _parsed.path == '/api/mp/state':
            params = parse_qs(_parsed.query)
            code = params.get('code', [''])[0].upper()
            if code not in mp_games:
                resp = {'status': 'notfound'}
            else:
                g = mp_games[code]
                resp = {'status': 'playing' if g['p2_joined'] else 'waiting',
                        'state': g['state'], 'seq': g['seq'],
                        'opponent_name': g['p2_name'] if g['p2_joined'] else ''}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/eval_games':
            content = '[]'
            if os.path.exists(EVAL_GAMES_FILE):
                with open(EVAL_GAMES_FILE, 'r') as f:
                    content = f.read()
            body = content.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/log':
            content = ''
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r') as f:
                    content = f.read()
            # Robust check: 'running' if heartbeat file was modified in last 120s
            import time
            HEARTBEAT_FILE = os.path.join(BASE_DIR, 'engine', 'checkpoints', 'train.heartbeat')
            running = False
            if os.path.exists(HEARTBEAT_FILE):
                mtime = os.path.getmtime(HEARTBEAT_FILE)
                if (time.time() - mtime) < 120:
                    running = True
            
            # Fallback: check ps if heartbeat is old (maybe just started)
            if not running:
                try:
                    ps_out = subprocess.getoutput("ps -A | grep -iE 'train.py|pretrain.py' | grep -v 'grep'")
                    running = len(ps_out.strip()) > 0
                except:
                    running = False
            body = json.dumps({'log': content, 'running': running}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/mp/create':
            import time as _t
            _mp_cleanup()
            clen = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(clen).decode()) if clen else {}
            code = _mp_code()
            mp_games[code] = {'p2_joined': False, 'state': None, 'seq': 0,
                              'created': _t.time(), 'p1_name': data.get('name', 'Player 1'), 'p2_name': ''}
            body = json.dumps({'code': code}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/mp/join':
            clen = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(clen).decode())
            code = data.get('code', '').upper()
            if code not in mp_games:
                resp = {'error': 'Game not found — check the code and try again'}
            elif mp_games[code]['p2_joined']:
                resp = {'error': 'Game is already full'}
            else:
                mp_games[code]['p2_joined'] = True
                mp_games[code]['p2_name'] = data.get('name', 'Player 2')
                resp = {'ok': True, 'opponent_name': mp_games[code]['p1_name']}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/mp/move':
            clen = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(clen).decode())
            code = data.get('code', '').upper()
            if code in mp_games:
                mp_games[code]['state'] = data.get('state')
                mp_games[code]['seq'] += 1
                resp = {'ok': True, 'seq': mp_games[code]['seq']}
            else:
                resp = {'error': 'Game not found'}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/start_training':
            import sys, time
            ENGINE_DIR = os.path.join(BASE_DIR, 'engine')
            # Check if already running
            ps_out = subprocess.getoutput("ps -A | grep -iE 'train.py|pretrain.py' | grep -v grep")
            if ps_out.strip():
                body = json.dumps({'status': 'already_running'}).encode()
            else:
                # Clear old log and launch pretrain.py as detached subprocess
                log_path = os.path.join(ENGINE_DIR, 'training.log')
                open(log_path, 'w').close()
                subprocess.Popen(
                    [sys.executable, os.path.join(ENGINE_DIR, 'pretrain.py')],
                    cwd=ENGINE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                body = json.dumps({'status': 'started'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/bot_move':
            clen = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(clen).decode())
            
            # Use engine/bot_api.py to get move
            # We import it here to avoid top-level dependency issues
            import sys
            ENGINE_DIR = os.path.join(BASE_DIR, 'engine')
            if ENGINE_DIR not in sys.path:
                sys.path.append(ENGINE_DIR)
            from bot_api import get_bot_move
            
            board  = data.get('board')
            level  = data.get('level', 'advanced')
            player = int(data.get('player', 2))

            move = get_bot_move(board, level, player=player)
            body = json.dumps(move).encode()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # silence request logs

if __name__ == '__main__':
    port = 8080
    print(f'Stolcade server running at http://localhost:{port}')
    http.server.HTTPServer(('', port), Handler).serve_forever()
