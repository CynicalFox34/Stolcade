"""
Stolcade FastAPI server.

Serves the frontend and exposes all API endpoints including auth.

Run:
  cd /Users/lincoln/Documents/stolcade
  python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload

Open: http://localhost:8080
"""
import os, sys, json, random, string, subprocess, time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .database import engine, Base
from .routers import auth as auth_router

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD_GAME_DIR = os.path.join(BASE_DIR, 'board-game')
ENGINE_DIR     = os.path.join(BASE_DIR, 'engine')
LOG_FILE       = os.path.join(ENGINE_DIR, 'training.log')
EVAL_GAMES_FILE = os.path.join(ENGINE_DIR, 'eval_games.json')
HEARTBEAT_FILE = os.path.join(ENGINE_DIR, 'checkpoints', 'train.heartbeat')

# ── LAN multiplayer sessions (in-memory, same as server.py) ──
mp_games: dict = {}

def _mp_code():
    chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    while True:
        code = ''.join(random.choices(chars, k=4))
        if code not in mp_games:
            return code

def _mp_cleanup():
    now = time.time()
    for c in list(mp_games.keys()):
        if now - mp_games[c]['created'] > 7200:
            del mp_games[c]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)  # create DB tables on startup
    yield

app = FastAPI(title="Stolcade", lifespan=lifespan)

# ── Auth router ───────────────────────────────────────────────
app.include_router(auth_router.router)


# ── Existing API endpoints (ported from server.py) ────────────

@app.get("/api/log")
def get_log():
    content = ''
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            content = f.read()
    running = False
    if os.path.exists(HEARTBEAT_FILE):
        if (time.time() - os.path.getmtime(HEARTBEAT_FILE)) < 120:
            running = True
    if not running:
        try:
            ps = subprocess.getoutput("ps -A | grep -iE 'train.py|pretrain.py' | grep -v grep")
            running = len(ps.strip()) > 0
        except Exception:
            pass
    return JSONResponse({"log": content, "running": running},
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/api/eval_games")
def get_eval_games():
    content = '[]'
    if os.path.exists(EVAL_GAMES_FILE):
        with open(EVAL_GAMES_FILE) as f:
            content = f.read()
    return Response(content, media_type="application/json",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/api/mp/state")
def mp_state(code: str = ''):
    code = code.upper()
    if code not in mp_games:
        resp = {'status': 'notfound'}
    else:
        g = mp_games[code]
        resp = {'status': 'playing' if g['p2_joined'] else 'waiting',
                'state': g['state'], 'seq': g['seq'],
                'opponent_name': g['p2_name'] if g['p2_joined'] else ''}
    return JSONResponse(resp, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.post("/api/mp/create")
async def mp_create(request: Request):
    _mp_cleanup()
    data = await request.json() if int(request.headers.get('content-length', 0)) else {}
    code = _mp_code()
    mp_games[code] = {'p2_joined': False, 'state': None, 'seq': 0,
                      'created': time.time(),
                      'p1_name': data.get('name', 'Player 1'), 'p2_name': ''}
    return {'code': code}

@app.post("/api/mp/join")
async def mp_join(request: Request):
    data = await request.json()
    code = data.get('code', '').upper()
    if code not in mp_games:
        return JSONResponse({'error': 'Game not found — check the code and try again'})
    if mp_games[code]['p2_joined']:
        return JSONResponse({'error': 'Game is already full'})
    mp_games[code]['p2_joined'] = True
    mp_games[code]['p2_name'] = data.get('name', 'Player 2')
    return {'ok': True, 'opponent_name': mp_games[code]['p1_name']}

@app.post("/api/mp/move")
async def mp_move(request: Request):
    data = await request.json()
    code = data.get('code', '').upper()
    if code not in mp_games:
        return JSONResponse({'error': 'Game not found'})
    mp_games[code]['state'] = data.get('state')
    mp_games[code]['seq'] += 1
    return {'ok': True, 'seq': mp_games[code]['seq']}

@app.post("/api/start_training")
def start_training():
    ps = subprocess.getoutput("ps -A | grep -iE 'train.py|pretrain.py' | grep -v grep")
    if ps.strip():
        return {'status': 'already_running'}
    log_path = os.path.join(ENGINE_DIR, 'training.log')
    open(log_path, 'w').close()
    subprocess.Popen([sys.executable, os.path.join(ENGINE_DIR, 'pretrain.py')],
                     cwd=ENGINE_DIR, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return {'status': 'started'}

@app.post("/api/bot_move")
async def bot_move(request: Request):
    data = await request.json()
    if ENGINE_DIR not in sys.path:
        sys.path.append(ENGINE_DIR)
    from bot_api import get_bot_move
    board  = data.get('board')
    level  = data.get('level', 'advanced')
    player = int(data.get('player', 2))
    move   = get_bot_move(board, level, player=player)
    return JSONResponse(move)


# ── Serve frontend static files (must be last) ────────────────
app.mount("/", StaticFiles(directory=BOARD_GAME_DIR, html=True), name="frontend")
