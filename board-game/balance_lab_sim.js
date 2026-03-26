// Balance Lab Simulation — extracted from index.html
// Runs 200 games (100 P1-first, 100 P2-first) using the greedy racer strategy.

// ─── CONSTANTS ───────────────────────────────────────────────
const ROWS = 20, COLS = 11, MID_COL = 5;
const P1_GOAL = 19, P2_GOAL = 0, WIN_COUNT = 3;
const DIRS = [[-1,0],[1,0],[0,-1],[0,1]];

const BLOCKED_BORDER = new Set();
[9, 10].forEach(r => [2, 3, 7, 8].forEach(c => BLOCKED_BORDER.add(r * COLS + c)));

function isBlocked(r, c) { return BLOCKED_BORDER.has(r * COLS + c); }

// ─── GLOBAL BOARD STATE (swapped by withState) ───────────────
let boardState = [];

// ─── HELPERS ─────────────────────────────────────────────────
function countAtGoal(player) {
    const row = player === 1 ? P1_GOAL : P2_GOAL;
    let n = 0;
    for (let c = 0; c < COLS; c++) if (boardState[row][c]?.player === player) n++;
    return n;
}
function veylantSteps(player) { return 2 + countAtGoal(player); }

function didCross(player, fromR, toR) {
    return player === 1 ? (fromR < 10 && toR >= 10) : (fromR >= 10 && toR < 10);
}
function blockedByBorder(piece, toR) {
    if (!piece.hasCrossed) return false;
    if (piece.player === 1 && toR < 10)  return true;
    if (piece.player === 2 && toR >= 10) return true;
    return false;
}

// ─── VALID MOVES ─────────────────────────────────────────────
function movesForRegular(r, c) {
    const piece = boardState[r][c];
    if (piece.player === 1 && r === P1_GOAL) return [];
    if (piece.player === 2 && r === P2_GOAL) return [];

    const inHome = (piece.player === 1 && r < 10) || (piece.player === 2 && r >= 10);
    const out = [];

    for (const [dr, dc] of DIRS) {
        let nr = r + dr, nc = c + dc;
        let pushPath = [[r, c]];
        let canPush = true;
        const isSideways = (dr === 0);

        while (true) {
            if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) { canPush = false; break; }
            if (isBlocked(nr, nc)) { canPush = false; break; }
            const target = boardState[nr][nc];

            if (!target) {
                if (blockedByBorder(piece, nr)) canPush = false;
                break;
            }

            if (target.player !== piece.player) {
                if (nc === MID_COL) { canPush = false; break; }
                if (isSideways && inHome) { canPush = false; break; }
                if (!isSideways) {
                    const isFwd = (piece.player === 1 && dr === 1) || (piece.player === 2 && dr === -1);
                    if (!isFwd) { canPush = false; break; }
                }
                const pushStrength = pushPath.length;
                let enemyChainLen = 0;
                let scanR = nr, scanC = nc;
                let chainOk = true;
                while (true) {
                    enemyChainLen++;
                    const nR = scanR + dr, nC = scanC + dc;
                    if (nR < 0 || nR >= ROWS || nC < 0 || nC >= COLS) { chainOk = false; break; }
                    if (isBlocked(nR, nC)) { chainOk = false; break; }
                    const s = boardState[nR][nC];
                    if (!s) { break; }
                    if (s.player === piece.player) { chainOk = false; break; }
                    if (s.player !== target.player) { chainOk = false; break; }
                    scanR = nR; scanC = nC;
                }
                if (!chainOk || enemyChainLen > 1 || pushStrength <= enemyChainLen) { canPush = false; break; }
                out.push({ isEnemyPush: true, target: [nr, nc], dr, dc });
                canPush = false; break;
            }

            if ((piece.player === 1 && nr >= 10 && dr === 1) ||
                (piece.player === 2 && nr < 10  && dr === -1)) { canPush = false; break; }
            const isEnemyTerritory = (piece.player === 1 && nr >= 10) || (piece.player === 2 && nr < 10);
            if (isEnemyTerritory && !isSideways) { canPush = false; break; }
            const isOwnTerritory = (piece.player === 1 && nr < 10) || (piece.player === 2 && nr >= 10);
            const isForward = (piece.player === 1 && dr === 1) || (piece.player === 2 && dr === -1);
            if (isOwnTerritory && !isForward) { canPush = false; break; }

            pushPath.push([nr, nc]);
            nr += dr; nc += dc;
        }

        if (canPush) out.push({ path: pushPath, target: [nr, nc], dr, dc });
    }
    return out;
}

function movesForVeylant(r, c) {
    const piece = boardState[r][c];
    if (piece.player === 1 && r === P1_GOAL) return [];
    if (piece.player === 2 && r === P2_GOAL) return [];

    const steps = veylantSteps(piece.player);
    const reachable = new Map();
    const visited = new Map();

    const cloneState = (bs) => bs.map(row => [...row]);

    function simulatePush(board, sr, sc, dr, dc, eR, eC) {
        const nb = cloneState(board);
        let pR = eR - dr, pC = eC - dc;
        let emptyR = eR, emptyC = eC;
        let loops = 0;
        while (loops++ < 30) {
            if (pR < 0 || pR >= ROWS || pC < 0 || pC >= COLS) break;
            const p = nb[pR][pC];
            if (!p) break;
            nb[emptyR][emptyC] = p;
            if (pR === sr && pC === sc) break;
            emptyR = pR; emptyC = pC;
            pR -= dr; pC -= dc;
        }
        nb[sr][sc] = null;
        return nb;
    }

    const queue = [[r, c, 0, [], boardState]];
    visited.set(r * COLS + c, 0);

    while (queue.length) {
        const [cr, cc, used, pathSoFar, currentBoard] = queue.shift();
        if (used >= steps) continue;

        const inHome = (piece.player === 1 && cr < 10) || (piece.player === 2 && cr >= 10);

        for (const [dr, dc] of DIRS) {
            let nr = cr + dr, nc = cc + dc;
            let canPush = true;
            const isSideways = (dr === 0);
            let pushCount = 0;
            let hitEnemy = false;
            let firstEnemyR = -1, firstEnemyC = -1;

            while (true) {
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) { canPush = false; break; }
                if (isBlocked(nr, nc)) { canPush = false; break; }
                const target = currentBoard[nr][nc];

                if (!target) {
                    if (!hitEnemy && blockedByBorder(piece, nr)) canPush = false;
                    break;
                }

                if (target.player !== piece.player) {
                    if (!hitEnemy) {
                        if (nc === MID_COL) { canPush = false; break; }
                        if (isSideways && inHome) { canPush = false; break; }
                        if (!isSideways) {
                            const isFwd = (piece.player === 1 && dr === 1) || (piece.player === 2 && dr === -1);
                            if (!isFwd) { canPush = false; break; }
                        }
                        const pushStrength = pushCount + 1;
                        let enemyChainLen = 0;
                        let scanR = nr, scanC = nc;
                        let chainOk = true;
                        while (true) {
                            enemyChainLen++;
                            const nR = scanR + dr, nC = scanC + dc;
                            if (nR < 0 || nR >= ROWS || nC < 0 || nC >= COLS) { chainOk = false; break; }
                            if (isBlocked(nR, nC)) { chainOk = false; break; }
                            const s = currentBoard[nR][nC];
                            if (!s) { break; }
                            if (s.player === piece.player) { chainOk = false; break; }
                            if (s.player !== target.player) { chainOk = false; break; }
                            scanR = nR; scanC = nC;
                        }
                        if (!chainOk || enemyChainLen > 1 || pushStrength <= enemyChainLen) { canPush = false; break; }
                        firstEnemyR = nr; firstEnemyC = nc;
                        hitEnemy = true;
                    }
                    nr += dr; nc += dc;
                    continue;
                }

                if (hitEnemy) { canPush = false; break; }
                if ((piece.player === 1 && nr >= 10 && dr === 1) || (piece.player === 2 && nr < 10 && dr === -1)) { canPush = false; break; }
                const isEnemyTerritory = (piece.player === 1 && nr >= 10) || (piece.player === 2 && nr < 10);
                if (isEnemyTerritory && !isSideways) { canPush = false; break; }
                const isOwnTerritory = (piece.player === 1 && nr < 10) || (piece.player === 2 && nr >= 10);
                const isForward = (piece.player === 1 && dr === 1) || (piece.player === 2 && dr === -1);
                if (isOwnTerritory && !isForward) { canPush = false; break; }
                pushCount++;
                nr += dr; nc += dc;
            }

            if (!canPush) continue;

            const nextUsed = used + 1;
            if (nextUsed > steps) continue;

            const veylantNewR = cr + dr;
            const veylantNewC = cc + dc;
            const newPath = [...pathSoFar, { dr, dc, emptyR: nr, emptyC: nc, isEnemyPush: hitEnemy, enemyTarget: hitEnemy ? [firstEnemyR, firstEnemyC] : null }];

            if (hitEnemy) {
                const landR = cr + dr, landC = cc + dc;
                const eKey = landR * COLS + landC;
                if (eKey !== r * COLS + c && !reachable.has(eKey)) reachable.set(eKey, { target: [landR, landC], isVeylantMulti: true, pathSequence: newPath });
                const vKey = landR * COLS + landC;
                if (!visited.has(vKey) || visited.get(vKey) >= nextUsed) {
                    visited.set(vKey, nextUsed);
                    const nextBoard = simulatePush(currentBoard, cr, cc, dr, dc, nr, nc);
                    const justCrossed = didCross(piece.player, cr, landR);
                    if (!justCrossed) queue.push([landR, landC, nextUsed, newPath, nextBoard]);
                }
            } else {
                const key = nr * COLS + nc;
                if (key !== r * COLS + c && !reachable.has(key)) reachable.set(key, { target: [nr, nc], isVeylantMulti: true, pathSequence: newPath });
                const visitedKey = veylantNewR * COLS + veylantNewC;
                const justCrossed = didCross(piece.player, cr, veylantNewR);
                if (!justCrossed && (!visited.has(visitedKey) || visited.get(visitedKey) >= nextUsed)) {
                    visited.set(visitedKey, nextUsed);
                    let nextBoard;
                    if (pushCount > 0) {
                        nextBoard = simulatePush(currentBoard, cr, cc, dr, dc, nr, nc);
                    } else {
                        nextBoard = cloneState(currentBoard);
                        nextBoard[veylantNewR][veylantNewC] = nextBoard[cr][cc];
                        nextBoard[cr][cc] = null;
                    }
                    queue.push([veylantNewR, veylantNewC, nextUsed, newPath, nextBoard]);
                }
            }
        }
    }
    return Array.from(reachable.values());
}

function getValidMoves(r, c) {
    const piece = boardState[r][c];
    if (!piece) return [];
    return piece.isVeylant ? movesForVeylant(r, c) : movesForRegular(r, c);
}

// ─── STATE UTILITIES ─────────────────────────────────────────
function withState(state, fn) {
    const saved = boardState;
    boardState = state;
    const result = fn();
    boardState = saved;
    return result;
}

function getAllMoves(state, player) {
    return withState(state, () => {
        const moves = [];
        for (let r = 0; r < ROWS; r++)
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c]?.player !== player) continue;
                for (const m of getValidMoves(r, c)) moves.push({ fromR: r, fromC: c, moveObj: m });
            }
        return moves;
    });
}

function cloneDeep(state) {
    return state.map(row => row.map(p => p ? { ...p } : null));
}

function applyMove(state, fromR, fromC, moveObj, player) {
    const s = cloneDeep(state);
    let bonusEarned = false;

    function pushLine(sr, sc, dr, dc, eR, eC) {
        let pR = eR - dr, pC = eC - dc, emptyR = eR, emptyC = eC, loops = 0;
        while (loops++ < 30) {
            if (pR < 0 || pR >= ROWS || pC < 0 || pC >= COLS) break;
            const p = s[pR][pC];
            if (!p) break;
            s[emptyR][emptyC] = p;
            if (p.player === player && didCross(player, pR, emptyR)) { p.hasCrossed = true; bonusEarned = true; if (p.isVeylant) p.isVeylant = false; }
            if (p.player !== player) {
                if (p.player === 2 && p.hasCrossed && emptyR >= 10) p.hasCrossed = false;
                if (p.player === 1 && p.hasCrossed && emptyR < 10)  p.hasCrossed = false;
            }
            if (pR === sr && pC === sc) break;
            emptyR = pR; emptyC = pC; pR -= dr; pC -= dc;
        }
        s[sr][sc] = null;
    }

    if (moveObj.isEnemyPush) {
        const { dr, dc } = moveObj;
        const [tR, tC] = moveObj.target;
        let scanR = tR, scanC = tC;
        while (true) {
            const nR = scanR + dr, nC = scanC + dc;
            if (nR < 0 || nR >= ROWS || nC < 0 || nC >= COLS) break;
            if (isBlocked(nR, nC)) break;
            const sv = s[nR][nC];
            if (!sv || sv.player === player) break;
            scanR = nR; scanC = nC;
        }
        pushLine(fromR, fromC, dr, dc, scanR + dr, scanC + dc);
    } else if (moveObj.path) {
        pushLine(fromR, fromC, moveObj.dr, moveObj.dc, moveObj.target[0], moveObj.target[1]);
    } else if (moveObj.isVeylantMulti) {
        let cr = fromR, cc = fromC;
        for (const step of moveObj.pathSequence) {
            if (step.isEnemyPush) {
                let scanR = step.enemyTarget[0], scanC = step.enemyTarget[1];
                while (true) {
                    const nR = scanR + step.dr, nC = scanC + step.dc;
                    if (nR < 0 || nR >= ROWS || nC < 0 || nC >= COLS) break;
                    if (isBlocked(nR, nC)) break;
                    const sv = s[nR][nC];
                    if (!sv || sv.player === player) break;
                    scanR = nR; scanC = nC;
                }
                pushLine(cr, cc, step.dr, step.dc, scanR + step.dr, scanC + step.dc);
                cr = step.enemyTarget[0]; cc = step.enemyTarget[1];
            } else {
                pushLine(cr, cc, step.dr, step.dc, step.emptyR, step.emptyC);
                cr += step.dr; cc += step.dc;
            }
        }
    }
    return { newState: s, bonusEarned };
}

function countOnGoal(state, player) {
    const row = player === 1 ? P1_GOAL : P2_GOAL;
    let n = 0;
    for (let c = 0; c < COLS; c++) if (state[row][c]?.player === player) n++;
    return n;
}

// ─── BALANCE LAB ─────────────────────────────────────────────
function labInitialState() {
    const state = [];
    for (let r = 0; r < ROWS; r++) {
        const row = [];
        for (let c = 0; c < COLS; c++) {
            if (c === MID_COL) {
                const pl = r < 10 ? 1 : 2;
                const isVeylant = (pl === 1 && r === 0) || (pl === 2 && r === 19);
                row.push({ player: pl, isVeylant, hasCrossed: false });
            } else row.push(null);
        }
        state.push(row);
    }
    return state;
}

function labGreedyMove(state, player) {
    const goal = player === 1 ? P1_GOAL : P2_GOAL;
    const moves = getAllMoves(state, player);
    if (!moves.length) return null;

    let best = null, bestScore = -Infinity;
    for (const { fromR, fromC, moveObj } of moves) {
        const [tr] = moveObj.target;
        const dist = Math.abs(tr - goal);
        let score = (ROWS - 1 - dist) * 10;
        if (tr === goal) score += 1000;
        if (didCross(player, fromR, tr)) score += 200;
        const piece = state[fromR][fromC];
        if (piece && piece.hasCrossed) score += 30;
        if (piece && piece.isVeylant) score += 5;
        score += Math.random() * 2;
        if (score > bestScore) { bestScore = score; best = { fromR, fromC, moveObj }; }
    }
    return best;
}

function labSimulateGame(firstPlayer) {
    let state = labInitialState();
    let currentPlayer = firstPlayer;
    let actionsLeft = 1;
    let bonusPend = [false, false, false];
    let totalMoves = 0;
    const MAX_MOVES = 5000;

    while (totalMoves < MAX_MOVES) {
        if (countOnGoal(state, 1) >= WIN_COUNT) return { winner: 1, moves: totalMoves };
        if (countOnGoal(state, 2) >= WIN_COUNT) return { winner: 2, moves: totalMoves };

        const best = labGreedyMove(state, currentPlayer);

        if (!best) {
            const next = currentPlayer === 1 ? 2 : 1;
            const nb = [...bonusPend];
            actionsLeft = nb[next] ? 2 : 1; nb[next] = false;
            bonusPend = nb; currentPlayer = next;
            totalMoves++; continue;
        }

        const { newState, bonusEarned } = applyMove(state, best.fromR, best.fromC, best.moveObj, currentPlayer);
        const nb = [...bonusPend];
        if (bonusEarned) nb[currentPlayer] = true;
        state = newState;
        totalMoves++;

        if (actionsLeft > 1) {
            actionsLeft--;
        } else {
            const next = currentPlayer === 1 ? 2 : 1;
            actionsLeft = nb[next] ? 2 : 1; nb[next] = false;
            bonusPend = nb; currentPlayer = next;
        }
    }
    return { winner: 0, moves: totalMoves };
}

// ─── MAIN ─────────────────────────────────────────────────────
const TOTAL_GAMES = 200; // 100 P1-first, 100 P2-first
const stats = { p1: 0, p2: 0, draws: 0, lengths: [], firstMoverWins: 0 };

console.log(`Running ${TOTAL_GAMES} games (${TOTAL_GAMES / 2} P1-first, ${TOTAL_GAMES / 2} P2-first)...\n`);
const startTime = Date.now();

for (let g = 0; g < TOTAL_GAMES; g++) {
    const firstPlayer = g % 2 === 0 ? 1 : 2;
    const result = labSimulateGame(firstPlayer);

    if (result.winner === 1) stats.p1++;
    else if (result.winner === 2) stats.p2++;
    else stats.draws++;

    if (result.winner !== 0 && result.winner === firstPlayer) stats.firstMoverWins++;
    stats.lengths.push(result.moves);

    if ((g + 1) % 20 === 0) process.stdout.write(`  Game ${g + 1}/${TOTAL_GAMES} done\n`);
}

const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
const decided = TOTAL_GAMES - stats.draws;
const avgLen  = Math.round(stats.lengths.reduce((a, b) => a + b, 0) / stats.lengths.length);
const minLen  = Math.min(...stats.lengths);
const maxLen  = Math.max(...stats.lengths);
const p1Pct   = (stats.p1 / TOTAL_GAMES * 100).toFixed(1);
const p2Pct   = (stats.p2 / TOTAL_GAMES * 100).toFixed(1);
const drawPct = (stats.draws / TOTAL_GAMES * 100).toFixed(1);
const fmPct   = decided > 0 ? (stats.firstMoverWins / decided * 100).toFixed(1) : 'N/A';

console.log('\n══════════════════════════════════════════');
console.log('         BALANCE LAB RESULTS              ');
console.log('══════════════════════════════════════════');
console.log(`Games run      : ${TOTAL_GAMES} (${TOTAL_GAMES/2} P1-first, ${TOTAL_GAMES/2} P2-first)`);
console.log(`Time           : ${elapsed}s`);
console.log('──────────────────────────────────────────');
console.log(`P1 win rate    : ${stats.p1} / ${TOTAL_GAMES}  (${p1Pct}%)`);
console.log(`P2 win rate    : ${stats.p2} / ${TOTAL_GAMES}  (${p2Pct}%)`);
console.log(`Draw rate      : ${stats.draws} / ${TOTAL_GAMES}  (${drawPct}%)`);
console.log('──────────────────────────────────────────');
console.log(`First-mover wins (of decided): ${stats.firstMoverWins} / ${decided}  (${fmPct}%)`);
console.log('──────────────────────────────────────────');
console.log(`Avg game length: ${avgLen} half-moves`);
console.log(`Min game length: ${minLen} half-moves`);
console.log(`Max game length: ${maxLen} half-moves`);
console.log('══════════════════════════════════════════');
