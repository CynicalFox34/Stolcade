// --- board.js ---
const ROWS = 20;
const COLS = 11;

let boardState = [];

function initBoard(container) {
    container.innerHTML = '';
    boardState = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));

    for(let r = 0; r < ROWS; r++) {
        if(r < 10) {
            boardState[r][5] = { player: 1 };
        } else {
            boardState[r][5] = { player: 2 };
        }
    }

    renderCells(container);
}

function renderCells(container) {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.row = r;
            cell.dataset.col = c;
            
            if (r < 10) {
                cell.dataset.rowHalf = 'top';
            } else {
                cell.dataset.rowHalf = 'bottom';
            }

            const pieceData = boardState[r][c];
            if (pieceData) {
                const pieceLine = document.createElement('div');
                pieceLine.classList.add('piece');
                pieceLine.classList.add(pieceData.player === 1 ? 'p1' : 'p2');
                cell.appendChild(pieceLine);
            }

            cell.addEventListener('click', () => handleCellClick(r, c));
            container.appendChild(cell);
        }
    }
}

function handleCellClick(row, col) {
    console.log(`Clicked row ${row}, col ${col}`);
}

// --- dashboard.js ---
function renderDashboard(container) {
    container.innerHTML = `
        <div class="view-dashboard fade-in">
            <h1>Welcome back, Player 1</h1>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Current ELO</h3>
                    <div class="value">1200</div>
                </div>
                <div class="stat-card">
                    <h3>Matches Played</h3>
                    <div class="value">0</div>
                </div>
                <div class="stat-card">
                    <h3>Win Rate</h3>
                    <div class="value">0%</div>
                </div>
                <div class="stat-card">
                    <h3>Global Rank</h3>
                    <div class="value">#--</div>
                </div>
            </div>
            <div style="margin-top: 3rem;">
                <h2>Recent Matches</h2>
                <div style="background: var(--bg-secondary); padding: 2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-top: 1rem; color: var(--text-secondary);">
                    No matches played yet. Go to the Play Game tab to find a match!
                </div>
            </div>
        </div>
    `;
}

// --- game.js ---
function renderPlayArea(container) {
    container.innerHTML = `
        <div class="view-play fade-in">
            <div class="board-container">
                <div class="player-info p1">
                    <div class="avatar">P1</div>
                    <h3>Player 1</h3>
                    <span style="color: var(--text-secondary)">ELO: 1200</span>
                </div>
                
                <div id="game-board"></div>
                
                <div class="player-info p2">
                    <div class="avatar">Bot</div>
                    <h3>Offline Bot</h3>
                    <span style="color: var(--text-secondary)">ELO: 1000</span>
                </div>
            </div>
        </div>
    `;
    
    // Make sure we evaluate after DOM is appended
    setTimeout(() => {
        const boardElement = document.getElementById('game-board');
        if(boardElement) {
            initBoard(boardElement);
        }
    }, 50);
}

// --- Router Logic ---
document.addEventListener('DOMContentLoaded', () => {
    const contentArea = document.getElementById('content');
    const navLinks = document.querySelectorAll('.nav-links a');

    function navigateTo(route) {
        navLinks.forEach(link => {
            link.classList.remove('active');
            if(link.dataset.route === route) {
                link.classList.add('active');
            }
        });

        contentArea.innerHTML = '';
        
        if (route === 'dashboard') {
            renderDashboard(contentArea);
        } else if (route === 'play') {
            renderPlayArea(contentArea);
        } else if (route === 'leaderboard') {
            contentArea.innerHTML = `
                <div class="view-dashboard fade-in">
                    <h1>Global Rankings</h1>
                    <p>Leaderboard coming soon once the backend is connected...</p>
                </div>
            `;
        }
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const route = e.target.closest('a').dataset.route;
            navigateTo(route);
        });
    });

    // Initialize on Dashboard
    navigateTo('dashboard');
});
