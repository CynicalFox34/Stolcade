from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import math

from ..database import get_db
from ..models import User, Match
from ..auth import get_current_user

router = APIRouter(prefix="/api", tags=["matches"])


# ── Schemas ────────────────────────────────────────────────────

class MatchRecord(BaseModel):
    result: str          # 'win' | 'loss' | 'draw'
    is_bot: bool = True
    opponent: Optional[str] = None   # username of human opponent (if known)
    move_count: int = 0


class LeaderboardEntry(BaseModel):
    rank: int
    username: str
    elo: int
    wins: int
    losses: int
    draws: int
    is_me: bool = False

    model_config = {"from_attributes": True}


# ── Endpoints ──────────────────────────────────────────────────

@router.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    """Return top 50 players by ELO plus the current user's entry."""
    top = (db.query(User)
             .order_by(User.elo.desc())
             .limit(50)
             .all())

    result = []
    me_in_top = False
    for i, u in enumerate(top, start=1):
        is_me = u.id == current_user.id
        if is_me:
            me_in_top = True
        result.append({
            "rank": i, "username": u.username,
            "elo": round(u.elo), "wins": u.wins,
            "losses": u.losses, "draws": u.draws,
            "is_me": is_me,
        })

    # Always include current user at the bottom if not already in top 50
    if not me_in_top:
        total = db.query(User).count()
        rank  = db.query(User).filter(User.elo > current_user.elo).count() + 1
        result.append({
            "rank": rank, "username": current_user.username,
            "elo": round(current_user.elo), "wins": current_user.wins,
            "losses": current_user.losses, "draws": current_user.draws,
            "is_me": True,
        })

    return result


@router.post("/matches")
def record_match(body: MatchRecord,
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    """Record a completed match (bot games only — online matches are recorded server-side)."""
    if not body.is_bot:
        raise HTTPException(400, "Online match results are recorded automatically.")

    match = Match(
        player1_id = current_user.id,
        result_p1  = body.result,
        is_bot     = True,
        move_count = body.move_count,
        elo_p1_before = round(current_user.elo),
        elo_p1_after  = round(current_user.elo),  # bot games don't affect server ELO
    )
    db.add(match)

    # Update win/loss/draw counts even for bot games
    if body.result == 'win':
        current_user.wins   += 1
    elif body.result == 'loss':
        current_user.losses += 1
    else:
        current_user.draws  += 1

    db.commit()
    return {"ok": True}
