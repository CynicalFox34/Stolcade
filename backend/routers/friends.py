"""
Friends and challenge endpoints.

Friends flow:
  POST /api/friends/request     — send request by username
  GET  /api/friends             — list accepted friends
  GET  /api/friends/requests    — incoming pending requests
  PUT  /api/friends/{id}/accept — accept a pending request
  DELETE /api/friends/{id}      — reject / remove friend

Challenge flow:
  POST /api/challenges                — create challenge (friend_id, rated)
  GET  /api/challenges/incoming       — pending challenges sent TO me
  PUT  /api/challenges/{id}/accept    — accept → returns game_token
  PUT  /api/challenges/{id}/decline   — decline
  DELETE /api/challenges/{id}         — cancel (challenger only)
"""

import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from ..database import get_db
from ..models import Friendship, GameChallenge, User
from ..auth import get_current_user

router = APIRouter(prefix="/api", tags=["friends"])


# ── Schemas ────────────────────────────────────────────────────

class FriendRequest(BaseModel):
    username: str

class ChallengeCreate(BaseModel):
    friend_id: int
    rated: bool = True


# ── Friend helpers ─────────────────────────────────────────────

def _friendship(db, uid_a, uid_b):
    """Return the friendship row between two users in either direction."""
    from sqlalchemy import or_, and_
    return db.query(Friendship).filter(
        or_(
            and_(Friendship.requester_id == uid_a, Friendship.addressee_id == uid_b),
            and_(Friendship.requester_id == uid_b, Friendship.addressee_id == uid_a),
        )
    ).first()


# ── Friends endpoints ──────────────────────────────────────────

@router.post("/friends/request", status_code=201)
def send_friend_request(body: FriendRequest, db: Session = Depends(get_db),
                        me: User = Depends(get_current_user)):
    target = db.query(User).filter(User.username == body.username).first()
    if not target:
        raise HTTPException(404, "User not found")
    if target.id == me.id:
        raise HTTPException(400, "You cannot add yourself")

    existing = _friendship(db, me.id, target.id)
    if existing:
        if existing.status == "accepted":
            raise HTTPException(400, "Already friends")
        if existing.status == "pending":
            raise HTTPException(400, "Request already sent")
        # rejected → allow re-request by updating status
        existing.status = "pending"
        existing.requester_id = me.id
        existing.addressee_id = target.id
        db.commit()
        return {"ok": True}

    db.add(Friendship(requester_id=me.id, addressee_id=target.id))
    db.commit()
    return {"ok": True}


@router.get("/friends")
def list_friends(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    from sqlalchemy import or_, and_
    rows = db.query(Friendship).filter(
        or_(Friendship.requester_id == me.id, Friendship.addressee_id == me.id),
        Friendship.status == "accepted"
    ).all()

    friends = []
    for r in rows:
        fid = r.addressee_id if r.requester_id == me.id else r.requester_id
        u = db.query(User).filter(User.id == fid).first()
        if u:
            friends.append({
                "friendship_id": r.id,
                "id": u.id,
                "username": u.username,
                "elo": round(u.elo),
                "wins": u.wins,
                "losses": u.losses,
                "draws": u.draws,
            })
    return friends


@router.get("/friends/requests")
def incoming_requests(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    rows = db.query(Friendship).filter(
        Friendship.addressee_id == me.id,
        Friendship.status == "pending"
    ).all()
    result = []
    for r in rows:
        u = db.query(User).filter(User.id == r.requester_id).first()
        if u:
            result.append({"id": r.id, "from_id": u.id, "username": u.username, "elo": round(u.elo)})
    return result


@router.put("/friends/{fid}/accept")
def accept_friend(fid: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    row = db.query(Friendship).filter(
        Friendship.id == fid,
        Friendship.addressee_id == me.id,
        Friendship.status == "pending"
    ).first()
    if not row:
        raise HTTPException(404, "Request not found")
    row.status = "accepted"
    db.commit()
    return {"ok": True}


@router.delete("/friends/{fid}")
def remove_friend(fid: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    from sqlalchemy import or_
    row = db.query(Friendship).filter(
        Friendship.id == fid,
        or_(Friendship.requester_id == me.id, Friendship.addressee_id == me.id)
    ).first()
    if not row:
        raise HTTPException(404, "Not found")
    db.delete(row)
    db.commit()
    return {"ok": True}


# ── Challenge endpoints ────────────────────────────────────────

@router.post("/challenges", status_code=201)
def create_challenge(body: ChallengeCreate, db: Session = Depends(get_db),
                     me: User = Depends(get_current_user)):
    # Must be friends
    fs = _friendship(db, me.id, body.friend_id)
    if not fs or fs.status != "accepted":
        raise HTTPException(400, "You must be friends to challenge this player")

    # Cancel any existing pending challenge between them
    from sqlalchemy import or_, and_
    old = db.query(GameChallenge).filter(
        or_(
            and_(GameChallenge.challenger_id == me.id, GameChallenge.challenged_id == body.friend_id),
            and_(GameChallenge.challenger_id == body.friend_id, GameChallenge.challenged_id == me.id),
        ),
        GameChallenge.status == "pending"
    ).all()
    for c in old:
        c.status = "cancelled"

    token = secrets.token_urlsafe(24)
    ch = GameChallenge(
        challenger_id = me.id,
        challenged_id = body.friend_id,
        rated         = body.rated,
        game_token    = token,
    )
    db.add(ch)
    db.commit()
    db.refresh(ch)
    return {"id": ch.id, "game_token": token}


@router.get("/challenges/incoming")
def incoming_challenges(db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    # Expire old challenges (older than 5 minutes)
    cutoff = datetime.utcnow() - timedelta(minutes=5)
    db.query(GameChallenge).filter(
        GameChallenge.status == "pending",
        GameChallenge.created_at < cutoff
    ).update({"status": "cancelled"})
    db.commit()

    rows = db.query(GameChallenge).filter(
        GameChallenge.challenged_id == me.id,
        GameChallenge.status == "pending"
    ).all()
    result = []
    for c in rows:
        u = db.query(User).filter(User.id == c.challenger_id).first()
        if u:
            result.append({
                "id": c.id,
                "challenger": u.username,
                "challenger_elo": round(u.elo),
                "rated": c.rated,
                "game_token": c.game_token,
            })
    return result


@router.put("/challenges/{cid}/accept")
def accept_challenge(cid: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    ch = db.query(GameChallenge).filter(
        GameChallenge.id == cid,
        GameChallenge.challenged_id == me.id,
        GameChallenge.status == "pending"
    ).first()
    if not ch:
        raise HTTPException(404, "Challenge not found or already expired")
    ch.status = "accepted"
    db.commit()
    return {"game_token": ch.game_token, "rated": ch.rated}


@router.put("/challenges/{cid}/decline")
def decline_challenge(cid: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    ch = db.query(GameChallenge).filter(
        GameChallenge.id == cid,
        GameChallenge.challenged_id == me.id,
        GameChallenge.status == "pending"
    ).first()
    if not ch:
        raise HTTPException(404, "Challenge not found")
    ch.status = "declined"
    db.commit()
    return {"ok": True}


@router.delete("/challenges/{cid}")
def cancel_challenge(cid: int, db: Session = Depends(get_db), me: User = Depends(get_current_user)):
    ch = db.query(GameChallenge).filter(
        GameChallenge.id == cid,
        GameChallenge.challenger_id == me.id,
        GameChallenge.status.in_(["pending", "accepted"])
    ).first()
    if not ch:
        raise HTTPException(404, "Challenge not found")
    ch.status = "cancelled"
    db.commit()
    return {"ok": True}
