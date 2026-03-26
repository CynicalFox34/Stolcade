from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, func, or_, UniqueConstraint
from .database import Base

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    elo             = Column(Float, default=1000.0)
    wins            = Column(Integer, default=0)
    losses          = Column(Integer, default=0)
    draws           = Column(Integer, default=0)
    created_at      = Column(DateTime, server_default=func.now())


class PasswordReset(Base):
    __tablename__ = "password_resets"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    token      = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used       = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


class Match(Base):
    __tablename__ = "matches"

    id           = Column(Integer, primary_key=True, index=True)
    player1_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    player2_id   = Column(Integer, ForeignKey("users.id"), nullable=True)   # null = bot
    result_p1    = Column(String, nullable=False)   # 'win' | 'loss' | 'draw'
    elo_p1_before = Column(Integer, default=0)
    elo_p1_after  = Column(Integer, default=0)
    elo_p2_before = Column(Integer, nullable=True)
    elo_p2_after  = Column(Integer, nullable=True)
    is_bot       = Column(Boolean, default=False)
    rated        = Column(Boolean, default=True, nullable=True)
    move_count   = Column(Integer, default=0)
    created_at   = Column(DateTime, server_default=func.now())


class Friendship(Base):
    __tablename__ = "friendships"
    __table_args__ = (UniqueConstraint("requester_id", "addressee_id"),)

    id           = Column(Integer, primary_key=True, index=True)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    addressee_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status       = Column(String, default="pending")  # pending / accepted / rejected
    created_at   = Column(DateTime, server_default=func.now())


class GameChallenge(Base):
    __tablename__ = "game_challenges"

    id            = Column(Integer, primary_key=True, index=True)
    challenger_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    challenged_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rated         = Column(Boolean, default=True)
    status        = Column(String, default="pending")  # pending / accepted / declined / cancelled
    game_token    = Column(String, unique=True, index=True, nullable=False)
    created_at    = Column(DateTime, server_default=func.now())
