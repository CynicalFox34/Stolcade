from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, func
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
    move_count   = Column(Integer, default=0)
    created_at   = Column(DateTime, server_default=func.now())
