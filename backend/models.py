from sqlalchemy import Column, Integer, String, Float, DateTime, func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String, unique=True, index=True, nullable=False)
    email         = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    elo           = Column(Float, default=1000.0)
    wins          = Column(Integer, default=0)
    losses        = Column(Integer, default=0)
    draws         = Column(Integer, default=0)
    created_at    = Column(DateTime, server_default=func.now())
