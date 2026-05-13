"""
S01-03: Модели данных SQLAlchemy ORM
"""

from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Boolean, 
    ForeignKey, Text, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    series = relationship("TimeSeries", back_populates="user", cascade="all, delete-orphan")
    models = relationship("TrainedModel", back_populates="user", cascade="all, delete-orphan")


class TimeSeries(Base):
    __tablename__ = "time_series"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    values = Column(JSON, nullable=False)
    dates = Column(JSON, nullable=True)
    length = Column(Integer, nullable=False)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    avg_value = Column(Float, nullable=True)
    file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="series")
    models = relationship("TrainedModel", back_populates="series", cascade="all, delete-orphan")
    forecasts = relationship("Forecast", back_populates="series", cascade="all, delete-orphan")


class TrainedModel(Base):
    __tablename__ = "trained_models"
    
    id = Column(String(36), primary_key=True)
    series_id = Column(String(36), ForeignKey("time_series.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    model_type = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    hyperparams = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    lag_features = Column(JSON, nullable=True)
    file_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    series = relationship("TimeSeries", back_populates="models")
    user = relationship("User", back_populates="models")
    forecasts = relationship("Forecast", back_populates="model", cascade="all, delete-orphan")


class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(String(36), primary_key=True)
    series_id = Column(String(36), ForeignKey("time_series.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(36), ForeignKey("trained_models.id", ondelete="CASCADE"), nullable=False)
    horizon = Column(Integer, nullable=False)
    predictions = Column(JSON, nullable=False)
    lower_bounds = Column(JSON, nullable=True)
    upper_bounds = Column(JSON, nullable=True)
    alpha = Column(Float, default=0.05)
    created_at = Column(DateTime, server_default=func.now())
    
    series = relationship("TimeSeries", back_populates="forecasts")
    model = relationship("TrainedModel", back_populates="forecasts")

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    token = Column(String(255), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    user = relationship("User", backref="refresh_tokens")
