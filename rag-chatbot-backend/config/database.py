import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neon Postgres configuration
DATABASE_URL = os.getenv("NEON_DATABASE_URL", "postgresql://localhost/rag_chatbot")

# Create engine with proper SSL handling for Neon
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "sslmode": "require",
        "connect_timeout": 30,
        "keepalives_idle": 300,
        "keepalives_interval": 10,
        "keepalives_count": 3,
    },
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=299,    # Recycle connections before timeout (Neon has 5min timeout)
    pool_size=5,         # Reduced pool size to avoid connection limits
    max_overflow=10,     # Reduced overflow
    pool_timeout=30,
    pool_reset_on_return="commit",  # Reset connections on return to pool
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()