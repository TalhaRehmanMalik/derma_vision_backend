"""
database.py
MySQL connection via SQLAlchemy.
The database and all tables are created automatically on first startup.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from utils.logger import get_logger

load_dotenv()
logger = get_logger("database")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD", "legend")
DB_NAME = os.getenv("DB_NAME", "derma_vision")


def create_database_if_not_exists():
    """Connect without a database selected and create it if it does not exist."""
    engine_root = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}",
        echo=False
    )
    with engine_root.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        conn.commit()
    engine_root.dispose()
    logger.info(f"Database '{DB_NAME}' ready.")


# Main engine — connected to derma_vision database
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# engine       = create_engine(DATABASE_URL, echo=False)
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,    
    pool_recycle=280,      
    pool_size=5,           
    max_overflow=10        
)



SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session per request, always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()