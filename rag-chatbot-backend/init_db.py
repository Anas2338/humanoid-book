import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import engine
from models.database_models import Base  # Import all models to register them with Base

def init_db():
    """Initialize the database by creating all tables"""
    print("Initializing database...")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()