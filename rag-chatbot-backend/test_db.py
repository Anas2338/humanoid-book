import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import SessionLocal
from services.database_service import DatabaseService

def test_db_connection():
    """Test the database connection"""
    print("Testing database connection...")

    db = SessionLocal()
    try:
        db_service = DatabaseService(db)

        # Try to create a session
        session = db_service.create_session()
        print(f"Successfully created session: {session.id}")

        # Clean up - delete the test session
        db.delete(session)
        db.commit()

        print("Database connection test completed successfully!")
        return True
    except Exception as e:
        print(f"Database connection test failed: {str(e)}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    test_db_connection()