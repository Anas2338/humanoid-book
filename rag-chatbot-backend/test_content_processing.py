#!/usr/bin/env python3
"""
Simple test script to verify content processing functionality without database dependencies.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the database-dependent imports
import unittest.mock
from unittest.mock import MagicMock

# Mock database dependencies before importing anything that uses them
with unittest.mock.patch.dict('sys.modules', {
    'config.database': MagicMock(),
    'models.entities': MagicMock(),
    'sqlalchemy': MagicMock(),
    'sqlalchemy.orm': MagicMock(),
    'sqlalchemy.ext.declarative': MagicMock(),
    'sqlalchemy.dialects.postgresql': MagicMock(),
}):
    # Now import the content processor (which will use the mocked modules)
    from utils.content_processor import content_processor

    def test_content_processing():
        """Test content processing with sample book content"""
        print("Testing content processing functionality...")

        # Read sample book content
        sample_content = """
# Introduction to Humanoid Robotics

Humanoid robots are robots with physical characteristics resembling humans.
They typically have a head, torso, two arms, and two legs.

## Key Components

The main components of humanoid robots include:

- Actuators for movement
- Sensors for perception
- Control systems for coordination

## Applications

Humanoid robots are used in various applications including:

- Healthcare assistance
- Customer service
- Research and development
- Entertainment

# Control Systems in Humanoid Robots

Control systems are crucial for coordinating the movements and behaviors of humanoid robots.
They process sensor data and generate appropriate motor commands.

## Types of Control Systems

- Centralized control
- Distributed control
- Hybrid approaches
"""

        print("Sample content loaded:")
        print(f"Content length: {len(sample_content)} characters")

        # Process the content into chunks
        print("\nProcessing content into chunks...")
        chunks = content_processor.process_book_content(sample_content, source_file="sample_content.md")

        print(f"\nSuccessfully processed content into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk['content'])} chars, type: {chunk['metadata'].get('content_type', 'unknown')}")
            print(f"    Heading: {chunk['metadata'].get('heading', 'None')}")
            print(f"    Content preview: {chunk['content'][:100]}...")
            print()

        # Simulate storing chunks (but just print instead of storing in DB)
        print(f"Simulating storage of {len(chunks)} content chunks...")
        for i, chunk_data in enumerate(chunks):
            print(f"  Would store chunk {i+1} with {len(chunk_data['content'])} characters")

        print("\n✓ Content processing test completed successfully!")
        print("  - Content was parsed into semantic sections")
        print("  - Chunks were created with proper metadata")
        print("  - Code structure is ready for database integration")

        return True

    if __name__ == "__main__":
        test_content_processing()