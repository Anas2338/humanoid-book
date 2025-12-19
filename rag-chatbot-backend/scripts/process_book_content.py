#!/usr/bin/env python3
"""
Script to process book content and store it in the vector database with embeddings.
This script implements task T023: Store book content chunks with embeddings in Qdrant vector database.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.content_processor import content_processor
from utils.logging import get_logger

logger = get_logger(__name__)

def process_book_file(file_path: str):
    """Process a single book content file"""
    logger.info(f"Processing book content file: {file_path}")

    # Read the book content
    with open(file_path, 'r', encoding='utf-8') as f:
        book_content = f.read()

    logger.info(f"Read {len(book_content)} characters from {file_path}")

    # Process the content into chunks
    chunks = content_processor.process_book_content(book_content, source_file=file_path)

    # Store the chunks in the vector database
    content_processor.store_content_chunks(chunks)

    logger.info(f"Successfully processed and stored {len(chunks)} chunks from {file_path}")

def process_book_directory(directory_path: str):
    """Process all book content files in a directory"""
    logger.info(f"Processing all book content files in directory: {directory_path}")

    # Find all markdown files in the directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.md', '.mdx')):
                file_path = os.path.join(root, file)
                process_book_file(file_path)

def main():
    """Main function to process book content"""
    logger.info("Starting book content processing script")

    # Get the book content directory from command line argument or environment variable
    if len(sys.argv) > 1:
        content_path = sys.argv[1]
    else:
        content_path = os.getenv("BOOK_CONTENT_PATH", "./book_content")

    # Check if it's a file or directory
    if os.path.isfile(content_path):
        process_book_file(content_path)
    elif os.path.isdir(content_path):
        process_book_directory(content_path)
    else:
        logger.error(f"Path does not exist: {content_path}")
        print(f"Error: Path does not exist: {content_path}")
        sys.exit(1)

    logger.info("Book content processing completed successfully")

if __name__ == "__main__":
    main()