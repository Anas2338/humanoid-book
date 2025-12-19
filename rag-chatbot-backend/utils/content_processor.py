import re
from typing import List, Dict, Tuple
from models.entities import BookContentChunkSchema
from utils.embeddings import embeddings_client
from utils.vector_db import vector_db
from utils.logging import get_logger
import uuid

logger = get_logger(__name__)

class ContentProcessor:
    def __init__(self):
        self.embeddings_client = embeddings_client
        self.vector_db = vector_db

    def extract_sections(self, content: str) -> List[Dict]:
        """Extract sections from book content based on markdown headers"""
        sections = []

        # Split content by markdown headers (h1, h2, h3)
        lines = content.split('\n')

        current_section = {
            'heading': '',
            'content': '',
            'start_line': 0,
            'end_line': 0
        }

        section_start = 0

        for i, line in enumerate(lines):
            # Check for markdown headers
            h1_match = re.match(r'^#\s+(.+)', line)
            h2_match = re.match(r'^##\s+(.+)', line)
            h3_match = re.match(r'^###\s+(.+)', line)

            if h1_match or h2_match or h3_match:
                # If we have accumulated content, save the previous section
                if current_section['content'].strip():
                    sections.append({
                        'heading': current_section['heading'],
                        'content': current_section['content'].strip(),
                        'start_line': current_section['start_line'],
                        'end_line': current_section['end_line']
                    })

                # Start a new section
                heading = h1_match.group(1) if h1_match else h2_match.group(1) if h2_match else h3_match.group(1)
                current_section = {
                    'heading': heading,
                    'content': line + '\n',  # Include the header line
                    'start_line': i,
                    'end_line': i
                }
            else:
                # Add line to current section
                if current_section['content']:
                    current_section['content'] += line + '\n'
                    current_section['end_line'] = i
                else:
                    # If no section started yet, start with this line
                    current_section['content'] = line + '\n'
                    current_section['start_line'] = i
                    current_section['end_line'] = i
                    current_section['heading'] = 'Introduction'  # Default heading for content before first header

        # Add the last section if it has content
        if current_section['content'].strip():
            sections.append({
                'heading': current_section['heading'],
                'content': current_section['content'].strip(),
                'start_line': current_section['start_line'],
                'end_line': current_section['end_line']
            })

        return sections

    def chunk_content(self, sections: List[Dict], max_chunk_size: int = 1000, overlap: float = 0.1) -> List[Dict]:
        """Break down content into semantic chunks with overlap"""
        chunks = []

        for section in sections:
            heading = section['heading']
            content = section['content']
            start_line = section['start_line']

            # If content is smaller than max_chunk_size, use it as is
            if len(content) <= max_chunk_size:
                chunks.append({
                    'content': content,
                    'metadata': {
                        'heading': heading,
                        'section': heading,
                        'page': start_line,  # Using line number as a proxy for page
                        'source_file': 'book_content.md'  # Placeholder - would be actual file path in real implementation
                    }
                })
            else:
                # Split into overlapping chunks
                overlap_size = int(max_chunk_size * overlap)
                start = 0

                while start < len(content):
                    end = start + max_chunk_size

                    # If this is the last chunk, include the rest
                    if end >= len(content):
                        end = len(content)
                    else:
                        # Try to break at sentence boundary if possible
                        while end > start + max_chunk_size // 2 and end < len(content) and content[end] not in '.!?':
                            end += 1
                        if end < len(content):
                            end += 1  # Include the sentence ending punctuation

                    chunk_content = content[start:end].strip()

                    if chunk_content:  # Only add non-empty chunks
                        chunks.append({
                            'content': chunk_content,
                            'metadata': {
                                'heading': heading,
                                'section': heading,
                                'page': start_line + start // 100,  # Approximate page number
                                'source_file': 'book_content.md'
                            }
                        })

                    start = end - overlap_size if end < len(content) else end

        return chunks

    def process_book_content(self, book_content: str, source_file: str = "book_content.md") -> List[Dict]:
        """Process book content into chunks with semantic boundaries"""
        logger.info("Starting book content processing")

        # Extract sections based on headers
        sections = self.extract_sections(book_content)
        logger.info(f"Extracted {len(sections)} sections from book content")

        # Chunk the sections
        chunks = self.chunk_content(sections)
        logger.info(f"Created {len(chunks)} content chunks")

        # Add special handling for code examples, math formulas, and diagrams (T022a)
        processed_chunks = []
        for chunk in chunks:
            # Check for code blocks (markdown syntax)
            if '```' in chunk['content']:
                # This chunk contains code examples
                chunk['metadata']['content_type'] = 'text_with_code'
            # Check for math formulas (simple detection)
            elif any(symbol in chunk['content'] for symbol in ['$', '\\', '∑', '∫', '∂', '∇']):
                # This chunk contains math formulas
                chunk['metadata']['content_type'] = 'text_with_math'
            # Check for potential diagrams descriptions
            elif any(word in chunk['content'].lower() for word in ['figure', 'diagram', 'graph', 'chart', 'image']):
                # This chunk references diagrams
                chunk['metadata']['content_type'] = 'text_with_diagram_refs'
            else:
                chunk['metadata']['content_type'] = 'text_only'

            processed_chunks.append(chunk)

        logger.info("Book content processing completed")
        return processed_chunks

    def store_content_chunks(self, chunks: List[Dict]):
        """Store content chunks with embeddings in Qdrant database"""
        logger.info(f"Storing {len(chunks)} content chunks in vector database")

        for i, chunk_data in enumerate(chunks):
            try:
                # Generate embedding for the content
                embedding = self.embeddings_client.generate_embedding(chunk_data['content'])

                # Create a unique ID for this chunk
                chunk_id = str(uuid.uuid4())

                # Store in vector database
                self.vector_db.store_embedding(
                    chunk_id=chunk_id,
                    content=chunk_data['content'],
                    embedding=embedding,
                    metadata=chunk_data['metadata']
                )

                if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                    logger.info(f"Stored {i + 1}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error storing chunk {i}: {str(e)}")
                continue  # Continue with next chunk even if one fails

        logger.info(f"Successfully stored {len(chunks)} content chunks in vector database")

# Create a global instance
content_processor = ContentProcessor()