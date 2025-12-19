from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorDBClient:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Collection name for book content
        self.collection_name = "book_content_chunks"

        # Vector size for Qwen embeddings (1536 dimensions)
        self.vector_size = 1536

        # Initialize the collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def store_embedding(self, chunk_id: str, content: str, embedding: List[float], metadata: Dict):
        """Store a content chunk with its embedding in the vector database"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "metadata": metadata
                    }
                )
            ]
        )

    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Search for similar content chunks based on embedding similarity"""
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )

        results = []
        for result in search_results.points:  # Access the points attribute
            results.append({
                "id": result.id,
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "score": result.score
            })

        return results

    def search_similar_with_filters(self, query_embedding: List[float], limit: int = 5, filters: Dict = None) -> List[Dict]:
        """Search for similar content chunks with optional filters"""
        # Build Qdrant filter if filters are provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value)
                ))
            if conditions:
                qdrant_filter = models.Filter(must=conditions)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            query_filter=qdrant_filter
        )

        results = []
        for result in search_results.points:  # Access the points attribute
            results.append({
                "id": result.id,
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "score": result.score
            })

        return results

    def delete_collection(self):
        """Delete the entire collection (useful for resets)"""
        self.client.delete_collection(self.collection_name)

    def get_collection_info(self):
        """Get information about the collection"""
        return self.client.get_collection(self.collection_name)

# Create a global instance
vector_db = VectorDBClient()