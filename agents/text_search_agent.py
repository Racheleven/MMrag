from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSearchAgent:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "text_collection"
    ):
        """Initialize the text search agent with ColBERT and Qdrant.

        Args:
            qdrant_host (str): Qdrant server host
            qdrant_port (int): Qdrant server port
            collection_name (str): Name of the collection in Qdrant
        """
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port
        )
        
        # Initialize FastEmbed (ColBERT) model
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create a new collection in Qdrant if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # BGE-small embedding dimension
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def index_texts(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Index a list of texts with optional metadata.
        
        Args:
            texts (List[str]): List of text documents to index
            metadata (List[Dict]): List of metadata dictionaries for each text
        """
        if metadata is None:
            metadata = [{} for _ in texts]
            
        # Generate embeddings for all texts
        embeddings = self.embedding_model.embed(texts)
        
        # Prepare points for Qdrant
        points = []
        for i, (embedding, text, meta) in enumerate(zip(embeddings, texts, metadata)):
            points.append(models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": text,
                    **meta
                }
            ))
        
        # Upload points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Indexed {len(texts)} documents")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts using the query.
        
        Args:
            query (str): Search query text
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict]: List of search results with text and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed([query])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "text": result.payload["text"],
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })
            
        return results
    
    def delete_all(self):
        """Delete all documents from the collection."""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._create_collection()
            logger.info(f"Deleted and recreated collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}") 