import logging
import os
import uuid
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

from chonkie import BaseChunker, RecursiveChunker, RecursiveRules
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import List, Dict, Optional, Union, Tuple, Any
import hashlib
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np
import openai

from rna_rag.utils import batch_list, get_required_env_var

logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Define the log format
)


class Document(BaseModel):
    file_name: str
    content: str


class DenseEmbedder(ABC):
    """Abstract base class for dense text embeddings."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for the given texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dense embeddings (vectors)
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings produced by this embedder.
        
        Returns:
            The number of dimensions in the embedding vectors
        """
        pass


class FastEmbedEmbedder(DenseEmbedder):
    """Implementation of DenseEmbedder using FastEmbed."""

    def __init__(self, model_name: str = None):
        """
        Initialize the FastEmbed embedder.
        
        Args:
            model_name: Name of the model to use for embeddings
        """
        # Use environment variable if model_name not specified
        model_name = model_name or os.getenv("DENSE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings using FastEmbed.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dense embeddings (vectors)
        """
        return list(self.model.embed(texts))

    @property
    def dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings produced by FastEmbed.
        
        Returns:
            The number of dimensions in the embedding vectors
        """
        output = list(self.model.passage_embed(["halko"]))
        return len(output[0])


class OpenAIEmbedder(DenseEmbedder):
    """Implementation of DenseEmbedder using OpenAI's API."""

    def __init__(self, api_key: str = None, model_name: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedder.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI embedding model to use
        """
        if api_key is None:
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("API key must be provided or set as an environment variable.")
            api_key = os.environ["OPENAI_API_KEY"]

        self.client = openai.Client(api_key=api_key)
        self.model_name = model_name
        # Get dimensions based on model name
        if model_name == "text-embedding-3-small":
            self._dimensions = 1536
        elif model_name == "text-embedding-3-large":
            self._dimensions = 3072
        elif model_name == "text-embedding-ada-002":
            self._dimensions = 1536
        else:
            # Default to 1536 if model name is unknown
            self._dimensions = 1536

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings using OpenAI's API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dense embeddings (vectors)
        """
        result = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in result.data]

    @property
    def dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings produced by OpenAI.
        
        Returns:
            The number of dimensions in the embedding vectors
        """
        return self._dimensions


class QdrantVectorDatabase:

    def __init__(
            self,
            collection_name: str = None,
            location: str = None,
            dense_embedder: DenseEmbedder = None,
            dense_embedder_model_name: str = None,
            sparse_embedding_model: str = None,
    ):
        """
        Initialize Qdrant database connection and collection
        
        Args:
            collection_name: Name of the collection to use
            location: Location of the Qdrant server
            dense_embedder: Embedder for dense vectors (defaults to FastEmbedEmbedder)
            dense_embedder_model_name: Model name for dense embeddings (used if dense_embedder is None)
            sparse_embedding_model: Model name for sparse embeddings
        """
        self.client = QdrantClient(
            location=location or get_required_env_var("QDRANT_URL"),
            # prefer_grpc=True  # Better for large batches
        )
        self.collection_name = collection_name or get_required_env_var("QDRANT_COLLECTION_NAME")
        self.location = location or get_required_env_var("QDRANT_URL")
        
        # Get chunking parameters from environment
        chunk_size = int(get_required_env_var("CHUNK_SIZE"))
        min_characters_per_chunk = int(get_required_env_var("MIN_CHARACTERS_PER_CHUNK"))
        
        self.chunker = RecursiveChunker(
            tokenizer="gpt2",
            chunk_size=chunk_size,
            rules=RecursiveRules(),  # Default rules
            min_characters_per_chunk=min_characters_per_chunk,
        )

        # Initialize embedding models
        if dense_embedder is None and dense_embedder_model_name is not None:
            dense_embedder = FastEmbedEmbedder(model_name=dense_embedder_model_name)
        elif dense_embedder is None:
            dense_embedder = FastEmbedEmbedder()
            
        self.dense_embedder = dense_embedder
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name=sparse_embedding_model or get_required_env_var("SPARSE_EMBEDDING_MODEL")
        )

        if not self.client.collection_exists(self.collection_name):
            # Create collection with properly configured vector settings
            vectors_config = {
                "dense": models.VectorParams(
                    size=self.dense_embedder.dimensions,
                    distance=models.Distance.COSINE,
                ),
            }

            # Add sparse vectors configuration if sparse model is provided
            sparse_vectors_config = None
            if sparse_embedding_model:
                sparse_vectors_config = {
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }

            logging.info(f"Creating collection '{self.collection_name}' with dense vectors.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )

    def _get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Generate dense and sparse embeddings for the given texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple containing:
            - List of dense embeddings (vectors)
            - List of sparse embeddings (dictionaries of indices to values)
        """
        dense_embeddings = self.dense_embedder.embed(texts)
        sparse_embeddings = list(self.sparse_embedding_model.embed(texts))

        return dense_embeddings, sparse_embeddings

    def add_documents(self, documents: List[Document], min_doc_length: int = None, min_chunk_size: int = None):
        """Add documents to the vector database."""
        # Get required parameters
        min_doc_length = min_doc_length or int(get_required_env_var("MIN_DOC_LENGTH"))
        min_chunk_size = min_chunk_size or int(get_required_env_var("MIN_CHUNK_SIZE"))

        # TODO: Add a check for existing chunks to avoid duplicates
        chunks_to_ingest = []
        metadatas = []

        for doc in documents:
            if len(doc.content) < min_doc_length:
                continue
            chunks = self.chunker.chunk(doc.content)

            for chunk_id, chunk in enumerate(chunks):
                if len(chunk.text) < min_chunk_size:
                    continue
                chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                payload = {"chunk_hash": chunk_hash, "file_name": doc.file_name, "chunk_id": chunk_id}
                chunks_to_ingest.append(chunk.text)
                metadatas.append(payload)

        logging.info(f"Adding {len(chunks_to_ingest)} chunks to Qdrant collection '{self.collection_name}'")

        if chunks_to_ingest:
            # Generate embeddings for all chunks
            dense_vectors, sparse_vectors = self._get_embeddings(chunks_to_ingest)

            # Prepare points for insertion
            points = []
            for dense_vector, sparse_vector, metadata, chunk_of_text in tqdm(
                    zip(dense_vectors, sparse_vectors, metadatas, chunks_to_ingest)):
                # Create point with proper structure
                point = models.PointStruct(
                    # No ID provided - Qdrant will generate one automatically
                    id=uuid.uuid4().hex,
                    vector={
                        "dense": dense_vector,
                        "sparse": models.SparseVector(**sparse_vector.as_object())
                    },
                    payload={**metadata, "document": chunk_of_text},
                )

                points.append(point)

            # Insert points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

    def add_documents_from_files(self, file_paths: List[str], min_doc_length: int = None, min_chunk_size: int = None,
                             file_batch_size: int = 5):
        """
        Add documents from file paths to the database.
        """
        # Get required parameters
        min_doc_length = min_doc_length or int(get_required_env_var("MIN_DOC_LENGTH"))
        min_chunk_size = min_chunk_size or int(get_required_env_var("MIN_CHUNK_SIZE"))
        
        documents = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                file_content = file.read()
                documents.append(Document(file_name=Path(file_path).stem, content=file_content))
        for documents_batch in batch_list(documents, file_batch_size):
            self.add_documents(documents_batch, min_doc_length, min_chunk_size)

    def search(self, text: str, k: int = None, with_payload: bool = True):
        """
        Perform a search using only dense vectors
        
        Args:
            text: The query text
            k: Number of results to return
            with_payload: Whether to return payload with results
            
        Returns:
            Search results with scores
        """
        # Get required parameters
        k = k or int(get_required_env_var("RAG_RESULTS_COUNT"))
        
        # Generate embeddings for the query
        dense_vector = self.dense_embedder.embed([text])[0]

        # Perform the search
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using='dense',
            limit=k,
            with_payload=with_payload
        )

        return search_result

    def sparse_search(self, text: str, k: int = None, with_payload: bool = True):
        """
        Perform a search using only sparse vectors
        
        Args:
            text: The query text
            k: Number of results to return
            with_payload: Whether to return payload with results
            
        Returns:
            Search results with scores
        """
        # Get required parameters
        k = k or int(get_required_env_var("RAG_RESULTS_COUNT"))
        
        # Generate sparse embedding for the query
        sparse_vector = list(self.sparse_embedding_model.embed([text]))[0]

        # Perform the search
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values
            ),
            using='sparse',
            limit=k,
            with_payload=with_payload
        )

        return search_result

    def hybrid_search(self, text: str, k: int = None, with_payload: bool = True):
        """
        Perform a hybrid search using both dense and sparse embeddings with configurable weights
        
        Args:
            text: The query text
            k: Number of results to return
            with_payload: Whether to return payload with results
            
        Returns:
            Search results with combined scores
        """
        # Get required parameters
        k = k or int(get_required_env_var("RAG_RESULTS_COUNT"))
        
        # Generate embeddings for the query
        dense_vector = self.dense_embedder.embed([text])[0]
        sparse_vector = list(self.sparse_embedding_model.embed([text]))[0]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(**sparse_vector.as_object()),
                    using="sparse",
                    limit=20,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )

        return search_result

    def to_dict(self) -> dict:
        """
        Serialize the QdrantVectorDatabase configuration to a dictionary.
        
        Returns:
            dict: Dictionary containing the database configuration
        """
        config = {
            "collection_name": self.collection_name,
            "location": self.location,
            "sparse_embedding_model": self.sparse_embedding_model.model_name,
        }
        
        # Add dense embedder configuration
        if isinstance(self.dense_embedder, FastEmbedEmbedder):
            config["dense_embedder"] = {
                "type": "FastEmbedEmbedder",
                "model_name": self.dense_embedder.model.model_name
            }
        elif isinstance(self.dense_embedder, OpenAIEmbedder):
            config["dense_embedder"] = {
                "type": "OpenAIEmbedder",
                "model_name": self.dense_embedder.model_name
            }
        
        # Add chunker configuration
        config["chunker"] = {
            "tokenizer_name": self.chunker.tokenizer.name,  # Save the tokenizer name as a string
            "chunk_size": self.chunker.chunk_size,
            "min_characters_per_chunk": self.chunker.min_characters_per_chunk
        }
        
        return config

    @classmethod
    def from_dict(cls, config: dict) -> 'QdrantVectorDatabase':
        """
        Create a QdrantVectorDatabase instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing the database configuration
            
        Returns:
            QdrantVectorDatabase: New instance with the given configuration
        """
        # Initialize dense embedder based on configuration
        dense_embedder = None
        if "dense_embedder" in config:
            embedder_config = config["dense_embedder"]
            if embedder_config["type"] == "FastEmbedEmbedder":
                dense_embedder = FastEmbedEmbedder(model_name=embedder_config["model_name"])
            elif embedder_config["type"] == "OpenAIEmbedder":
                dense_embedder = OpenAIEmbedder(model_name=embedder_config["model_name"])
        
        # Create database instance
        db = cls(
            collection_name=config["collection_name"],
            location=config["location"],
            dense_embedder=dense_embedder,
            dense_embedder_model_name=config["dense_embedder"]["model_name"] if "dense_embedder" in config else None,
            sparse_embedding_model=config["sparse_embedding_model"]
        )
        
        # Update chunker configuration
        if "chunker" in config:
            chunker_config = config["chunker"]
            db.chunker = RecursiveChunker(
                tokenizer=chunker_config["tokenizer_name"],  # Use the saved tokenizer name
                chunk_size=chunker_config["chunk_size"],
                rules=RecursiveRules(),
                min_characters_per_chunk=chunker_config["min_characters_per_chunk"]
            )
        
        return db

    def save(self, file_path: str) -> None:
        """
        Save the database configuration to a JSON file.
        
        Args:
            file_path: Path to save the configuration file
        """
        import json
        config = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, file_path: str) -> 'QdrantVectorDatabase':
        """
        Load a QdrantVectorDatabase instance from a configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            QdrantVectorDatabase: New instance with the loaded configuration
        """
        import json
        with open(file_path, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)
