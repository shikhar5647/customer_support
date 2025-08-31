"""
SOP Embedding Module
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import asyncio
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class SOPEmbedding(BaseModel):
    """SOP text embedding"""
    sop_id: str
    chunk_id: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SOPEmbedder:
    """Generates embeddings for SOP chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    async def embed_sop_chunks(self, sop_id: str, chunks: List[Dict[str, Any]]) -> List[SOPEmbedding]:
        """Generate embeddings for SOP chunks"""
        try:
            embeddings = []
            
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", "")
                content = chunk.get("content", "")
                
                if not content:
                    continue
                
                # Check cache first
                cached_embedding = self._load_cached_embedding(sop_id, chunk_id)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue
                
                # Generate new embedding
                embedding_vector = await self._generate_embedding(content)
                
                sop_embedding = SOPEmbedding(
                    sop_id=sop_id,
                    chunk_id=chunk_id,
                    embedding=embedding_vector,
                    metadata={
                        "model": self.model_name,
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "chunk_size": len(content),
                        "generated_at": str(np.datetime64('now'))
                    }
                )
                
                # Cache the embedding
                self._cache_embedding(sop_embedding)
                embeddings.append(sop_embedding)
            
            self.logger.info(f"Generated {len(embeddings)} embeddings for SOP {sop_id}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to embed SOP chunks {sop_id}: {str(e)}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            # Use sentence transformers to generate embedding
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    def _get_cache_path(self, sop_id: str, chunk_id: str) -> Path:
        """Get cache file path for embedding"""
        safe_sop_id = sop_id.replace("/", "_").replace("\\", "_")
        safe_chunk_id = chunk_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_sop_id}_{safe_chunk_id}.pkl"
    
    def _cache_embedding(self, embedding: SOPEmbedding):
        """Cache embedding to disk"""
        try:
            cache_path = self._get_cache_path(embedding.sop_id, embedding.chunk_id)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {str(e)}")
    
    def _load_cached_embedding(self, sop_id: str, chunk_id: str) -> Optional[SOPEmbedding]:
        """Load cached embedding from disk"""
        try:
            cache_path = self._get_cache_path(sop_id, chunk_id)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        return None
    
    def clear_cache(self, sop_id: Optional[str] = None):
        """Clear embedding cache"""
        try:
            if sop_id:
                # Clear specific SOP cache
                pattern = f"{sop_id}_*.pkl"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                self.logger.info(f"Cleared cache for SOP {sop_id}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Cleared all embedding cache")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "total_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_directory": str(self.cache_dir)
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    def batch_embed(self, sop_chunks: List[Dict[str, Any]]) -> List[SOPEmbedding]:
        """Batch embed multiple SOP chunks"""
        try:
            texts = [chunk.get("content", "") for chunk in sop_chunks]
            sop_ids = [chunk.get("sop_id", "") for chunk in sop_chunks]
            chunk_ids = [chunk.get("chunk_id", "") for chunk in sop_chunks]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(texts)
            
            # Create SOPEmbedding objects
            sop_embeddings = []
            for i, embedding in enumerate(embeddings):
                sop_embedding = SOPEmbedding(
                    sop_id=sop_ids[i],
                    chunk_id=chunk_ids[i],
                    embedding=embedding.tolist(),
                    metadata={
                        "model": self.model_name,
                        "chunk_type": sop_chunks[i].get("chunk_type", "text"),
                        "chunk_size": len(texts[i]),
                        "generated_at": str(np.datetime64('now')),
                        "batch_processed": True
                    }
                )
                sop_embeddings.append(sop_embedding)
            
            self.logger.info(f"Batch processed {len(sop_embeddings)} embeddings")
            return sop_embeddings
            
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {str(e)}")
            return []
