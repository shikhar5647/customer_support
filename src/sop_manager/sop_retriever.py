"""
SOP Retrieval Module
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import asyncio
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class RetrievalResult(BaseModel):
    """Result of SOP retrieval"""
    sop_id: str
    chunk_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SOPRetriever:
    """Retrieves relevant SOP chunks based on similarity"""
    
    def __init__(self, embeddings_dir: Optional[str] = None):
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else Path("embeddings_cache")
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded embeddings
        self.embeddings_cache = {}
        self.sop_metadata_cache = {}
    
    async def search_similar_sops(
        self, 
        query: str, 
        domain: str = "general",
        intent: str = "",
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """Search for similar SOPs based on query"""
        try:
            # Load embeddings for the domain
            domain_embeddings = await self._load_domain_embeddings(domain)
            
            if not domain_embeddings:
                self.logger.warning(f"No embeddings found for domain: {domain}")
                return []
            
            # Generate query embedding (simplified - in production use proper embedding)
            query_embedding = self._generate_simple_embedding(query)
            
            # Calculate similarities
            similarities = []
            for sop_id, chunks in domain_embeddings.items():
                for chunk in chunks:
                    similarity = self._calculate_similarity(query_embedding, chunk["embedding"])
                    if similarity >= similarity_threshold:
                        similarities.append({
                            "sop_id": sop_id,
                            "chunk_id": chunk["chunk_id"],
                            "content": chunk["content"],
                            "similarity_score": similarity,
                            "metadata": chunk.get("metadata", {})
                        })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = similarities[:top_k]
            
            # Convert to RetrievalResult objects
            results = [
                RetrievalResult(**result) for result in top_results
            ]
            
            self.logger.info(f"Retrieved {len(results)} similar SOPs for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"SOP retrieval failed: {str(e)}")
            return []
    
    async def _load_domain_embeddings(self, domain: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load embeddings for a specific domain"""
        try:
            if domain in self.embeddings_cache:
                return self.embeddings_cache[domain]
            
            # Load embeddings from files
            embeddings = {}
            pattern = f"*_{domain}_*.pkl" if domain != "general" else "*.pkl"
            
            for embedding_file in self.embeddings_dir.glob(pattern):
                try:
                    # Parse filename to extract SOP ID
                    filename = embedding_file.stem
                    parts = filename.split("_")
                    
                    if len(parts) >= 2:
                        sop_id = parts[0]
                        chunk_id = "_".join(parts[1:])
                        
                        # Load embedding data
                        embedding_data = self._load_embedding_file(embedding_file)
                        if embedding_data:
                            if sop_id not in embeddings:
                                embeddings[sop_id] = []
                            
                            embeddings[sop_id].append({
                                "chunk_id": chunk_id,
                                "embedding": embedding_data.embedding,
                                "content": embedding_data.metadata.get("content", ""),
                                "metadata": embedding_data.metadata
                            })
                
                except Exception as e:
                    self.logger.warning(f"Failed to load embedding file {embedding_file}: {str(e)}")
                    continue
            
            # Cache the results
            self.embeddings_cache[domain] = embeddings
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to load domain embeddings for {domain}: {str(e)}")
            return {}
    
    def _load_embedding_file(self, file_path: Path) -> Optional[Any]:
        """Load embedding from pickle file"""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load embedding file {file_path}: {str(e)}")
            return None
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate simple embedding for query (fallback method)"""
        # This is a simplified embedding method
        # In production, use the same embedding model as SOPs
        
        # Simple TF-IDF like approach
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create a simple vector (this is just for demonstration)
        # In reality, you'd use the same embedding model
        embedding = [0.0] * 384  # Default dimension
        
        # Simple hash-based embedding
        for i, word in enumerate(word_freq.keys()):
            hash_val = hash(word) % 384
            embedding[hash_val] = word_freq[word] * 0.1
        
        return embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    async def get_sop_by_id(self, sop_id: str) -> Optional[Dict[str, Any]]:
        """Get SOP by ID"""
        try:
            # Check metadata cache first
            if sop_id in self.sop_metadata_cache:
                return self.sop_metadata_cache[sop_id]
            
            # Load SOP metadata
            sop_file = Path(f"data/sops/{sop_id}.json")
            if sop_file.exists():
                with open(sop_file, 'r', encoding='utf-8') as f:
                    sop_data = json.load(f)
                
                # Cache the metadata
                self.sop_metadata_cache[sop_id] = sop_data
                return sop_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get SOP {sop_id}: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear all caches"""
        self.embeddings_cache.clear()
        self.sop_metadata_cache.clear()
        self.logger.info("Cleared SOP retriever caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "embeddings_cache_size": len(self.embeddings_cache),
            "metadata_cache_size": len(self.sop_metadata_cache),
            "total_cached_embeddings": sum(len(chunks) for chunks in self.embeddings_cache.values())
        }
    
    async def batch_search(
        self, 
        queries: List[str], 
        domain: str = "general",
        top_k: int = 3
    ) -> List[List[RetrievalResult]]:
        """Batch search for multiple queries"""
        try:
            results = []
            for query in queries:
                query_results = await self.search_similar_sops(
                    query=query,
                    domain=domain,
                    top_k=top_k
                )
                results.append(query_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch search failed: {str(e)}")
            return [[] for _ in queries]
