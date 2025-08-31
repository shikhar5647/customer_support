"""
SOP Processing Module
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import asyncio
from pathlib import Path
import json

from ..sop_manager.sop_loader import SOPLoader
from ..sop_manager.sop_chunker import SOPChunker
from ..sop_manager.sop_embedder import SOPEmbedder
from ..sop_manager.sop_retriever import SOPRetriever
from ..sop_manager.sop_validator import SOPValidator

logger = logging.getLogger(__name__)

class SOPProcessingResult(BaseModel):
    """Result of SOP processing"""
    sop_id: str
    processing_status: str  # loaded, chunked, embedded, validated, completed
    chunks_count: int = 0
    embeddings_count: int = 0
    validation_score: float = 0.0
    processing_time: float = 0.0
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SOPProcessor:
    """Main processor for SOP documents"""
    
    def __init__(self, data_dir: str = "data/sops"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sop_loader = SOPLoader()
        self.sop_chunker = SOPChunker()
        self.sop_embedder = SOPEmbedder()
        self.sop_retriever = SOPRetriever()
        self.sop_validator = SOPValidator()
        
        # Processing cache
        self.processed_sops = {}
    
    async def process_sop(self, sop_id: str) -> SOPProcessingResult:
        """Process a single SOP document"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Load SOP
            sop_data = await self.sop_loader.load_sop(sop_id)
            if not sop_data:
                return SOPProcessingResult(
                    sop_id=sop_id,
                    processing_status="failed",
                    errors=[f"Failed to load SOP {sop_id}"]
                )
            
            # Chunk SOP
            chunks = self.sop_chunker.chunk_sop(sop_data.content, sop_id)
            if not chunks:
                return SOPProcessingResult(
                    sop_id=sop_id,
                    processing_status="failed",
                    errors=[f"Failed to chunk SOP {sop_id}"]
                )
            
            # Generate embeddings
            chunk_data = [{"chunk_id": chunk.chunk_id, "content": chunk.content} for chunk in chunks]
            embeddings = await self.sop_embedder.embed_sop_chunks(sop_id, chunk_data)
            
            # Validate SOP
            validation_result = await self.sop_validator.validate_sop(sop_data.dict())
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create result
            result = SOPProcessingResult(
                sop_id=sop_id,
                processing_status="completed",
                chunks_count=len(chunks),
                embeddings_count=len(embeddings),
                validation_score=validation_result.validation_score,
                processing_time=processing_time,
                errors=validation_result.errors,
                metadata={
                    "chunk_types": [chunk.chunk_type for chunk in chunks],
                    "validation_warnings": validation_result.warnings,
                    "embedding_model": self.sop_embedder.model_name
                }
            )
            
            # Cache result
            self.processed_sops[sop_id] = result
            
            self.logger.info(f"Successfully processed SOP {sop_id}: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process SOP {sop_id}: {str(e)}")
            return SOPProcessingResult(
                sop_id=sop_id,
                processing_status="failed",
                errors=[f"Processing error: {str(e)}"]
            )
    
    async def process_domain_sops(self, domain: str) -> List[SOPProcessingResult]:
        """Process all SOPs for a specific domain"""
        try:
            # Get all SOPs for the domain
            sop_files = list(self.data_dir.glob(f"*_{domain}_*.json"))
            sop_files.extend(list(self.data_dir.glob(f"{domain}_*.json")))
            
            if not sop_files:
                self.logger.warning(f"No SOP files found for domain: {domain}")
                return []
            
            # Process each SOP
            results = []
            for sop_file in sop_files:
                sop_id = sop_file.stem
                result = await self.process_sop(sop_id)
                results.append(result)
            
            self.logger.info(f"Processed {len(results)} SOPs for domain {domain}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process domain SOPs for {domain}: {str(e)}")
            return []
    
    async def batch_process_sops(self, sop_ids: List[str]) -> List[SOPProcessingResult]:
        """Process multiple SOPs in batch"""
        try:
            results = []
            for sop_id in sop_ids:
                result = await self.process_sop(sop_id)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return []
    
    async def search_sops(
        self, 
        query: str, 
        domain: str = "general",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant SOPs"""
        try:
            # Use the retriever to search
            results = await self.sop_retriever.search_similar_sops(
                query=query,
                domain=domain,
                top_k=top_k
            )
            
            # Convert to dictionary format
            return [result.dict() for result in results]
            
        except Exception as e:
            self.logger.error(f"SOP search failed: {str(e)}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.processed_sops:
            return {}
        
        total_sops = len(self.processed_sops)
        completed_sops = sum(1 for r in self.processed_sops.values() if r.processing_status == "completed")
        failed_sops = total_sops - completed_sops
        
        total_chunks = sum(r.chunks_count for r in self.processed_sops.values())
        total_embeddings = sum(r.embeddings_count for r in self.processed_sops.values())
        avg_validation_score = sum(r.validation_score for r in self.processed_sops.values()) / total_sops if total_sops > 0 else 0
        
        return {
            "total_sops": total_sops,
            "completed_sops": completed_sops,
            "failed_sops": failed_sops,
            "success_rate": completed_sops / total_sops if total_sops > 0 else 0,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "average_validation_score": avg_validation_score,
            "average_processing_time": sum(r.processing_time for r in self.processed_sops.values()) / total_sops if total_sops > 0 else 0
        }
    
    def clear_cache(self):
        """Clear processing cache"""
        self.processed_sops.clear()
        self.sop_embedder.clear_cache()
        self.sop_retriever.clear_cache()
        self.logger.info("Cleared SOP processing cache")
    
    async def validate_all_sops(self) -> Dict[str, Any]:
        """Validate all processed SOPs"""
        try:
            validation_results = []
            
            for sop_id in self.processed_sops.keys():
                # Load SOP data
                sop_data = await self.sop_loader.load_sop(sop_id)
                if sop_data:
                    validation_result = await self.sop_validator.validate_sop(sop_data.dict())
                    validation_results.append(validation_result)
            
            # Get validation summary
            summary = self.sop_validator.get_validation_summary(validation_results)
            
            self.logger.info(f"Validation completed: {summary.get('valid_sops', 0)}/{summary.get('total_sops', 0)} SOPs valid")
            return summary
            
        except Exception as e:
            self.logger.error(f"SOP validation failed: {str(e)}")
            return {}
    
    def export_processing_report(self, file_path: str = "sop_processing_report.json"):
        """Export processing report to file"""
        try:
            report = {
                "processing_stats": self.get_processing_stats(),
                "processed_sops": {sop_id: result.dict() for sop_id, result in self.processed_sops.items()},
                "export_timestamp": str(asyncio.get_event_loop().time()),
                "system_info": {
                    "embedding_model": self.sop_embedder.model_name,
                    "chunker_config": {
                        "max_chunk_size": self.sop_chunker.max_chunk_size,
                        "overlap": self.sop_chunker.overlap
                    }
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"Processing report exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export processing report: {str(e)}")
            return False
