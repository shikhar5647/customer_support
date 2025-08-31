"""
SOP Text Chunking Module
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SOPChunk(BaseModel):
    """Individual chunk of SOP text"""
    chunk_id: str
    content: str
    chunk_type: str = "text"  # text, header, list, table, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    position: int = 0

class SOPChunker:
    """Chunks SOP documents into smaller, manageable pieces"""
    
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_sop(self, sop_content: str, sop_id: str) -> List[SOPChunk]:
        """Chunk SOP content into smaller pieces"""
        try:
            chunks = []
            
            # Split by sections (headers)
            sections = self._split_by_sections(sop_content)
            
            for section_idx, section in enumerate(sections):
                section_chunks = self._chunk_section(section, section_idx)
                chunks.extend(section_chunks)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.chunk_id = f"{sop_id}_chunk_{i:03d}"
                chunk.position = i
                chunk.metadata.update({
                    "sop_id": sop_id,
                    "total_chunks": len(chunks),
                    "chunk_index": i
                })
            
            self.logger.info(f"Chunked SOP {sop_id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk SOP {sop_id}: {str(e)}")
            return []
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by section headers"""
        # Common header patterns
        header_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^[A-Z][A-Z\s]+\n[-=]+\n',  # Underlined headers
            r'^\d+\.\s+[A-Z][^.]*\.',  # Numbered sections
            r'^[A-Z][^.]*\.',  # Capitalized sentences
        ]
        
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            is_header = any(re.match(pattern, line) for pattern in header_patterns)
            
            if is_header and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _chunk_section(self, section: str, section_idx: int) -> List[SOPChunk]:
        """Chunk a section into smaller pieces"""
        chunks = []
        
        # If section is small enough, keep as one chunk
        if len(section) <= self.max_chunk_size:
            chunks.append(SOPChunk(
                content=section,
                chunk_type="section",
                metadata={"section_index": section_idx}
            ))
            return chunks
        
        # Split by paragraphs
        paragraphs = section.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                chunks.append(SOPChunk(
                    content=current_chunk.strip(),
                    chunk_type="paragraph",
                    metadata={"section_index": section_idx}
                ))
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add remaining content
        if current_chunk:
            chunks.append(SOPChunk(
                content=current_chunk.strip(),
                chunk_type="paragraph",
                metadata={"section_index": section_idx}
            ))
        
        return chunks
    
    def merge_chunks(self, chunks: List[SOPChunk], max_size: Optional[int] = None) -> List[SOPChunk]:
        """Merge small chunks to optimize size"""
        if not chunks:
            return []
        
        max_size = max_size or self.max_chunk_size
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            combined_size = len(current_chunk.content) + len(next_chunk.content)
            
            if combined_size <= max_size:
                # Merge chunks
                current_chunk.content += "\n\n" + next_chunk.content
                current_chunk.metadata["merged_chunks"] = current_chunk.metadata.get("merged_chunks", 0) + 1
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def get_chunk_statistics(self, chunks: List[SOPChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        chunk_types = [chunk.chunk_type for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_type_distribution": {chunk_type: chunk_types.count(chunk_type) for chunk_type in set(chunk_types)},
            "total_content_length": sum(chunk_sizes)
        }