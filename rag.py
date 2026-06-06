import os
import json
import asyncio
import httpx
from typing import Optional

# Simple in-memory store for note chunks
# In production, swap with ChromaDB or Pinecone
_note_store: dict[str, list[str]] = {}


async def store_notes(submission_id: str, text: str, chunk_size: int = 500):
    """Chunk and store notes for a submission."""
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    _note_store[submission_id] = chunks
    return len(chunks)


async def retrieve_notes(submission_id: str, max_chars: int = 2000) -> str:
    """Retrieve stored notes for a submission (simple full retrieval for now)."""
    chunks = _note_store.get(submission_id, [])
    if not chunks:
        return ""
    
    combined = " ".join(chunks)
    return combined[:max_chars]


async def extract_text_from_url(file_url: str) -> str:
    """Extract text from a file URL (supports plain text and basic PDFs)."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(file_url)
        content_type = response.headers.get("content-type", "")
        
        if "text" in content_type:
            return response.text[:10000]
        
        # For PDFs, try to extract readable text
        # Basic extraction — for production use PyMuPDF or pdfplumber
        raw = response.content
        try:
            # Try UTF-8 decode, strip non-printable
            text = raw.decode("utf-8", errors="ignore")
            printable = "".join(c for c in text if c.isprintable() or c in "\n\t ")
            return printable[:10000]
        except:
            return ""
