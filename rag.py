@app.post("/index-notes")
async def index_notes(req: IndexNotesRequest):
    """Extract and index student notes for RAG retrieval."""
    text = req.text_content
    
    if not text and req.file_url:
        text = await extract_text_from_url(req.file_url)
    
    if not text:
        raise HTTPException(400, "No text content or file URL provided")
    
    chunk_count = await store_notes(req.submission_id, text)
    
    return {
        "success": True,
        "submission_id": req.submission_id,
        "chunks": chunk_count,
        "chars_indexed": len(text),
  }

Commit message: Add rag.py
