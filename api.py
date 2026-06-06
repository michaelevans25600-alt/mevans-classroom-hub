import os
import json
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit.api import AccessToken, VideoGrants
from rag import store_notes, extract_text_from_url
import uvicorn

app = FastAPI(title="ThinkSpace Agent API")

# Allow requests from your frontend app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]


class TokenRequest(BaseModel):
    room_name: str
    participant_name: str
    participant_identity: str
    metadata: dict = {}


class IndexNotesRequest(BaseModel):
    submission_id: str
    file_url: str = ""
    text_content: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "service": "thinkspace-agent"}


@app.post("/token")
async def get_token(req: TokenRequest):
    """Generate a LiveKit room token for a participant."""
    token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(req.participant_identity)
    token.with_name(req.participant_name)
    token.with_grants(VideoGrants(
        room_join=True,
        room=req.room_name,
        can_publish=True,
        can_subscribe=True,
    ))
    token.with_metadata(json.dumps(req.metadata))
    token.with_ttl(timedelta(hours=2))
    
    return {
        "token": token.to_jwt(),
        "room": req.room_name,
        "url": LIVEKIT_URL,
    }


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
