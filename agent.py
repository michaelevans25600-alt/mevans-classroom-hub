
### 📁 File Structure

thinkspace agent/
 agent.py           Main LiveKit agent
 rag.py             Note scanning / RAG logic  
 api.py             HTTP endpoints (token, upload)
 requirements.txt   Dependencies
 Procfile           For Railway/Render
 .env.example       Environment variables template
 README.md          Deploy instructions




### `agent.py`
```python
import asyncio
import os
import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, google, silero
from rag import retrieve_notes

load_dotenv()
logger = logging.getLogger("thinkspace-agent")

PERSONAS = {
    "dr_amara": {
        "name": "Dr. Amara",
        "prompt": """You are Dr. Amara, a warm and intellectually rigorous academic tutor. 
        You specialize in Socratic questioning — you never just give answers, you guide students 
        to discover them. You are encouraging, precise, and push students to think deeper.
        Keep responses concise (2-3 sentences max) for natural conversation flow.
        When student notes are provided, reference them directly to show you've read their work."""
    },
    "marcus": {
        "name": "Marcus",
        "prompt": """You are Marcus, a sharp peer-level academic challenger. 
        You debate ideas constructively, push back on weak arguments, and celebrate strong reasoning.
        You speak like a brilliant fellow student, not a teacher. Be direct and energetic.
        Keep responses concise (2-3 sentences max) for natural conversation flow."""
    },
    "sofia": {
        "name": "Sofia",
        "prompt": """You are Sofia, a creative and interdisciplinary thinker who connects ideas 
        across subjects. You help students see unexpected links between concepts.
        You are enthusiastic, curious, and inspiring.
        Keep responses concise (2-3 sentences max) for natural conversation flow."""
    },
    "kenji": {
        "name": "Kenji",
        "prompt": """You are Kenji, a methodical and precise academic examiner. 
        You focus on evidence, logical structure, and academic rigor.
        You ask probing questions about sources and methodology.
        Keep responses concise (2-3 sentences max) for natural conversation flow."""
    },
    "zara": {
        "name": "Zara",
        "prompt": """You are Zara, an empathetic learning coach who focuses on building 
        student confidence and metacognitive skills. You help students understand HOW they learn.
        You are patient, reflective, and nurturing.
        Keep responses concise (2-3 sentences max) for natural conversation flow."""
    },
}

def build_system_prompt(persona_id: str, subject: str, notes_context: str, mode: str) -> str:
    persona = PERSONAS.get(persona_id, PERSONAS["dr_amara"])
    
    base_prompt = persona["prompt"]
    
    mode_instruction = ""
    if mode == "group":
        mode_instruction = "\nThis is a GROUP session. Multiple students are present. Address the group, manage turn-taking naturally, and encourage all participants to contribute."
    else:
        mode_instruction = "\nThis is a 1-on-1 session. Focus entirely on this individual student's development."
    
    notes_section = ""
    if notes_context:
        notes_section = f"\n\n--- STUDENT'S NOTES (you have read these) ---\n{notes_context}\n--- END OF NOTES ---\nReference these notes naturally in your responses to show you understand their specific work."
    
    subject_section = f"\n\nSubject area: {subject}" if subject else ""
    
    return f"{base_prompt}{mode_instruction}{subject_section}{notes_section}"


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Extract metadata from room
    room_meta = {}
    if ctx.room.metadata:
        import json
        try:
            room_meta = json.loads(ctx.room.metadata)
        except:
            pass
    
    persona_id = room_meta.get("persona", "dr_amara")
    subject = room_meta.get("subject", "General")
    session_id = room_meta.get("session_id", "")
    submission_id = room_meta.get("submission_id", "")
    mode = room_meta.get("mode", "1on1")
    
    # RAG: retrieve relevant notes
    notes_context = ""
    if submission_id:
        try:
            notes_context = await retrieve_notes(submission_id)
            logger.info(f"Retrieved notes context: {len(notes_context)} chars")
        except Exception as e:
            logger.warning(f"Could not retrieve notes: {e}")
    
    system_prompt = build_system_prompt(persona_id, subject, notes_context, mode)
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    persona = PERSONAS.get(persona_id, PERSONAS["dr_amara"])
    
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(
            min_silence_duration=0.6,
            min_speech_duration=0.1,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en-US",
            punctuate=True,
            smart_format=True,
        ),
        llm=google.LLM(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
        ),
        tts=google.TTS(
            voice_name="en-US-Journey-F",
            speaking_rate=1.0,
        ),
        chat_ctx=initial_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        min_endpointing_delay=0.5,
    )
    
    agent.start(ctx.room)
    
    # Greet the student(s)
    greetings = {
        "dr_amara": f"Hello! I'm Dr. Amara. I've reviewed your notes on {subject}. What aspect would you like to explore today?",
        "marcus": f"Hey! Marcus here. Ready to dig into {subject}? Let's see what you've got.",
        "sofia": f"Hi there! I'm Sofia. I love {subject} — there are so many fascinating connections to explore. Where shall we begin?",
        "kenji": f"Good day. I'm Kenji. I've examined your work on {subject}. Let's discuss your methodology and evidence.",
        "zara": f"Hi! I'm Zara. I'm here to support your learning journey in {subject}. How are you feeling about the material?",
    }
    
    greeting = greetings.get(persona_id, greetings["dr_amara"])
    await agent.say(greeting, allow_interruptions=True)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
```

---

### `rag.py`
```python
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
```

---

### `api.py`
```python
import os
import json
import asyncio
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit.api import AccessToken, VideoGrants
from rag import store_notes, extract_text_from_url
import uvicorn

app = FastAPI(title="ThinkSpace Agent API")

# Allow requests from your Base44 app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Base44 app URL in production
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
    token = AccessToken(LIVEKIT API KEY, LIVEKIT API SECRET)
    token.with_identity(req.participant_identity)
    token.with_name(req.participant_name)
    token.with_grants(VideoGrants(
        room join=True,
        room=req.room_name,
        can publish=True,
        can subscribe=True,
    ))
    token.with_metadata(json.dumps(req.metadata))
    token.with_ttl(timedelta(hours=2))
    
    return {
        "token": token.to_jwt(),
        "room": req.room_name,
        "url": LIVEKIT URL,
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
        "submission id": req.submission_id,
        "chunks": chunk_count,
        "chars indexed": len(text),
    }


if name == "main":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
```

---

### `requirements.txt`
```
livekit-agents[deepgram,google,silero]>=0.8.0
livekit-api>=0.7.0
fastapi>=0.111.0
uvicorn>=0.30.0
python-dotenv>=1.0.0
httpx>=0.27.0
pydantic>=2.0.0
```

---

### `Procfile`
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
worker: python agent.py start
```

---

### `.env.example`
```bash
LIVEKIT URL=wss://your-project.livekit.cloud
LIVEKIT API KEY=your_livekit_api_key
LIVEKIT API SECRET=your_livekit_api_secret
DEEPGRAM_API_KEY=your_deepgram_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
```

---

### `README.md`
```markdown
# ThinkSpace Agent Server

## Deploy to Railway (Recommended)

1. Create a free account at railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Push this code to a GitHub repo and connect it
4. Add environment variables in Railway dashboard:
    LIVEKIT_URL
    LIVEKIT_API_KEY  
    LIVEKIT_API_SECRET
    DEEPGRAM_API_KEY
    GOOGLE_API_KEY
5. Railway will auto-detect Python and deploy
6. Copy your Railway URL (e.g. https://thinkspace-agent.up.railway.app)
7. Paste it into your Base44 app settings

## Deploy to Render

1. Create free account at render.com
2. New  Web Service : Connect GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy!

## Endpoints

 GET  /health:         Check server is running
 POST /token:          Get LiveKit room token
 POST /index-notes:    Index student notes for RAG

commit message:Update agent.py with personas and RAG
