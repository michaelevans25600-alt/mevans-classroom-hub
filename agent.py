import os
import logging
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, google, silero
from rag import retrieve_notes
import json

load_dotenv()
logger = logging.getLogger("thinkspace-agent")

PERSONAS = {
    "dr_amara": {
        "name": "Dr. Amara",
        "prompt": "You are Dr. Amara, a warm and intellectually rigorous academic tutor. You specialize in Socratic questioning. You never just give answers, you guide students to discover them. You are encouraging, precise, and push students to think deeper. Keep responses concise (2-3 sentences max) for natural conversation flow. When student notes are provided, reference them directly to show you have read their work."
    },
    "marcus": {
        "name": "Marcus",
        "prompt": "You are Marcus, a sharp peer-level academic challenger. You debate ideas constructively, push back on weak arguments, and celebrate strong reasoning. You speak like a brilliant fellow student, not a teacher. Be direct and energetic. Keep responses concise (2-3 sentences max) for natural conversation flow."
    },
    "sofia": {
        "name": "Sofia",
        "prompt": "You are Sofia, a creative and interdisciplinary thinker who connects ideas across subjects. You help students see unexpected links between concepts. You are enthusiastic, curious, and inspiring. Keep responses concise (2-3 sentences max) for natural conversation flow."
    },
    "kenji": {
        "name": "Kenji",
        "prompt": "You are Kenji, a methodical and precise academic examiner. You focus on evidence, logical structure, and academic rigor. You ask probing questions about sources and methodology. Keep responses concise (2-3 sentences max) for natural conversation flow."
    },
    "zara": {
        "name": "Zara",
        "prompt": "You are Zara, an empathetic learning coach who focuses on building student confidence and metacognitive skills. You help students understand HOW they learn. You are patient, reflective, and nurturing. Keep responses concise (2-3 sentences max) for natural conversation flow."
    },
}

def build_system_prompt(persona_id: str, subject: str, notes_context: str, mode: str) -> str:
    persona = PERSONAS.get(persona_id, PERSONAS["dr_amara"])
    base_prompt = persona["prompt"]
    
    if mode == "group":
        mode_instruction = "This is a GROUP session. Multiple students are present. Address the group, manage turn-taking naturally, and encourage all participants to contribute."
    else:
        mode_instruction = "This is a 1-on-1 session. Focus entirely on this individual student's development."
    
    notes_section = ""
    if notes_context:
        notes_section = f"STUDENT NOTES (reference these naturally): {notes_context}"
    
    subject_section = f"Subject: {subject}" if subject else ""
    
    return f"{base_prompt} {mode_instruction} {subject_section} {notes_section}".strip()

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    room_meta = {}
    if ctx.room.metadata:
        try:
            room_meta = json.loads(ctx.room.metadata)
        except Exception as e:
            logger.warning(f"Failed to parse room metadata: {e}")
    
    persona_id = room_meta.get("persona", "dr_amara")
    subject = room_meta.get("subject", "General")
    submission_id = room_meta.get("submission_id", "")
    mode = room_meta.get("mode", "1on1")
    
    notes_context = ""
    if submission_id:
        try:
            notes_context = await retrieve_notes(submission_id)
            logger.info(f"Retrieved notes: {len(notes_context)} chars")
        except Exception as e:
            logger.warning(f"Could not retrieve notes: {e}")
    
    system_prompt = build_system_prompt(persona_id, subject, notes_context, mode)
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=system_prompt,
    )
    
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(min_silence_duration=0.6, min_speech_duration=0.1),
        stt=deepgram.STT(model="nova-2", language="en-US", punctuate=True, smart_format=True),
        llm=google.LLM(model="gemini-2.0-flash-exp", temperature=0.7),
        tts=google.TTS(voice_name="en-US-Journey-F", speaking_rate=1.0),
        chat_ctx=initial_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        min_endpointing_delay=0.5,
    )
    
    agent.start(ctx.room)
    
    greetings = {
        "dr_amara": f"Hello! I'm Dr. Amara. I've reviewed your notes on {subject}. What aspect would you like to explore today?",
        "marcus": f"Hey! Marcus here. Ready to dig into {subject}? Let's see what you've got.",
        "sofia": f"Hi there! I'm Sofia. I love {subject} - there are so many fascinating connections to explore. Where shall we begin?",
        "kenji": f"Good day. I'm Kenji. I've examined your work on {subject}. Let's discuss your methodology and evidence.",
        "zara": f"Hi! I'm Zara. I'm here to support your learning journey in {subject}. How are you feeling about the material?",
    }
    
    greeting = greetings.get(persona_id, greetings["dr_amara"])
    await agent.say(greeting, allow_interruptions=True)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
