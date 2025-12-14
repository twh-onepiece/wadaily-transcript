import asyncio
import numpy as np
import base64
import uuid
import os
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline
from contextlib import asynccontextmanager
import logging
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = None
inference_semaphore: asyncio.Semaphore = None

INPUT_SAMPLE_RATE = 24000
MODEL_SAMPLE_RATE = 16000

DEVICE = os.getenv("DEVICE", "cuda")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, inference_semaphore

    logger.info("Loading kotoba-whisper-v2.0...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.0",
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    inference_semaphore = asyncio.Semaphore(4)
    logger.info("Ready")
    yield


app = FastAPI(lifespan=lifespan)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return signal.resample_poly(audio, target_sr, orig_sr).astype(np.float32)


class StreamingTranscriber:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.min_samples = int(INPUT_SAMPLE_RATE * 0.8)
        self.max_samples = int(INPUT_SAMPLE_RATE * 4.0)
        self.silence_samples = int(INPUT_SAMPLE_RATE * 0.3)

    def add(self, audio_base64: str):
        audio_bytes = base64.b64decode(audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, audio])

    def should_process(self) -> bool:
        if len(self.buffer) >= self.max_samples:
            return True
        if len(self.buffer) >= self.min_samples:
            tail = self.buffer[-self.silence_samples:]
            return np.abs(tail).mean() < 0.008
        return False

    def get_chunk(self) -> np.ndarray:
        chunk_24k = self.buffer[:self.max_samples]
        self.buffer = self.buffer[len(chunk_24k):]
        return resample(chunk_24k, INPUT_SAMPLE_RATE, MODEL_SAMPLE_RATE)

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)


def create_event(event_type: str, **kwargs) -> dict:
    return {
        "type": event_type,
        "event_id": f"event_{uuid.uuid4().hex[:24]}",
        **kwargs
    }


async def do_transcribe(audio: np.ndarray) -> str:
    result = await asyncio.to_thread(
        pipe,
        {"array": audio, "sampling_rate": MODEL_SAMPLE_RATE},
    )
    return result["text"]


@app.websocket("/")
async def realtime_transcribe(websocket: WebSocket):
    await websocket.accept()

    transcriber = StreamingTranscriber()
    session_id = f"sess_{uuid.uuid4().hex[:24]}"

    logger.info(f"[{session_id}] Connected")

    await websocket.send_json(create_event(
        "session.created",
        session={
            "id": session_id,
            "model": "kotoba-whisper-v2.0",
            "input_audio_format": "pcm16",
            "input_audio_sample_rate": INPUT_SAMPLE_RATE,
        }
    ))

    try:
        while True:
            msg = await websocket.receive_json()
            event_type = msg.get("type")

            if event_type == "input_audio_buffer.append":
                audio_base64 = msg.get("audio", "")
                if audio_base64:
                    transcriber.add(audio_base64)

                    if transcriber.should_process():
                        chunk = transcriber.get_chunk()
                        item_id = f"item_{uuid.uuid4().hex[:24]}"

                        await websocket.send_json(create_event(
                            "input_audio_buffer.speech_started",
                            audio_start_ms=0,
                            item_id=item_id,
                        ))

                        async with inference_semaphore:
                            text = await do_transcribe(chunk)

                        await websocket.send_json(create_event(
                            "input_audio_buffer.speech_stopped",
                            audio_end_ms=int(len(chunk) / MODEL_SAMPLE_RATE * 1000),
                            item_id=item_id,
                        ))

                        if text.strip():
                            await websocket.send_json(create_event(
                                "conversation.item.input_audio_transcription.completed",
                                item_id=item_id,
                                content_index=0,
                                transcript=text.strip(),
                            ))

            elif event_type == "input_audio_buffer.commit":
                if len(transcriber.buffer) > 0:
                    chunk = transcriber.get_chunk()
                    item_id = f"item_{uuid.uuid4().hex[:24]}"

                    async with inference_semaphore:
                        text = await do_transcribe(chunk)

                    await websocket.send_json(create_event(
                        "input_audio_buffer.committed",
                        item_id=item_id,
                    ))

                    if text.strip():
                        await websocket.send_json(create_event(
                            "conversation.item.input_audio_transcription.completed",
                            item_id=item_id,
                            content_index=0,
                            transcript=text.strip(),
                        ))

            elif event_type == "input_audio_buffer.clear":
                transcriber.clear()
                await websocket.send_json(create_event(
                    "input_audio_buffer.cleared"
                ))

            elif event_type == "session.update":
                await websocket.send_json(create_event(
                    "session.updated",
                    session={"id": session_id, "model": "kotoba-whisper-v2.0"}
                ))

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Disconnected")
    except Exception as e:
        logger.error(f"[{session_id}] Error: {e}")
        await websocket.send_json(create_event(
            "error",
            error={"type": "server_error", "message": str(e)}
        ))


@app.get("/health")
async def health():
    return {"status": "ok"}
