"""WebSocket endpoint for real-time FSL interpretation from webcam frames.

Protocol
--------
Client → Server  (JSON or binary):
  { "type": "frame", "data": "<base64-JPEG>", "lang": "fil" }
  { "type": "flush" }                         — force sentence flush
  { "type": "reset" }                         — reset buffer

Server → Client  (JSON):
  Prediction update:
    { "type": "prediction", "gloss": str, "confidence": float,
      "top5": [...], "committed": bool, "current_glosses": [...],
      "sentence": str, "has_hands": bool, "has_pose": bool }

  Sentence flush:
    { "type": "sentence", "glosses": [...], "sentence_fil": str,
      "sentence_en": str, "current_glosses": [] }

  Error:
    { "type": "error", "message": str }
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.ws.session import StreamSession

logger = logging.getLogger(__name__)
router = APIRouter()

# Dedicated thread pool for CPU-bound frame processing
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fsl-ws")


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = StreamSession(session_id=session_id)
    logger.info("WS session %s connected", session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "frame")

            if msg_type == "reset":
                session.reset()
                await websocket.send_json({"type": "reset_ack"})

            elif msg_type == "flush":
                response = await asyncio.get_event_loop().run_in_executor(
                    _executor, session.flush_sentence
                )
                await websocket.send_json(response)

            elif msg_type == "frame":
                b64_data = msg.get("data", "")
                lang = msg.get("lang", "fil")
                # lm: list of [{wrist_y, wrist_x, label, lms, tips, knuckles}, ...]
                lm_data = msg.get("lm", None)
                mode    = msg.get("mode", "all")
                session.lang = lang
                session.mode = mode

                try:
                    jpeg_bytes = base64.b64decode(b64_data)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid base64 frame"})
                    continue

                response = await asyncio.get_event_loop().run_in_executor(
                    _executor, session.process_frame, jpeg_bytes, lm_data
                )
                if response:
                    await websocket.send_json(response)

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("WS session %s disconnected", session_id)
    except Exception as exc:
        logger.exception("WS session %s error: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
