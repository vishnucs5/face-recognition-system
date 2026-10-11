import json
from collections import defaultdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.schemas.generation import GenerationProgressUpdate

router = APIRouter(tags=["WebSockets"])


class ConnectionManager:
    def __init__(self):
        # Maps job_id -> list of active WebSockets
        self.active_connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if websocket in self.active_connections[job_id]:
            self.active_connections[job_id].remove(websocket)
        if not self.active_connections[job_id]:
            del self.active_connections[job_id]

    async def broadcast_progress(self, update: GenerationProgressUpdate):
        job_id = update.job_id
        if job_id in self.active_connections:
            message_text = update.model_dump_json()
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_text(message_text)
                except Exception:
                    disconnected.append(connection)
            for conn in disconnected:
                self.disconnect(conn, job_id)


manager = ConnectionManager()


@router.websocket("/ws/generations/{job_id}")
async def websocket_generation_progress(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep-alive loop; wait for client messages if any
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception:
        manager.disconnect(websocket, job_id)
