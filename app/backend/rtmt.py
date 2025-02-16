import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger("voicerag")

class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2

class ToolResult:
    text: str
    destination: ToolResultDirection

    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if type(self.text) == str else json.dumps(self.text)

class Tool:
    target: Callable[..., ToolResult]
    schema: Any

    def __init__(self, target: Any, schema: Any):
        self.target = target
        self.schema = schema

class RTToolCall:
    tool_call_id: str
    previous_id: str

    def __init__(self, tool_call_id: str, previous_id: str):
        self.tool_call_id = tool_call_id
        self.previous_id = previous_id

class RTMiddleTier:
    tools: dict[str, Tool] = {}
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    disable_audio: Optional[bool] = None
    voice_choice: Optional[str] = None
    model = None

    def __init__(self, model=None, voice_choice: Optional[str] = None):
        self.model = model
        self.voice_choice = voice_choice
        if voice_choice is not None:
            logger.info("Realtime voice choice set to %s", voice_choice)

    async def _websocket_handler(self, request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Handle different message types
                    if data["type"] == "chat":
                        response = await self._handle_chat(data["message"])
                        await ws.send_json({"type": "response", "text": response})
                else:
                    logger.warning(f"Unexpected message type: {msg.type}")
        
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            return ws
    
    async def _handle_chat(self, message: str) -> str:
        try:
            response = await self.model.generate_content(message)
            return response.text
        except Exception as e:
            logger.error(f"Chat handling error: {str(e)}")
            return "Sorry, I encountered an error processing your request."
    
    def attach_to_app(self, app, path):
        app.router.add_get(path, self._websocket_handler)