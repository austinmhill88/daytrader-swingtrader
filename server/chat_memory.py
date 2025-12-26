"""
Simple in-memory chat session store.
Each session keeps a rolling list of messages for the model (role: system|user|assistant).
"""
from typing import Dict, List, Any, Optional
from uuid import uuid4

class ChatMemory:
    def __init__(self, max_messages: int = 100):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_messages = max_messages

    def open(self, system_context: Optional[str] = None) -> str:
        sid = uuid4().hex
        self._sessions[sid] = []
        if system_context:
            self._sessions[sid].append({"role": "system", "content": system_context})
        return sid

    def add_user(self, sid: str, content: str):
        self._ensure(sid)
        self._sessions[sid].append({"role": "user", "content": content})
        self._trim(sid)

    def add_assistant(self, sid: str, content: str):
        self._ensure(sid)
        self._sessions[sid].append({"role": "assistant", "content": content})
        self._trim(sid)

    def messages(self, sid: str) -> List[Dict[str, str]]:
        self._ensure(sid)
        return list(self._sessions[sid])

    def inject_system(self, sid: str, content: str):
        self._ensure(sid)
        self._sessions[sid].append({"role": "system", "content": content})
        self._trim(sid)

    def _ensure(self, sid: str):
        if sid not in self._sessions:
            self._sessions[sid] = []

    def _trim(self, sid: str):
        msgs = self._sessions.get(sid, [])
        if len(msgs) > self.max_messages:
            self._sessions[sid] = msgs[-self.max_messages:]