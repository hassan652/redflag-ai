"""
Playground session management and rate limiting.

Handles:
- Max concurrent sessions (10)
- Per-session file upload with size/count limits
- Session cleanup after timeout
- Simple token-based access (no full auth for beta)
"""

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

MAX_CONCURRENT_SESSIONS = 10
SESSION_TIMEOUT_SECONDS = 900  # 15 minutes
MAX_UPLOAD_SIZE_MB = 50  # Per file
MAX_FILES_PER_SESSION = 20
UPLOAD_BASE_DIR = Path(os.environ.get("REDFLAG_UPLOAD_DIR", "/tmp/redflag_sessions"))


# =============================================================================
# Session Model
# =============================================================================


@dataclass
class PlaygroundSession:
    """A single user's playground session."""

    session_id: str
    created_at: float
    last_active: float
    upload_dir: Path
    files_uploaded: int = 0
    queries_made: int = 0
    is_processing: bool = False
    chat_history: list[dict[str, str]] = field(default_factory=list)


# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """Manages concurrent playground sessions with limits."""

    def __init__(self):
        self._sessions: dict[str, PlaygroundSession] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def create_session(self) -> PlaygroundSession | None:
        """
        Create a new session. Returns None if at capacity.
        """
        async with self._lock:
            # Clean expired sessions first
            self._cleanup_expired()

            if len(self._sessions) >= MAX_CONCURRENT_SESSIONS:
                return None

            session_id = uuid.uuid4().hex[:12]
            upload_dir = UPLOAD_BASE_DIR / session_id
            upload_dir.mkdir(parents=True, exist_ok=True)

            session = PlaygroundSession(
                session_id=session_id,
                created_at=time.time(),
                last_active=time.time(),
                upload_dir=upload_dir,
            )
            self._sessions[session_id] = session
            return session

    async def get_session(self, session_id: str) -> PlaygroundSession | None:
        """Get an active session by ID, refreshing its timeout."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_active > SESSION_TIMEOUT_SECONDS:
                self._destroy_session(session_id)
                return None
            session.last_active = time.time()
            return session

    async def end_session(self, session_id: str) -> None:
        """Explicitly end and clean up a session."""
        async with self._lock:
            self._destroy_session(session_id)

    def _cleanup_expired(self) -> None:
        """Remove sessions that have timed out."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > SESSION_TIMEOUT_SECONDS
        ]
        for sid in expired:
            self._destroy_session(sid)

    def _destroy_session(self, session_id: str) -> None:
        """Remove session and delete its uploaded files."""
        session = self._sessions.pop(session_id, None)
        if session and session.upload_dir.exists():
            shutil.rmtree(session.upload_dir, ignore_errors=True)

    async def get_status(self) -> dict:
        """Get current capacity status."""
        async with self._lock:
            self._cleanup_expired()
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": MAX_CONCURRENT_SESSIONS,
                "available_slots": MAX_CONCURRENT_SESSIONS - len(self._sessions),
            }


# Global singleton
session_manager = SessionManager()
