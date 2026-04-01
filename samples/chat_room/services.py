import time
import threading
from typing import Dict, List, Set
from application_builder import ILogger
from interfaces import IMessageFormatter, ISessionContext, IRoomRegistry


class TimestampFormatter(IMessageFormatter):
    def format(self, user: str, message: str) -> str:
        ts = time.strftime("%H:%M:%S")
        return f"[{ts}] {message}"


class UsernameFormatter(IMessageFormatter):
    def format(self, user: str, message: str) -> str:
        return f"<{user}> {message}"


class SessionContext(ISessionContext):
    """Scoped — each DI scope gets its own instance."""

    def __init__(self):
        self._user = ""
        self._room = ""

    def set_user(self, user: str) -> None:
        self._user = user

    def set_room(self, room: str) -> None:
        self._room = room

    def get_user(self) -> str:
        return self._user

    def get_room(self) -> str:
        return self._room


class InMemoryRoomRegistry(IRoomRegistry):
    def __init__(self, logger: ILogger):
        self._rooms: Dict[str, List[str]] = {}
        self._users: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
        self._logger = logger

    def join(self, room: str, user: str) -> None:
        with self._lock:
            if room not in self._rooms:
                self._rooms[room] = []
                self._users[room] = set()
                self._logger.info(f"Room #{room} created")
            self._users[room].add(user)

    def leave(self, room: str, user: str) -> None:
        with self._lock:
            users = self._users.get(room)
            if users:
                users.discard(user)

    def post_message(self, room: str, message: str) -> None:
        with self._lock:
            if room in self._rooms:
                self._rooms[room].append(message)

    def get_history(self, room: str) -> List[str]:
        with self._lock:
            return list(self._rooms.get(room, []))

    def list_rooms(self) -> List[str]:
        with self._lock:
            return list(self._rooms.keys())
