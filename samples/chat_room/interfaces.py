from abc import ABC, abstractmethod
from typing import List


class IMessageFormatter(ABC):
    @abstractmethod
    def format(self, user: str, message: str) -> str:
        pass


class ISessionContext(ABC):
    @abstractmethod
    def set_user(self, user: str) -> None:
        pass

    @abstractmethod
    def set_room(self, room: str) -> None:
        pass

    @abstractmethod
    def get_user(self) -> str:
        pass

    @abstractmethod
    def get_room(self) -> str:
        pass


class IRoomRegistry(ABC):
    @abstractmethod
    def join(self, room: str, user: str) -> None:
        pass

    @abstractmethod
    def leave(self, room: str, user: str) -> None:
        pass

    @abstractmethod
    def post_message(self, room: str, message: str) -> None:
        pass

    @abstractmethod
    def get_history(self, room: str) -> List[str]:
        pass

    @abstractmethod
    def list_rooms(self) -> List[str]:
        pass
