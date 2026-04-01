from abc import ABC, abstractmethod


class IStoragePlugin(ABC):
    @abstractmethod
    def store(self, name: str, data: bytes) -> str:
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        pass


class ICompressionPlugin(ABC):
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def get_extension(self) -> str:
        pass


class INotificationPlugin(ABC):
    @abstractmethod
    def notify(self, subject: str, body: str) -> None:
        pass
