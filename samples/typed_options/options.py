from dataclasses import dataclass


@dataclass
class DatabaseOptions:
    host: str = "localhost"
    port: int = 5432
    name: str = "mydb"
    max_connections: int = 5


@dataclass
class CacheOptions:
    enabled: bool = True
    ttl_seconds: int = 300
    max_size: int = 1000
