# Python Application Builder

A lightweight dependency injection framework for Python applications that provides service management, factory patterns, worker threading, and comprehensive logging capabilities.

## 🚀 Features

- **Dependency Injection**: Clean, testable code with automatic dependency resolution
- **Service Management**: Singleton services with automatic initialization
- **Factory Pattern**: On-demand creation of transient objects with dependency context access
- **Worker Threading**: Background task execution with built-in thread management
- **Comprehensive Logging**: Beautiful, structured logging with loguru integration
- **Configuration Management**: Environment variable integration and custom configuration
- **Type Safety**: Full type hints and generic support
- **Dependency Context**: Factory functions have access to the full DI container

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher

### Install Dependencies

```bash
pip install loguru
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

Here's a simple example to get you started:

```python
from application_builder import ApplicationBuilder, Dependency, Worker

# 1. Define a service
class GreetingService(Dependency):
    def say_hello(self, name: str):
        self.logger.info(f"Hello, {name}!")
        return f"Hello, {name}!"

# 2. Define a worker
class MainWorker(Worker):
    def run(self):
        greeting_service = self.get_service(GreetingService)
        greeting_service.say_hello("World")

# 3. Build and run the application
app_builder = ApplicationBuilder()
app_builder.add_service(GreetingService)
app_builder.add_worker(MainWorker)
app_builder.run()
```

## 🎯 Core Concepts

### Dependency Injection

The framework uses dependency injection to manage object creation and lifetime. Dependencies are automatically resolved and injected when needed.

### Services (Singletons)

Services are singleton objects that live for the entire application lifetime. They're created once during application startup and reused throughout the application.

```python
# Register a service
app_builder.add_service(PostgreSQLDatabaseService)

# Use the service in any dependency
db_service = self.get_service(BaseDatabaseService)  # Gets the PostgreSQL implementation
```

### Factories (On-Demand Creation)

Factories create new instances every time they're called. Factory functions now receive a `dependency_context` parameter that provides access to services, configurations, and other factories.

```python
# Register a factory
app_builder.add_factory(EmailMessage)

# Create new instances with full DI support
factory = self.get_factory(EmailMessage)
email1 = factory.create()  # New instance with logger and DI
email2 = factory.create()  # Another new instance with logger and DI
```

### Workers (Runtime Loops)

Workers run in separate threads and handle background tasks or main application logic. They have access to all registered services and factories.

```python
class BackgroundWorker(Worker):
    def run(self):
        db_service = self.get_service(BaseDatabaseService)
        while True:
            # Your background logic here
            db_service.process_pending_records()
            time.sleep(1)
```

## 📚 Usage Examples

### Example 1: Basic Service Usage

```python
from application_builder import ApplicationBuilder, Dependency, Worker

class ConfigService(Dependency):
    def initialize(self):
        self.app_name = self.get_configuration("APP_NAME", "MyApp")
        self.logger.info(f"Initialized {self.app_name}")

class AppWorker(Worker):
    def run(self):
        config = self.get_service(ConfigService)
        self.logger.info(f"Running {config.app_name}")

# Setup
app_builder = ApplicationBuilder()
app_builder.add_configuration("APP_NAME", "Python Builder")
app_builder.add_service(ConfigService)
app_builder.add_worker(AppWorker)
app_builder.run()
```

### Example 2: Interface-Based Services

```python
from application_builder import ApplicationBuilder, Dependency, Worker
from typing import Dict, Any

# Define interface
class BaseDatabaseService(Dependency):
    def save(self, table: str, data: Dict[str, Any]) -> int:
        raise NotImplementedError()

# Implementations
class PostgreSQLService(BaseDatabaseService):
    def save(self, table: str, data: Dict[str, Any]) -> int:
        self.logger.info(f"PostgreSQL save: {table}")
        return 123

class MongoDBService(BaseDatabaseService):
    def save(self, table: str, data: Dict[str, Any]) -> int:
        self.logger.info(f"MongoDB save: {table}")
        return 456

class DataWorker(Worker):
    def run(self):
        db = self.get_service(BaseDatabaseService)
        db.save("users", {"name": "Alice", "timestamp": "2025-06-18 19:05:13"})

# Setup - switch PostgreSQLService to MongoDBService as needed
app_builder = ApplicationBuilder()
app_builder.add_service(PostgreSQLService)
app_builder.add_worker(DataWorker)
app_builder.run()
```

### Example 3: Multiple Services with Keys

```python
from application_builder import ApplicationBuilder, Dependency, Worker

class CacheService(Dependency):
    def __init__(self):
        self.data = {}
    
    def set(self, key: str, value):
        self.data[key] = value
        self.logger.info(f"[{self._dependency_local_key}] Cached: {key}")

class CacheWorker(Worker):
    def run(self):
        user_cache = self.get_service(CacheService, "users")
        session_cache = self.get_service(CacheService, "sessions")
        
        user_cache.set("user:123", {"name": "Alice"})
        session_cache.set("session:abc", {"user_id": 123})

# Setup
app_builder = ApplicationBuilder()
app_builder.add_service(CacheService, "users")
app_builder.add_service(CacheService, "sessions")
app_builder.add_worker(CacheWorker)
app_builder.run()
```

### Example 4: Configuration-Driven Selection

```python
from application_builder import ApplicationBuilder, Dependency, Worker, DependencyCore

class BaseStorageService(Dependency):
    def store(self, filename: str) -> str:
        raise NotImplementedError()

class LocalStorageService(BaseStorageService):
    def store(self, filename: str) -> str:
        path = f"./storage/{filename}"
        self.logger.info(f"Stored locally: {path}")
        return path

class CloudStorageService(BaseStorageService):
    def store(self, filename: str) -> str:
        url = f"https://cloud.com/{filename}"
        self.logger.info(f"Stored in cloud: {url}")
        return url

class StorageWorker(Worker):
    def run(self):
        factory = self.get_factory(BaseStorageService)
        storage = factory.create()
        storage.store("document.txt")

def create_storage(dependency_context: DependencyCore):
    # Environment variables are automatically available via get_configuration
    storage_type = dependency_context.get_configuration("STORAGE_TYPE", "local")
    if storage_type == "local":
        return LocalStorageService()
    else:
        return CloudStorageService()

app_builder = ApplicationBuilder()
app_builder.add_factory(BaseStorageService, custom_factory=create_storage)
app_builder.add_worker(StorageWorker)

# Run with: STORAGE_TYPE=cloud python main.py
app_builder.run()
```

## 📖 API Reference

### ApplicationBuilder

The main builder class for configuring the application.

#### Constructor

```python
ApplicationBuilder(log_dir: str = os.path.join(os.getcwd(), "logs"))
```

#### Methods

- `add_service(service_type, local_key=None, name=None)`: Register a singleton service
- `add_factory(factory_type, local_key=None, name=None, custom_factory=None)`: Register a factory
- `add_worker(worker_type, name=None)`: Register a worker
- `add_configuration(key, value)`: Add a configuration value
- `run()`: Start the application

### DependencyCore

Base class that provides dependency injection functionality.

#### Methods

- `get_service(service_type, local_key=None)`: Get a service instance
- `get_services(service_type)`: Get all services of a type
- `get_factory(factory_type, local_key=None)`: Get a factory instance
- `get_factories(factory_type)`: Get all factories of a type
- `get_configuration(key, default=None)`: Get a configuration value

### Dependency

Base class for all injectable components (inherits from DependencyCore).

#### Methods

- `initialize()`: Override for custom initialization logic

#### Properties

- `logger`: Loguru logger instance bound to the dependency context
- `_dependency_name`: The name of the dependency
- `_dependency_local_key`: The local key used for registration
- `_dependency_keys`: List of all type keys this dependency is registered under

### Worker

Base class for worker components that run in separate threads.

#### Methods

- `run()`: Override with your worker logic (required)

### DependencyFactory

Factory class for creating dependency instances with full DI support.

#### Methods

- `create()`: Create a new instance of the dependency with proper DI wiring

### Custom Factory Functions

Custom factory functions now receive a `dependency_context` parameter:

```python
def my_custom_factory(dependency_context: DependencyCore) -> MyService:
    # Access services, configurations, and factories
    config_value = dependency_context.get_configuration("MY_CONFIG")
    other_service = dependency_context.get_service(OtherService)
    
    service = MyService()
    service.config = config_value
    return service
```

## 🏗️ Project Structure

```
python_application_builder/
├── application_builder.py          # Core framework classes
├── main.py                         # Application entry point
├── requirements.txt                # Dependencies
├── workers/
│   └── main_worker.py              # Main application worker
├── application/                    # Application layer (interfaces/contracts)
│   ├── database/
│   │   └── services/
│   │       └── base_database_service.py
│   └── storage/
│       └── services/
│           └── base_storage_service.py
└── infrastructure/                 # Infrastructure layer (implementations)
    ├── database/
    │   └── services/
    │       ├── postgresql_database_service.py
    │       └── mongodb_database_service.py
    └── storage/
        └── services/
            ├── local_storage_service.py
            └── cloud_storage_service.py
```

### Architecture Benefits

1. **Testability**: Easy to mock services and factories for unit tests
2. **Flexibility**: Switch implementations without changing business logic
3. **Scalability**: Add new implementations by extending base interfaces
4. **Separation of Concerns**: Business logic depends on abstractions, not concrete implementations
5. **Factory Context**: Custom factories can access the full DI container

## 🔍 Advanced Features

### Dependency Context in Factories

Factory functions receive a `DependencyCore` parameter that provides access to:
- Configuration values via `get_configuration()`
- Registered services via `get_service()` and `get_services()`
- Other factories via `get_factory()` and `get_factories()`

### Automatic Base Class Registration

Dependencies are automatically registered for all their base classes:

```python
# Register concrete implementation
app_builder.add_service(PostgreSQLDatabaseService)

# Retrieve by interface
db_service = self.get_service(BaseDatabaseService)  # Gets PostgreSQL implementation
```

### Environment Variable Integration

All environment variables are automatically loaded as configurations when the application starts. You can access them directly via `get_configuration()` without manually adding them:

```python
class DatabaseService(Dependency):
    def initialize(self):
        # Environment variables are automatically available
        self.host = self.get_configuration("DB_HOST", "localhost")
        self.port = self.get_configuration("DB_PORT", "5432")
        self.user = self.get_configuration("DB_USER", "admin")
```

### Thread-Safe Logging with Context

Each dependency gets its own logger with colored output and automatic file rotation.

## 🧪 Testing

Services and factories can be easily mocked for testing:

```python
from application_builder import DependencyCore

class MockStorageService(BaseStorageService):
    def store(self, filename: str) -> str:
        return f"mock://{filename}"

def mock_storage_factory(dependency_context: DependencyCore):
    return MockStorageService()

# Use in tests
app_builder = ApplicationBuilder()
app_builder.add_factory(BaseStorageService, custom_factory=mock_storage_factory)
```

## 🚀 Performance Considerations

- **Services**: Initialized once at startup - use for expensive resources
- **Factories**: Create instances on-demand with dependency context access
- **Workers**: Run in separate threads - design for thread safety
- **Logging**: Asynchronous with automatic rotation - minimal performance impact
- **Dependency Context**: Provides efficient access to DI container in factory functions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/Kiryuumaru/python_application_builder.git
cd python_application_builder
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [loguru](https://github.com/Delgan/loguru) for excellent logging capabilities
- Inspired by dependency injection frameworks from other languages
- Thanks to the Python community for continuous inspiration

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Kiryuumaru/python_application_builder/issues) page
2. Create a new issue with a detailed description
3. Provide code examples and error messages when possible

---

**Happy coding! 🐍✨**