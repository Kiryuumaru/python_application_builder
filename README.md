# Python Application Builder

A lightweight dependency injection framework for Python applications that provides service management, configuration handling, worker threading, and structured logging capabilities. Inspired by .NET's host builder pattern.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🚀 Features

- **💉 Dependency Injection**: Complete IoC container with automatic dependency resolution
- **⚙️ Service Lifetimes**: Singleton, Scoped, and Transient service management
- **🏗️ Configuration System**: Multi-source configuration (environment, JSON, in-memory)
- **🧵 Worker Threading**: Background services with lifecycle management
- **📊 Structured Logging**: Contextual logging with loguru integration
- **🔧 Service Scoping**: Isolated service scopes for request-level dependencies
- **🛡️ Type Safety**: Full type hints and generic support

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Concepts](#core-concepts)
- [Configuration System](#configuration-system)
- [Dependency Injection](#dependency-injection)
- [Worker System](#worker-system)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)

## 🔧 Installation

```bash
pip install loguru
```

## 🏃 Quick Start

```python
from python_application_builder import (
    ApplicationBuilder, IConfiguration, ILogger, Worker
)

class GreetingService:
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.config = config
        self.logger = logger
        self.app_name = config.get("App:Name", "Demo App")
    
    def greet(self, name: str) -> str:
        message = f"Hello, {name}! Welcome to {self.app_name}"
        self.logger.info(message)
        return message

class GreetingWorker(Worker):
    def __init__(self, greeting_service: GreetingService, logger: ILogger):
        super().__init__()
        self.greeting_service = greeting_service
        self.logger = logger
    
    def execute(self):
        self.greeting_service.greet("World")

def main():
    app = ApplicationBuilder()
    app.add_configuration_dictionary({"App": {"Name": "My App"}})
    app.add_singleton(GreetingService)
    app.add_worker(GreetingWorker)
    app.run()

if __name__ == "__main__":
    main()
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │    Workers      │  │    Services     │  │ Controllers  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Dependency Injection Core                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Service Provider│  │ Service Scopes  │  │   Service    │ │
│  │   (Container)   │  │   (Isolation)   │  │  Lifetimes   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Core Concepts

### Service Lifetimes

```python
# Singleton - One instance for entire application
app.add_singleton(DatabaseService)

# Scoped - One instance per scope
app.add_scoped(RequestContextService)

# Transient - New instance every time
app.add_transient(EmailMessage)
```

### Service Registration

```python
# By type
app.add_singleton(EmailService)

# By interface and implementation
app.add_singleton(IEmailService, EmailService)

# By factory
app.add_singleton_factory(IEmailService, 
    lambda provider: EmailService(provider.get_required_service(IConfiguration)))

# By instance
app.add_singleton_instance(IEmailService, EmailService())
```

## ⚙️ Configuration System

### Multi-Source Configuration

```python
app = ApplicationBuilder()

# Environment variables
app.add_configuration(lambda builder: 
    builder.add_environment_variables("MYAPP_"))

# JSON file
app.add_configuration(lambda builder: 
    builder.add_json_file("appsettings.json"))

# In-memory (highest priority)
app.add_configuration_dictionary({
    "Database": {"ConnectionString": "Server=localhost"},
    "Logging": {"Level": "INFO", "File": "logs/app.log"}
})
```

### Configuration Usage

```python
class DatabaseService:
    def __init__(self, config: IConfiguration, logger: ILogger):
        # Hierarchical access
        db_config = config.get_section("Database")
        self.connection_string = db_config.get("ConnectionString")
        
        # Type conversion
        self.timeout = config.get_int("Database:Timeout", 30)
        self.debug = config.get_bool("Debug:Enabled", False)
        
        # Lists and dictionaries
        self.hosts = config.get_list("Security:AllowedHosts", [])
        self.features = config.get_dict("Features", {})
```

## 💉 Dependency Injection

### Automatic Constructor Injection

```python
class OrderService:
    def __init__(self, 
                 database: IDatabaseService,
                 logger: ILogger,
                 email_service: IEmailService):
        self.database = database
        self.logger = logger
        self.email_service = email_service
    
    def process_order(self, order_data: dict):
        self.logger.info(f"Processing order: {order_data['id']}")
        self.database.save_order(order_data)
        self.email_service.send_confirmation(order_data['email'])
```

### Interface-Based Development

```python
class IRepository(ABC):
    @abstractmethod
    def save(self, entity: dict) -> int: pass

class SqlRepository(IRepository):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.config = config
        self.logger = logger
    
    def save(self, entity: dict) -> int:
        self.logger.info("Saving to SQL Server")
        return 1

# Register based on configuration
database_type = os.getenv("DATABASE_TYPE", "sql")
if database_type == "mongo":
    app.add_singleton(IRepository, MongoRepository)
else:
    app.add_singleton(IRepository, SqlRepository)
```

## 🧵 Worker System

### Background Workers

```python
class DataProcessingWorker(Worker):
    def __init__(self, data_service: IDataService, logger: ILogger):
        super().__init__()
        self.data_service = data_service
        self.logger = logger
    
    def execute(self):
        self.logger.info("Worker started")
        
        while not self.is_stopping():
            try:
                batch = self.data_service.get_pending_batch(10)
                if not batch:
                    self.wait_for_stop(5.0)
                    continue
                
                for item in batch:
                    if self.is_stopping():
                        break
                    self.data_service.process_item(item)
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                self.wait_for_stop(10.0)
```

### Timed Workers

```python
class HealthCheckWorker(TimedWorker):
    def __init__(self, health_service: IHealthService, logger: ILogger):
        super().__init__(interval_seconds=30)  # Every 30 seconds
        self.health_service = health_service
        self.logger = logger
    
    def do_work(self):
        status = self.health_service.check_all_systems()
        if status.is_healthy:
            self.logger.info("All systems healthy")
        else:
            self.logger.warning(f"Issues: {status.issues}")
```

## 📚 Usage Examples

### Example 1: Web API Service

```python
from abc import ABC, abstractmethod

# Interfaces
class IUserRepository(ABC):
    @abstractmethod
    def create_user(self, user_data: dict) -> int: pass

class IEmailService(ABC):
    @abstractmethod
    def send_welcome_email(self, email: str, name: str): pass

# Business Logic
class UserService:
    def __init__(self, user_repo: IUserRepository, email_service: IEmailService, logger: ILogger):
        self.user_repo = user_repo
        self.email_service = email_service
        self.logger = logger
    
    def register_user(self, user_data: dict) -> dict:
        user_id = self.user_repo.create_user(user_data)
        self.email_service.send_welcome_email(user_data['email'], user_data['name'])
        return {"id": user_id, "status": "registered"}

# Infrastructure
class SqlUserRepository(IUserRepository):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.connection_string = config.get("Database:ConnectionString")
        self.logger = logger
    
    def create_user(self, user_data: dict) -> int:
        self.logger.info(f"Saving user: {user_data['email']}")
        return 123

class SmtpEmailService(IEmailService):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.smtp_host = config.get("Email:SmtpHost")
        self.logger = logger
    
    def send_welcome_email(self, email: str, name: str):
        self.logger.info(f"Sending welcome email to: {email}")

# Setup
def create_app():
    app = ApplicationBuilder()
    app.add_configuration_dictionary({
        "Database": {"ConnectionString": "Server=localhost"},
        "Email": {"SmtpHost": "smtp.gmail.com"}
    })
    app.add_singleton(IUserRepository, SqlUserRepository)
    app.add_singleton(IEmailService, SmtpEmailService)
    app.add_singleton(UserService)
    return app.build()

# Usage in web framework
service_provider = create_app()

def register_endpoint(user_data: dict):
    scope_factory = service_provider.get_required_service(ScopeFactory)
    with scope_factory.create_scope_context() as scope:
        user_service = scope.get_required_service(UserService)
        return user_service.register_user(user_data)
```

### Example 2: Data Pipeline

```python
# Pipeline Components
class IDataSource(ABC):
    @abstractmethod
    def get_next_batch(self, size: int) -> list: pass

class IDataProcessor(ABC):
    @abstractmethod
    def process(self, data: dict) -> dict: pass

class FileDataSource(IDataSource):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.file_path = config.get("Pipeline:SourceFile")
        self.logger = logger
        self.position = 0
    
    def get_next_batch(self, size: int) -> list:
        batch = [{"id": i, "data": f"record_{i}"} for i in range(self.position, self.position + size)]
        self.position += size
        return batch if self.position < 1000 else []

class DataTransformProcessor(IDataProcessor):
    def process(self, data: dict) -> dict:
        return {
            "id": data["id"],
            "processed_data": data["data"].upper(),
            "timestamp": time.time()
        }

# Pipeline Worker
class PipelineWorker(Worker):
    def __init__(self, source: IDataSource, processor: IDataProcessor, logger: ILogger):
        super().__init__()
        self.source = source
        self.processor = processor
        self.logger = logger
    
    def execute(self):
        while not self.is_stopping():
            batch = self.source.get_next_batch(10)
            if not batch:
                break
            
            for item in batch:
                processed = self.processor.process(item)
                self.logger.debug(f"Processed: {processed['id']}")

# Setup
def main():
    app = ApplicationBuilder()
    app.add_configuration_dictionary({"Pipeline": {"SourceFile": "data.json"}})
    app.add_singleton(IDataSource, FileDataSource)
    app.add_singleton(IDataProcessor, DataTransformProcessor)
    app.add_worker(PipelineWorker)
    app.run()
```

### Example 3: Health Monitoring

```python
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

class HealthCheckResult:
    def __init__(self, name: str, status: HealthStatus, message: str = ""):
        self.name = name
        self.status = status
        self.message = message

class DatabaseHealthCheck:
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.connection_string = config.get("Database:ConnectionString")
        self.logger = logger
    
    def check_health(self) -> HealthCheckResult:
        try:
            # Simulate database check
            return HealthCheckResult("database", HealthStatus.HEALTHY, "OK")
        except Exception as e:
            return HealthCheckResult("database", HealthStatus.UNHEALTHY, str(e))

class HealthMonitorWorker(TimedWorker):
    def __init__(self, db_check: DatabaseHealthCheck, logger: ILogger):
        super().__init__(interval_seconds=15)
        self.db_check = db_check
        self.logger = logger
    
    def do_work(self):
        result = self.db_check.check_health()
        if result.status == HealthStatus.HEALTHY:
            self.logger.info(f"{result.name}: {result.message}")
        else:
            self.logger.error(f"{result.name}: {result.message}")

# Setup
def main():
    app = ApplicationBuilder()
    app.add_singleton(DatabaseHealthCheck)
    app.add_worker(HealthMonitorWorker)
    app.run()
```

## 📖 API Reference

### ApplicationBuilder

```python
class ApplicationBuilder:
    # Service registration
    def add_singleton(self, service_type: Type[T], impl_type: Type = None) -> 'ApplicationBuilder'
    def add_scoped(self, service_type: Type[T], impl_type: Type = None) -> 'ApplicationBuilder'
    def add_transient(self, service_type: Type[T], impl_type: Type = None) -> 'ApplicationBuilder'
    def add_singleton_factory(self, service_type: Type[T], factory: Callable) -> 'ApplicationBuilder'
    def add_singleton_instance(self, service_type: Type[T], instance: T) -> 'ApplicationBuilder'
    
    # Worker registration
    def add_worker(self, worker_type: Type[IWorker]) -> 'ApplicationBuilder'
    
    # Configuration
    def add_configuration_dictionary(self, config: Dict[str, Any]) -> 'ApplicationBuilder'
    def add_configuration(self, configure: Callable[[ConfigurationBuilder], None]) -> 'ApplicationBuilder'
    
    # Build and run
    def build(self, auto_start_workers: bool = True) -> 'ServiceProvider'
    def run(self) -> None
```

### ServiceProvider

```python
class ServiceProvider:
    def get_service(self, service_type: Type[T]) -> Optional[T]
    def get_required_service(self, service_type: Type[T]) -> T
    def get_services(self, service_type: Type[T]) -> List[T]
    def create_scope(self) -> 'ServiceProvider'
```

### Configuration Interfaces

```python
class IConfiguration:
    def get(self, key: str, default: Any = None) -> Any
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]
    def get_list(self, key: str, default: Optional[List] = None) -> Optional[List]
    def get_dict(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]
    def get_section(self, key: str) -> 'IConfigurationSection'
```

### Worker Base Classes

```python
class Worker:
    def start(self) -> None
    def stop(self) -> None
    def is_stopping(self) -> bool
    def wait_for_stop(self, timeout: float = None) -> bool
    
    @abstractmethod
    def execute(self) -> None

class TimedWorker(Worker):
    def __init__(self, interval_seconds: float = 5)
    
    @abstractmethod
    def do_work(self) -> None
```

## 🧪 Testing

### Unit Testing

```python
import unittest
from unittest.mock import Mock

class TestUserService(unittest.TestCase):
    def setUp(self):
        self.mock_repo = Mock()
        self.mock_email = Mock()
        self.mock_logger = Mock()
        
        self.user_service = UserService(
            self.mock_repo, self.mock_email, self.mock_logger
        )
    
    def test_register_user(self):
        # Arrange
        self.mock_repo.create_user.return_value = 123
        user_data = {"name": "John", "email": "john@test.com"}
        
        # Act
        result = self.user_service.register_user(user_data)
        
        # Assert
        self.assertEqual(result["id"], 123)
        self.mock_repo.create_user.assert_called_once()
        self.mock_email.send_welcome_email.assert_called_once()
```

### Integration Testing

```python
class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.app = ApplicationBuilder()
        self.app.add_configuration_dictionary({"Database": {"ConnectionString": "test"}})
        self.app.add_singleton(IUserRepository, InMemoryUserRepository)
        self.app.add_singleton(UserService)
        self.provider = self.app.build()
    
    def test_user_service_integration(self):
        user_service = self.provider.get_required_service(UserService)
        result = user_service.register_user({"name": "Jane", "email": "jane@test.com"})
        self.assertIsNotNone(result["id"])
```

## 🚀 Advanced Features

### Service Scoping

```python
scope_factory = service_provider.get_required_service(ScopeFactory)
with scope_factory.create_scope_context() as scope:
    scoped_service = scope.get_required_service(IScopedService)
    # All scoped services share instances within this context
```

### Factory Pattern

```python
def create_storage_service(provider: ServiceProvider) -> IStorageService:
    config = provider.get_required_service(IConfiguration)
    storage_type = config.get("Storage:Type", "local")
    
    if storage_type == "cloud":
        return CloudStorageService(config)
    return LocalStorageService(config)

app.add_singleton_factory(IStorageService, create_storage_service)
```

### Configuration Reload

```python
class ConfigurableService:
    def __init__(self, config: IConfiguration):
        self.config = config
        self.update_settings()
    
    def update_settings(self):
        self.timeout = self.config.get_int("Service:Timeout", 5000)
    
    def reload_config(self):
        self.config.reload()
        self.update_settings()
```

## ✅ Best Practices

```python
# ✅ Good: Interface-based design
class IEmailService(ABC):
    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> bool: pass

# ✅ Good: Proper error handling in workers
class RobustWorker(Worker):
    def execute(self):
        while not self.is_stopping():
            try:
                self.do_work()
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                self.wait_for_stop(5.0)

# ✅ Good: Hierarchical configuration
{"Database": {"Primary": {"ConnectionString": "..."}}}

# ❌ Bad: Flat configuration
{"DatabasePrimaryConnectionString": "..."}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/Kiryuumaru/python_application_builder.git
cd python_application_builder
pip install -r requirements.txt
python python_application_builder/main.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[loguru](https://github.com/Delgan/loguru)** - Excellent logging
- **.NET Core** - DI and host builder pattern inspiration
- **Python Community** - Continuous inspiration

---

**Built with ❤️ for the Python community! 🐍✨**