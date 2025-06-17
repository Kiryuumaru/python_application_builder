# Python Application Builder

A lightweight dependency injection framework for Python applications that provides service management, factory patterns, worker threading, and comprehensive logging capabilities.

## Features

- **Dependency Injection**: Automatic dependency resolution and injection
- **Service Management**: Register and manage singleton services
- **Factory Pattern**: Create instances on-demand using factory services
- **Worker Threading**: Multi-threaded worker execution
- **Configuration Management**: Environment variable integration and custom configurations
- **Structured Logging**: Built-in logging with Loguru integration
- **Type Safety**: Full type annotations and generic type support

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kiryuumaru/python_application_builder.git
cd python_application_builder
pip install -r requirements.txt
```

## Quick Start

Here's a basic example of how to use the Application Builder:

```python
from application.application_builder import ApplicationBuilder
from application.workers.main_worker import MainWorker
from infrastructure.animal_samples.services.cat_service import CatService
from infrastructure.animal_samples.services.dog_service import DogService
from infrastructure.petfood.services.pet_food_service import PetFoodService

# Create application builder instance
app_builder = ApplicationBuilder()

# Register services (singletons)
app_builder.add_service(CatService)
app_builder.add_service(DogService)

# Register factories (for on-demand instance creation)
app_builder.add_factory(PetFoodService)

# Register workers (background tasks)
app_builder.add_worker(MainWorker)

# Run the application
app_builder.run()
```

## Core Concepts

### Services

Services are singleton instances that are initialized once and reused throughout the application:

```python
from application.application_builder import Dependency

class MyService(Dependency):
    def initialize(self):
        self.logger.info("Service initialized")
        # Initialization logic here
    
    def do_something(self):
        self.logger.info("Doing something...")
```

### Factories

Factories create new instances on-demand, useful for stateful objects:

```python
class MyFactoryService(Dependency):
    def __init__(self):
        self.instance_data = None
    
    def initialize(self):
        self.logger.info("Factory instance created")
```

### Workers

Workers run in separate threads and handle background tasks:

```python
from application.application_builder import Worker

class MyWorker(Worker):
    def run(self):
        # Get services
        my_service = self.get_service(MyService)
        
        # Worker logic here
        while True:
            my_service.do_something()
            time.sleep(1)
```

## Dependency Injection

The framework automatically resolves dependencies based on type annotations:

```python
class ServiceA(Dependency):
    def do_work(self):
        return "Work from A"

class ServiceB(Dependency):
    def initialize(self):
        # Get other services
        service_a = self.get_service(ServiceA)
        
        # Get factory instances
        factory = self.get_factory(SomeFactoryService)
        instance = factory.create()
        
        # Get configuration
        config_value = self.get_configuration("MY_CONFIG", "default_value")
```

## Configuration Management

The framework integrates with environment variables and supports custom configurations:

```python
# Add custom configuration
app_builder.add_configuration("custom_key", "custom_value")

# Access in services
class MyService(Dependency):
    def initialize(self):
        # Gets from environment variables or custom configs
        value = self.get_configuration("DATABASE_URL", "sqlite://memory")
```

## Logging

Built-in structured logging with context awareness:

```python
class MyService(Dependency):
    def do_something(self):
        self.logger.info("This will be logged with service context")
        self.logger.error("Error occurred", extra={"data": "additional_info"})
```

Log files are automatically rotated daily and retained for 30 days in the `logs/` directory.

## Project Structure

```
python_application_builder/
├── application/
│   ├── application_builder.py    # Core framework
│   ├── services/                 # Base service definitions
│   └── workers/                  # Worker implementations
├── infrastructure/               # Concrete service implementations
│   ├── animal_samples/
│   └── petfood/
├── main.py                       # Application entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Advanced Usage

### Custom Service Names

```python
app_builder.add_service(MyService, name="CustomServiceName")
```

### Multiple Service Implementations

```python
# Register multiple implementations of the same interface
app_builder.add_service(DatabaseService, name="PostgreSQL")
app_builder.add_service(DatabaseService, name="MySQL")

# Get all implementations
services = self.get_services(DatabaseService)
```

### Environment Integration

The framework automatically loads environment variables into the configuration system:

```bash
export DATABASE_URL=postgresql://localhost/mydb
export LOG_LEVEL=DEBUG
```

## Requirements

- Python 3.7+
- loguru 0.7.3+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.

## Examples

See the included example services in the `infrastructure/` directory for practical implementations of the animal and pet food services that demonstrate the framework's capabilities.
