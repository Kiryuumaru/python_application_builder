import inspect
import json
from multiprocessing import context
import os
import signal
import sys
import threading
import time
import loguru
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, Tuple, get_type_hints

# Type variable for generic methods
T = TypeVar('T')

def create_loguru_logger(context, log_level, log_file):
    log_format = '<fg 95,95,95>{time}</> <level>{level: <8}</> - [<fg 95,95,95>{extra[context]}</>] <level>{message}</>'
    logger = loguru.logger.bind(context=context)
    logger.level("TRACE", color="<fg #444444>")
    logger.level("DEBUG", color="<fg #666666>")
    logger.level("INFO", color="<fg #FFFFFF>")
    logger.level("SUCCESS", color="<fg #00CC99>")
    logger.level("WARNING", color="<fg #FFBB00>")
    logger.level("ERROR", color="<fg #FF4444>")
    logger.level("CRITICAL", color="<fg #FF00FF>")
    logger.remove()
    logger.add(sys.stdout, colorize=True, format=log_format, level=log_level)
    if log_file:
        logger.add(log_file, rotation="10 MB", level=log_level, format=log_format)
    return logger

logger = create_loguru_logger("Default", "TRACE", None)

###########################################
# Configuration System
###########################################

class IConfigurationSection(ABC):
    """Interface for a section of configuration values."""
    
    @property
    @abstractmethod
    def key(self) -> str:
        """Gets the key of this configuration section."""
        pass
    
    @property
    @abstractmethod
    def path(self) -> str:
        """Gets the full path to this configuration section."""
        pass
    
    @property
    @abstractmethod
    def value(self) -> Optional[str]:
        """Gets the value of this configuration section."""
        pass
    
    @abstractmethod
    def get_section(self, key: str) -> 'IConfigurationSection':
        """Gets a configuration sub-section with the specified key."""
        pass
    
    @abstractmethod
    def get_children(self) -> List['IConfigurationSection']:
        """Gets the immediate children sub-sections."""
        pass
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value with the specified key."""
        pass
    
    @abstractmethod
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Gets a configuration value as an integer."""
        pass
    
    @abstractmethod
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Gets a configuration value as a float."""
        pass
    
    @abstractmethod
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Gets a configuration value as a boolean."""
        pass
    
    @abstractmethod
    def get_dict(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Gets a configuration value as a dictionary."""
        pass
    
    @abstractmethod
    def get_list(self, key: str, default: Optional[List] = None) -> Optional[List]:
        """Gets a configuration value as a list."""
        pass


class IConfiguration(IConfigurationSection):
    """Interface for application configuration."""
    
    @abstractmethod
    def reload(self) -> None:
        """Reloads configuration from all providers."""
        pass


class ConfigurationProvider(ABC):
    """Base class for configuration providers."""
    
    @abstractmethod
    def load(self) -> Dict[str, str]:
        """Loads configuration key-value pairs from the source."""
        pass


class EnvironmentVariablesConfigurationProvider(ConfigurationProvider):
    """Configuration provider that reads from environment variables."""
    
    def __init__(self, prefix: str = None):
        self.prefix = prefix
    
    def load(self) -> Dict[str, str]:
        """Loads configuration from environment variables."""
        result = {}
        for key, value in os.environ.items():
            if self.prefix is None or key.startswith(self.prefix):
                result_key = key[len(self.prefix):] if self.prefix and key.startswith(self.prefix) else key
                # Convert environment variable naming conventions to configuration style
                # Example: APP_SETTING_NAME becomes AppSetting:Name
                if "__" in result_key:
                    result_key = result_key.replace("__", ":")
                elif "_" in result_key:
                    parts = result_key.split("_")
                    result_key = ":".join(parts)
                
                result[result_key] = value
        return result


class JsonFileConfigurationProvider(ConfigurationProvider):
    """Configuration provider that reads from a JSON file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> Dict[str, str]:
        """Loads configuration from a JSON file."""
        if not os.path.exists(self.file_path):
            return {}
        
        try:
            with open(self.file_path, 'r') as file:
                json_data = json.load(file)
            
            # Flatten the JSON into key-value pairs
            return self._flatten_json(json_data)
        except Exception as e:
            logger.error(f"Error loading configuration from {self.file_path}: {e}")
            return {}
    
    def _flatten_json(self, json_data: Dict, prefix: str = "") -> Dict[str, str]:
        """Flattens a nested JSON object into key-value pairs."""
        result = {}
        for key, value in json_data.items():
            new_key = f"{prefix}:{key}" if prefix else key
            
            if isinstance(value, dict):
                nested = self._flatten_json(value, new_key)
                result.update(nested)
            elif isinstance(value, (list, tuple)):
                # Convert lists to JSON strings
                result[new_key] = json.dumps(value)
            else:
                result[new_key] = str(value)
        
        return result


class MemoryConfigurationProvider(ConfigurationProvider):
    """Configuration provider that reads from an in-memory dictionary."""
    
    def __init__(self, initial_data: Dict[str, str] = None):
        self.data = initial_data or {}
    
    def load(self) -> Dict[str, str]:
        """Loads configuration from the in-memory dictionary."""
        return self.data.copy()
    
    def set(self, key: str, value: str) -> None:
        """Sets a configuration value."""
        self.data[key] = value


class ConfigurationBuilder:
    """Builder for creating Configuration instances."""
    
    def __init__(self):
        self._providers: List[ConfigurationProvider] = []
    
    def add_provider(self, provider: ConfigurationProvider) -> 'ConfigurationBuilder':
        """Adds a configuration provider."""
        self._providers.append(provider)
        return self
    
    def add_environment_variables(self, prefix: str = None) -> 'ConfigurationBuilder':
        """Adds a configuration provider that reads from environment variables."""
        return self.add_provider(EnvironmentVariablesConfigurationProvider(prefix))
    
    def add_json_file(self, file_path: str) -> 'ConfigurationBuilder':
        """Adds a configuration provider that reads from a JSON file."""
        return self.add_provider(JsonFileConfigurationProvider(file_path))
    
    def add_in_memory_collection(self, initial_data: Dict[str, str] = None) -> 'ConfigurationBuilder':
        """Adds a configuration provider that reads from an in-memory dictionary."""
        return self.add_provider(MemoryConfigurationProvider(initial_data))
    
    def build(self) -> 'Configuration':
        """Builds a Configuration instance with the registered providers."""
        return Configuration(self._providers)


class ConfigurationSection(IConfigurationSection):
    """Implementation of a configuration section."""
    
    def __init__(self, 
                 configuration: 'Configuration', 
                 path: str, 
                 key: str = None):
        self._configuration = configuration
        self._path = path
        self._key = key or path
    
    @property
    def key(self) -> str:
        """Gets the key of this configuration section."""
        return self._key
    
    @property
    def path(self) -> str:
        """Gets the full path to this configuration section."""
        return self._path
    
    @property
    def value(self) -> Optional[str]:
        """Gets the value of this configuration section."""
        return self._configuration.get(self._path)
    
    def get_section(self, key: str) -> IConfigurationSection:
        """Gets a configuration sub-section with the specified key."""
        if not key:
            return self
        
        new_path = f"{self._path}:{key}" if self._path else key
        return ConfigurationSection(self._configuration, new_path, key)
    
    def get_children(self) -> List[IConfigurationSection]:
        """Gets the immediate children sub-sections."""
        children = []
        prefix = f"{self._path}:" if self._path else ""
        prefix_len = len(prefix)
        
        # Get all keys that start with the path prefix
        for full_key in self._configuration._data.keys():
            if full_key.startswith(prefix):
                # Extract the part after the prefix
                remaining = full_key[prefix_len:]
                # Get the first segment (up to the next ':')
                segment = remaining.split(':', 1)[0] if ':' in remaining else remaining
                
                # Create a section for this segment if we haven't already
                if segment and not any(section.key == segment for section in children):
                    section_path = f"{self._path}:{segment}" if self._path else segment
                    children.append(ConfigurationSection(self._configuration, section_path, segment))
        
        return children
    
    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value with the specified key."""
        if not key:
            return self.value or default
        
        path = f"{self._path}:{key}" if self._path else key
        return self._configuration.get(path, default)
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Gets a configuration value as an integer."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Gets a configuration value as a float."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Gets a configuration value as a boolean."""
        value = self.get(key)
        if value is None:
            return default
        
        true_values = ('true', 'yes', '1', 'on')
        false_values = ('false', 'no', '0', 'off')
        
        if value.lower() in true_values:
            return True
        elif value.lower() in false_values:
            return False
        return default
    
    def get_dict(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Gets a configuration value as a dictionary."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, default: Optional[List] = None) -> Optional[List]:
        """Gets a configuration value as a list."""
        value = self.get(key)
        if value is None:
            return default
        
        try:
            # Try to parse as JSON
            return json.loads(value)
        except (ValueError, TypeError):
            # Fall back to comma-separated values
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            return default


class Configuration(IConfiguration):
    """Implementation of application configuration."""
    
    def __init__(self, providers: List[ConfigurationProvider] = None):
        self._providers = providers or []
        self._data: Dict[str, str] = {}
        self.reload()
    
    @property
    def key(self) -> str:
        """Gets the key of the root configuration."""
        return ""
    
    @property
    def path(self) -> str:
        """Gets the path of the root configuration."""
        return ""
    
    @property
    def value(self) -> None:
        """The root configuration doesn't have a value."""
        return None
    
    def reload(self) -> None:
        """Reloads configuration from all providers."""
        self._data = {}
        
        # Load configuration from each provider in order
        for provider in self._providers:
            provider_data = provider.load()
            # Later providers override earlier ones
            self._data.update(provider_data)
    
    def get_section(self, key: str) -> IConfigurationSection:
        """Gets a configuration sub-section with the specified key."""
        return ConfigurationSection(self, key)
    
    def get_children(self) -> List[IConfigurationSection]:
        """Gets the immediate children sub-sections."""
        seen_keys = set()
        children = []
        
        for full_key in self._data.keys():
            # Get the first segment (up to the first ':')
            segment = full_key.split(':', 1)[0] if ':' in full_key else full_key
            
            if segment not in seen_keys:
                seen_keys.add(segment)
                children.append(ConfigurationSection(self, segment))
        
        return children
    
    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value with the specified key."""
        return self._data.get(key, default)
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Gets a configuration value as an integer."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Gets a configuration value as a float."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Gets a configuration value as a boolean."""
        value = self.get(key)
        if value is None:
            return default
        
        true_values = ('true', 'yes', '1', 'on')
        false_values = ('false', 'no', '0', 'off')
        
        if value.lower() in true_values:
            return True
        elif value.lower() in false_values:
            return False
        return default
    
    def get_dict(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Gets a configuration value as a dictionary."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, default: Optional[List] = None) -> Optional[List]:
        """Gets a configuration value as a list."""
        value = self.get(key)
        if value is None:
            return default
        
        try:
            # Try to parse as JSON
            return json.loads(value)
        except (ValueError, TypeError):
            # Fall back to comma-separated values
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            return default


###########################################
# Logging System
###########################################

class ILogger(ABC):
    """Interface for logging services."""
    
    @abstractmethod
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message."""
        pass
    
    @abstractmethod
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an informational message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message."""
        pass
    
    @abstractmethod
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log a critical message."""
        pass


class LoguruLogger(ILogger):
    """Implementation of ILogger using loguru."""
    
    def __init__(self, config: IConfiguration, context: str):
        self.config = config
        self.context = context
        
        log_level = config.get("Logging:Level", "TRACE")
        log_file = config.get("Logging:File")

        self.bound_logger = create_loguru_logger(context, log_level=log_level, log_file=log_file)

    def debug(self, message: str, *args, **kwargs) -> None:
        self.bound_logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        self.bound_logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        self.bound_logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        self.bound_logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        self.bound_logger.critical(message, *args, **kwargs)
    
    def with_context(self, context: str) -> 'LoguruLogger':
        """Create a new logger with the specified context."""
        return LoguruLogger(self.config, context)


###########################################
# Hosted Worker System
###########################################

class WorkerState(Enum):
    """Represents the state of a hosted worker."""
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()

class IWorker(ABC):
    """Interface for hosted services that run in the background."""
    
    @abstractmethod
    def start(self) -> None:
        """Starts the hosted service."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stops the hosted service."""
        pass

class Worker(IWorker):
    """Base class for hosted services that run in a background thread."""
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state = WorkerState.CREATED
    
    def start(self) -> None:
        """Starts the background service in a new thread."""
        if self._state != WorkerState.CREATED and self._state != WorkerState.STOPPED:
            return
            
        self._state = WorkerState.STARTING
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_worker, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Signals the background service to stop and waits for it to complete."""
        if self._state != WorkerState.RUNNING:
            return
            
        self._state = WorkerState.STOPPING
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)  # Wait up to 30 seconds for the thread to finish
        self._state = WorkerState.STOPPED
    
    def _run_worker(self) -> None:
        """Internal method that runs the worker thread."""
        try:
            self._state = WorkerState.RUNNING
            self.execute()
            self._state = WorkerState.STOPPED
        except Exception:
            self._state = WorkerState.FAILED
            logger.exception("Error in background service")
    
    @abstractmethod
    def execute(self) -> None:
        """Method to be implemented by derived classes to perform the background work."""
        pass
    
    def is_stopping(self) -> bool:
        """Checks if the service is stopping."""
        return self._stop_event.is_set()
    
    def wait_for_stop(self, timeout_seconds: float = None) -> bool:
        """Waits for the stop signal with an optional timeout."""
        return self._stop_event.wait(timeout=timeout_seconds)

class TimedWorker(Worker):
    """Background service that runs on a timed interval."""
    
    def __init__(self, interval_seconds: float = 5):
        super().__init__()
        self.interval_seconds = interval_seconds
    
    def execute(self) -> None:
        """Executes the timed service at the specified interval."""
        while not self.is_stopping():
            start_time = time.time()
            
            try:
                self.do_work()
            except Exception:
                logger.exception(f"Error in timed hosted service")
            
            # Calculate how long to wait before the next execution
            elapsed = time.time() - start_time
            wait_time = max(0, self.interval_seconds - elapsed)
            
            # Wait for the specified interval or until stop is requested
            if wait_time > 0:
                self.wait_for_stop(wait_time)
    
    @abstractmethod
    def do_work(self) -> None:
        """Method to be implemented by derived classes to perform the timed work."""
        pass

class WorkerManager:
    """Manages the lifecycle of hosted services."""
    
    def __init__(self, root_provider: 'ServiceProvider'):
        self._services: List[Tuple[IWorker, 'ServiceProvider']] = []
        self._started = False
        self._root_provider = root_provider
    
    def add_service(self, service_type: Type[IWorker]) -> None:
        """Adds a hosted service to be managed."""
        # Create a separate scope for each worker
        scope = self._root_provider.create_scope()
        service = scope.get_required_service(service_type)
        
        self._services.append((service, scope))
        if self._started:
            service.start()
    
    def start_all(self) -> None:
        """Starts all registered hosted services."""
        self._started = True
        for service, _ in self._services:
            service.start()
    
    def stop_all(self) -> None:
        """Stops all registered hosted services."""
        self._started = False
        # Stop services in reverse order
        for service, _ in reversed(self._services):
            try:
                service.stop()
            except Exception:
                logger.exception("Error stopping service")


###########################################
# Dependency Injection System
###########################################

class ServiceLifetime(Enum):
    """Defines the lifetime of a registered service."""
    SINGLETON = auto()  # One instance for the entire application
    SCOPED = auto()     # One instance per scope
    TRANSIENT = auto()  # New instance each time requested

class ServiceDescriptor:
    """Describes a service registration."""
    def __init__(
        self, 
        service_type: Type, 
        implementation_type: Optional[Type] = None,
        implementation_factory: Optional[Callable[['ServiceProvider'], Any]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        instance: Any = None
    ):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.implementation_factory = implementation_factory
        self.lifetime = lifetime
        self.instance = instance

class ApplicationBuilder:
    """Container for service registrations, similar to C#'s host builder pattern."""
    
    def __init__(self):
        self._descriptors: List[ServiceDescriptor] = []
        self._hosted_service_manager = None
        self._configuration_builder = ConfigurationBuilder()
        self._service_provider = None
        
        # Add environment variables by default
        self._configuration_builder.add_environment_variables()
    
    def add(self, descriptor: ServiceDescriptor) -> 'ApplicationBuilder':
        """Add a service descriptor to the collection."""
        self._descriptors.append(descriptor)
        return self
    
    def add_singleton(self, service_type: Type[T], implementation_type: Optional[Type] = None) -> 'ApplicationBuilder':
        """Register a singleton service."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.SINGLETON
        ))
    
    def add_singleton_instance(self, service_type: Type[T], instance: T) -> 'ApplicationBuilder':
        """Register an existing instance as a singleton."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        ))
    
    def add_singleton_factory(self, service_type: Type[T], factory: Callable[['ServiceProvider'], T]) -> 'ApplicationBuilder':
        """Register a singleton service with a factory function."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            implementation_factory=factory,
            lifetime=ServiceLifetime.SINGLETON
        ))
    
    def add_scoped(self, service_type: Type[T], implementation_type: Optional[Type] = None) -> 'ApplicationBuilder':
        """Register a scoped service."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.SCOPED
        ))
    
    def add_scoped_factory(self, service_type: Type[T], factory: Callable[['ServiceProvider'], T]) -> 'ApplicationBuilder':
        """Register a scoped service with a factory function."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            implementation_factory=factory,
            lifetime=ServiceLifetime.SCOPED
        ))
    
    def add_transient(self, service_type: Type[T], implementation_type: Optional[Type] = None) -> 'ApplicationBuilder':
        """Register a transient service."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.TRANSIENT
        ))
    
    def add_transient_factory(self, service_type: Type[T], factory: Callable[['ServiceProvider'], T]) -> 'ApplicationBuilder':
        """Register a transient service with a factory function."""
        return self.add(ServiceDescriptor(
            service_type=service_type,
            implementation_factory=factory,
            lifetime=ServiceLifetime.TRANSIENT
        ))
    
    def add_worker(self, implementation_type: Type[IWorker]) -> 'ApplicationBuilder':
        """Register a hosted service that will start when the service provider is built."""
        # Register as singleton to ensure there's only one instance
        self.add_singleton(implementation_type)
        
        # Register as IWorker for discovery
        if implementation_type != IWorker:
            self.add_singleton(IWorker, implementation_type)
        
        return self
    
    def add_configuration(self, configure_action: Callable[[ConfigurationBuilder], None]) -> 'ApplicationBuilder':
        """Configure the configuration system."""
        configure_action(self._configuration_builder)
        return self
    
    def add_configuration_dictionary(self, config_dict: Dict[str, Any]) -> 'ApplicationBuilder':
        """Add configuration from a dictionary."""
        # Flatten the dictionary if it contains nested dictionaries
        flat_dict = {}
        
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f"{prefix}:{key}" if prefix else key
                
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                else:
                    flat_dict[new_key] = str(value)
        
        flatten_dict(config_dict)
        
        # Add the flattened dictionary to the configuration
        self._configuration_builder.add_in_memory_collection(flat_dict)
        return self
    
    def build(self, auto_start_hosted_services: bool = True) -> 'ServiceProvider':
        """Build a service provider from this application builder."""
        # Build the configuration
        configuration = self._configuration_builder.build()
        
        # Register the configuration
        self.add_singleton_instance(IConfiguration, configuration)
        self.add_singleton_instance(Configuration, configuration)
        
        # Register the logger as a transient service that creates a new instance each time
        if not any(d.service_type == ILogger for d in self._descriptors):
            self.add_transient_factory(ILogger, lambda provider: LoguruLogger(
                provider.get_required_service(IConfiguration),
                "Default"  # Default context, will be overridden in _create_instance
            ))
        
        # Create the service provider
        provider = ServiceProvider(self._descriptors)
        
        # Discover and register hosted services
        if auto_start_hosted_services:
            provider.start_hosted_services()
        
        return provider
    
    def run(self) -> None:
        """Build the service provider and run the application until terminated."""
        # Build the service provider if not already built
        if not self._service_provider:
            self._service_provider = self.build(auto_start_hosted_services=True)
        
        # Get the logger
        log = self._service_provider.get_required_service(ILogger)
        
        log.info("Application started. Press Ctrl+C to exit.")
        
        # Set up signal handlers for graceful shutdown
        def handle_exit(sig, frame):
            log.info("Application shutting down...")
            if self._service_provider:
                self._service_provider.stop_hosted_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
        
        # Keep the application running
        try:
            # Block the main thread until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Application shutting down...")
            if self._service_provider:
                self._service_provider.stop_hosted_services()

class ServiceProvider:
    """Resolves services from the service collection, similar to C#'s IServiceProvider."""
    
    def __init__(self, descriptors: List[ServiceDescriptor]):
        self._descriptors = descriptors
        self._singleton_instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._hosted_service_manager = None  # Initialize to None
        
        # Register self as ServiceProvider
        self._singleton_instances[ServiceProvider] = self
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service of the specified type or None if not found."""
        for descriptor in reversed(self._descriptors):
            if descriptor.service_type == service_type:
                return self._resolve_service(descriptor)
        return None
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get a service of the specified type or raise an exception if not found."""
        service = self.get_service(service_type)
        if service is None:
            raise KeyError(f"Service {service_type.__name__} is not registered")
        return service
    
    def get_services(self, service_type: Type[T]) -> List[T]:
        """Get all services of the specified type."""
        services = []
        for descriptor in self._descriptors:
            if descriptor.service_type == service_type:
                service = self._resolve_service(descriptor)
                if service is not None:
                    services.append(service)
        return services
    
    def _resolve_service(self, descriptor: ServiceDescriptor) -> Any:
        """Resolve a service based on its descriptor and lifetime."""
        if descriptor.instance is not None:
            return descriptor.instance
        
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.implementation_type not in self._singleton_instances:
                self._singleton_instances[descriptor.implementation_type] = self._create_instance(descriptor)
            return self._singleton_instances[descriptor.implementation_type]
        
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if descriptor.implementation_type not in self._scoped_instances:
                self._scoped_instances[descriptor.implementation_type] = self._create_instance(descriptor)
            return self._scoped_instances[descriptor.implementation_type]
        
        else:  # TRANSIENT
            return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance of the service."""
        if descriptor.implementation_factory:
            return descriptor.implementation_factory(self)
        
        implementation_type = descriptor.implementation_type
        
        # Get constructor parameters
        init_signature = inspect.signature(implementation_type.__init__)
        init_params = init_signature.parameters
        
        # Skip 'self' parameter
        param_types = get_type_hints(implementation_type.__init__)
        if 'return' in param_types:
            del param_types['return']
        
        # Resolve dependencies
        dependencies = {}
        for param_name, param in init_params.items():
            if param_name == 'self':
                continue
                
            if param_name in param_types:
                param_type = param_types[param_name]
                dependency = self.get_service(param_type)
                
                # Set context for logger if it's being injected
                if dependency is not None and param_type == ILogger and isinstance(dependency, LoguruLogger):
                    # Use the class name as context
                    class_name = implementation_type.__name__
                    dependency = dependency.with_context(class_name)
                
                if dependency is None and param.default is param.empty:
                    raise ValueError(f"Cannot resolve parameter '{param_name}' of type {param_type} for {implementation_type.__name__}")
                
                if dependency is not None:
                    dependencies[param_name] = dependency
        
        # Create the instance with resolved dependencies
        return implementation_type(**dependencies)
    
    def create_scope(self) -> 'ServiceProvider':
        """Create a new scope with the same service registrations."""
        return ServiceScope(self)
    
    def start_hosted_services(self) -> None:
        """Discover and start all hosted services."""
        # Initialize worker manager if not already done
        if self._hosted_service_manager is None:
            self._hosted_service_manager = WorkerManager(self)
            
        worker_types = []
        for descriptor in self._descriptors:
            if (descriptor.service_type == IWorker or 
                (descriptor.implementation_type and issubclass(descriptor.implementation_type, IWorker))):
                worker_types.append(descriptor.implementation_type)
        
        for worker_type in worker_types:
            self._hosted_service_manager.add_service(worker_type)
        
        self._hosted_service_manager.start_all()
    
    def stop_hosted_services(self) -> None:
        """Stop all running hosted services."""
        if self._hosted_service_manager:
            self._hosted_service_manager.stop_all()

class ServiceScope(ServiceProvider):
    """Represents a scope for scoped services."""
    
    def __init__(self, root_provider: ServiceProvider):
        # Share descriptors and singleton instances with parent provider
        self._descriptors = root_provider._descriptors
        self._singleton_instances = root_provider._singleton_instances
        # New empty dict for scoped instances in this scope
        self._scoped_instances: Dict[Type, Any] = {}