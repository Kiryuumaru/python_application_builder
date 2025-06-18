from __future__ import annotations

from typing import List, Dict, Union, TypeVar, Generic, Callable, Any
import threading
import sys
import os
import loguru

TDependency = TypeVar("TDependency", bound='Dependency')
TWorker = TypeVar("TWorker", bound='Worker')


class Meta(type):
    """
    Metaclass that handles automatic initialization of base classes.
    Ensures that parent class __init__ methods are called before child class __init__.
    """
    def __init__(cls, name, bases, dct):
        def auto__call__init__(self, *a, **kw):
            for base in cls.__bases__:
                base.__init__(self, *a, **kw)
            cls.__init__child_(self, *a, **kw)
        cls.__init__child_ = cls.__init__
        cls.__init__ = auto__call__init__


class DependencyCore:
    """
    Core functionality shared between Dependency and DependencyFactory classes.
    Manages dependency metadata and application builder reference.
    """
    def __init__(self):
        self._dependency_name: Union[str, None] = None
        self._dependency_keys: List[str] = []
        self._dependency_local_key: Union[str, None] = None
        self.application_builder: Union[ApplicationBuilder, None] = None


class Dependency(DependencyCore, metaclass=Meta):
    """
    Base class for all injectable components in the application.
    Provides access to services, factories, and configuration.
    """
    def __init__(self):
        self._dependency_is_initialized = False
        self.logger: Union[loguru.Logger, None] = None

    def get_configuration(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key to retrieve
            default: Optional default value if the key doesn't exist
            
        Returns:
            The configuration value or default
            
        Raises:
            Exception: If the key doesn't exist and no default is provided
        """
        if key in self.application_builder.configurations:
            return self.application_builder.configurations[key]
        if default is not None:
            return default
        raise Exception(f"Configuration '{key}' does not exist")

    def get_services(self, service_type: type[TDependency]) -> Dict[str, TDependency]:
        """
        Get all registered services of the specified type.
        
        Args:
            service_type: The type of services to retrieve
            
        Returns:
            Dictionary of services with their local keys
            
        Raises:
            Exception: If no services of the specified type exist
        """
        key = service_type.__name__
        if key not in self.application_builder.services:
            raise Exception(f"Service '{key}' does not exist")
        services: Dict[str, TDependency] = {}
        for key, service in self.application_builder.services[key].items():
            services[key] = service
        return services

    def get_service(self, service_type: type[TDependency], local_key: Union[str, None] = None) -> TDependency:
        """
        Get a specific service of the specified type.
        
        Args:
            service_type: The type of service to retrieve
            local_key: Optional key to identify a specific service implementation
            
        Returns:
            The requested service instance
            
        Raises:
            Exception: If no service with the specified key exists or if multiple services
                      exist but no key is provided
        """
        key = service_type.__name__
        services = self.get_services(service_type)
        if local_key is None and len(services) > 1:
            raise Exception(f"Service '{key}' have more than one implementation: {[s._dependency_local_key for s in services.values()]}")
        if local_key is not None:
            if local_key not in services:
                raise Exception(f"Service '{local_key}' does not exist in '{key}'")
            return services[local_key]
        return next(iter(services.values()))

    def get_factories(self, factory_type: type[TDependency]) -> Dict[str, DependencyFactory[TDependency]]:
        """
        Get all registered factories for the specified type.
        
        Args:
            factory_type: The type of factories to retrieve
            
        Returns:
            Dictionary of factories with their local keys
            
        Raises:
            Exception: If no factories of the specified type exist
        """
        key = factory_type.__name__
        if key not in self.application_builder.factories:
            raise Exception(f"Factory '{key}' does not exist")
        factories: Dict[str, DependencyFactory[TDependency]] = {}
        for key, factory in self.application_builder.factories[key].items():
            factories[key] = factory
        return factories

    def get_factory(self, factory_type: type[TDependency], local_key: Union[str, None] = None) -> DependencyFactory[TDependency]:
        """
        Get a specific factory for the specified type.
        
        Args:
            factory_type: The type of factory to retrieve
            local_key: Optional key to identify a specific factory implementation
            
        Returns:
            The requested factory instance
            
        Raises:
            Exception: If no factory with the specified key exists or if multiple factories
                      exist but no key is provided
        """
        key = factory_type.__name__
        factories = self.get_factories(factory_type)
        if local_key is None and len(factories) > 1:
            raise Exception(f"Factory '{key}' have more than one implementation: {[s._dependency_local_key for s in factories.values()]}")
        if local_key is not None:
            if local_key not in factories:
                raise Exception(f"Factory '{local_key}' does not exist in '{key}'")
            return factories[local_key]
        return next(iter(factories.values()))

    def initialize(self):
        """
        Initialize the dependency. Called after dependency creation.
        Override this method to perform initialization logic.
        """
        pass


class DependencyFactory(Generic[TDependency], DependencyCore):
    """
    Factory for creating instances of a specific dependency type.
    Manages the creation and initialization of new instances.
    """
    def __init__(self, dependency_factory: Union[Callable[[], TDependency], None]):
        """
        Initialize a new dependency factory.
        
        Args:
            dependency_factory: Function that creates new instances of the dependency
        """
        super().__init__()
        self._dependency_factory: Callable[[], TDependency] = dependency_factory

    def create(self) -> TDependency:
        """
        Create a new instance of the dependency.
        
        Returns:
            A new, initialized instance of the dependency
        """
        service = self._dependency_factory()
        self.application_builder._wire_dependency(service, self._dependency_name)
        service.logger.debug(f"Service '{service._dependency_local_key}' initialized")
        service.initialize()
        return service


class Worker(Dependency):
    """
    Base class for worker components that run in separate threads.
    """
    def run(self):
        """
        Execute the worker's main logic. Must be implemented by subclasses.
        """
        raise NotImplementedError()


class ApplicationBuilder:
    """
    Main dependency injection container that manages services, factories, and workers.
    Coordinates application lifecycle and provides configuration management.
    """
    def __init__(self, log_dir: str = os.path.join(os.getcwd(), "logs")):
        """
        Initialize a new application builder.
        
        Args:
            log_dir: Directory where log files will be stored
        """
        self.configurations: Dict[str, Any] = {}
        self.services: Dict[str, Dict[str, TDependency]] = {}
        self.factories: Dict[str, Dict[str, DependencyFactory[TDependency]]] = {}
        self.services_to_initialize: List[TDependency] = []
        self.factories_to_initialize: List[TDependency] = []
        self.workers: Dict[str, TWorker] = {}
        self.log_dir = log_dir
        self.log_format = '<fg 95,95,95>{time}</> <level>{level: <8}</> - ' \
                          '[<fg 95,95,95>{extra[context]}</>] <level>{message}</>'

    def _wire_dependency(self, dependency: TDependency, key: Union[str, None] = None, name: Union[str, None] = None):
        """
        Configure a dependency with the application builder and logger.
        
        Args:
            dependency: The dependency to configure
            key: Optional local key for the dependency
            name: Optional name for the dependency
        """
        dependency._dependency_name = dependency.__class__.__name__ if name is None else name
        dependency._dependency_local_key = dependency.__class__.__name__ if key is None else key
        dependency._dependency_keys.append(dependency.__class__.__name__)
        for base_class in dependency.__class__.__bases__:
            self._add_base_classes(dependency._dependency_keys, base_class)
        dependency.application_builder = self
        dependency.logger = loguru.logger.bind(context=dependency._dependency_name)
        dependency.logger.level("TRACE", 
                                color="<fg #444444>")
        dependency.logger.level("DEBUG", 
                                color="<fg #666666>")
        dependency.logger.level("INFO", 
                                color="<fg #FFFFFF>")
        dependency.logger.level("SUCCESS", 
                                color="<fg #00CC99>")
        dependency.logger.level("WARNING", 
                                color="<fg #FFBB00>")
        dependency.logger.level("ERROR", 
                                color="<fg #FF4444>")
        dependency.logger.level("CRITICAL", 
                                color="<fg #FF00FF>")

    def _wire_dependency_factory(self, dependency_factory: DependencyFactory[TDependency], key: Union[str, None] = None, name: Union[str, None] = None):
        """
        Configure a dependency factory with the application builder.
        
        Args:
            dependency_factory: The factory to configure
            key: Optional local key for the factory
            name: Optional name for the factory
        """
        dependency_type = dependency_factory.__orig_class__.__args__[0]
        dependency_factory._dependency_name = dependency_type.__name__ if name is None else name
        dependency_factory._dependency_local_key = dependency_type.__name__ if key is None else key
        dependency_factory._dependency_keys.append(dependency_type.__name__)
        for base_class in dependency_type.__bases__:
            self._add_base_classes(dependency_factory._dependency_keys, base_class)
        dependency_factory.application_builder = self

    def _add_base_classes(self, keys: List[str], base_class: type):
        """
        Add base class names to the dependency keys list.
        
        Args:
            keys: List of keys to update
            base_class: Base class to process
        """
        keys.append(base_class.__name__)
        for parent_class in base_class.__bases__:
            self._add_base_classes(keys, parent_class)

    def _initialize_service(self, dependency: Dependency):
        """
        Initialize a service if not already initialized.
        
        Args:
            dependency: The service to initialize
        """
        if not dependency._dependency_is_initialized:
            dependency.initialize()
            dependency._dependency_is_initialized = True
            dependency.logger.debug(f"Service '{dependency._dependency_local_key}' initialized")

    def _run_worker(self, worker: Worker):
        """
        Run a worker in its own thread.
        
        Args:
            worker: The worker to run
        """
        worker.logger.debug(f"Worker '{worker._dependency_local_key}' started")
        worker.run()

    def add_configuration(self, key: str, value: Any):
        """
        Add a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.configurations[key] = value

    def add_service(self, service_type: type[TDependency], local_key: Union[str, None] = None, name: Union[str, None] = None):
        """
        Register a service with the application builder.
        
        Args:
            service_type: Type of service to create
            local_key: Optional key to identify this service
            name: Optional name for the service
            
        Raises:
            Exception: If a service with the same key already exists
        """
        service = service_type()
        self._wire_dependency(service, local_key, name)
        self.services_to_initialize.append(service)
        for key in service._dependency_keys:
            if key not in self.services:
                self.services[key] = {}
            if service._dependency_local_key in self.services[key]:
                raise Exception(f"Service '{service._dependency_local_key}' already exists in '{key}'")
            self.services[key][service._dependency_local_key] = service

    def add_factory(self, factory_type: type[TDependency], local_key: Union[str, None] = None, 
                    name: Union[str, None] = None, custom_factory: Union[Callable[[], TDependency], None] = None):
        """
        Register a factory for creating instances of the specified type.
        
        Args:
            factory_type: Type of dependency to create
            local_key: Optional key to identify this factory
            name: Optional name for the factory
            custom_factory: Optional custom factory function that returns instances of the dependency
            
        Raises:
            Exception: If a factory with the same key already exists
            
        Example:
            # Standard factory registration
            app_builder.add_factory(PetFoodService)
            
            # Custom factory function
            def create_custom_pet_food():
                pet_food = PetFoodService()
                pet_food.name = "premium"
                return pet_food
                
            # Register with custom factory
            app_builder.add_factory(PetFoodService, custom_factory=create_custom_pet_food)
        """
        def default_factory() -> TDependency:
            return factory_type()
        
        factory_function = custom_factory if custom_factory is not None else default_factory
        factory = DependencyFactory[factory_type](factory_function)
        self._wire_dependency_factory(factory, local_key, name)
        for key in factory._dependency_keys:
            if key not in self.factories:
                self.factories[key] = {}
            if factory._dependency_local_key in self.factories[key]:
                raise Exception(f"Factory '{factory._dependency_local_key}' already exists in '{key}'")
            self.factories[key][factory._dependency_local_key] = factory

    def add_worker(self, worker_type: type[TWorker], name: Union[str, None] = None):
        """
        Register a worker with the application builder.
        
        Args:
            worker_type: Type of worker to create
            name: Optional name for the worker
            
        Raises:
            Exception: If a worker with the same key already exists
        """
        worker = worker_type()
        self._wire_dependency(worker, name)
        key = worker._dependency_keys[0]
        if key in self.workers:
            raise Exception(f"Worker '{key}' already exists")
        self.workers[key] = worker

    def run(self):
        """
        Run the application.
        
        This method:
        1. Configures the logging system
        2. Loads environment variables as configurations
        3. Initializes all registered services
        4. Starts worker threads
        5. Waits for all worker threads to complete
        """
        loguru.logger.remove()
        loguru.logger.add(sys.stdout, colorize=True, format=self.log_format, level="TRACE")
        loguru.logger.add(os.path.join(self.log_dir, "main_{time:YYYY-MM-DD}.log"), format=self.log_format, level="TRACE",
                          rotation="1 day", retention="30 days")

        for key, value in os.environ.items():
            if key not in self.configurations:
                self.configurations[key] = value
                
        for service in self.services_to_initialize:
            self._initialize_service(service)

        threads = []
        for worker in self.workers.values():
            thread = threading.Thread(target=self._run_worker, args=[worker,])
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()