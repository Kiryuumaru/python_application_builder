from __future__ import annotations

from typing import List, Dict, Union, TypeVar, Generic, Callable, Any
import threading
import sys
import os
import loguru

TDependency = TypeVar("TDependency", bound='Dependency')
TWorker = TypeVar("TWorker", bound='Worker')


class Meta(type):
    def __init__(cls, name, bases, dct):
        def auto__call__init__(self, *a, **kw):
            for base in cls.__bases__:
                base.__init__(self, *a, **kw)
            cls.__init__child_(self, *a, **kw)
        cls.__init__child_ = cls.__init__
        cls.__init__ = auto__call__init__


class DependencyCore:
    def __init__(self):
        self._dependency_name: Union[str, None] = None
        self._dependency_keys: List[str] = []
        self._dependency_local_key: Union[str, None] = None
        self.application_builder: Union[ApplicationBuilder, None] = None


class Dependency(DependencyCore, metaclass=Meta):
    def __init__(self):
        self._dependency_is_initialized = False
        self.logger: Union[loguru.Logger, None] = None

    def get_services(self, service_type: type[TDependency]) -> Dict[str, TDependency]:
        return self.application_builder.get_services(service_type)

    def get_service(self, service_type: type[TDependency], local_key: Union[str, None] = None) -> TDependency:
        return self.application_builder.get_service(service_type, local_key)

    def get_factories(self, factory_type: type[TDependency]) -> Dict[str, DependencyFactory[TDependency]]:
        return self.application_builder.get_factories(factory_type)

    def get_factory(self, factory_type: type[TDependency], local_key: Union[str, None] = None) -> DependencyFactory[TDependency]:
        return self.application_builder.get_factory(factory_type, local_key)

    def get_configuration(self, key: str, default: Any = None) -> Any:
        return self.application_builder.get_configuration(key, default)

    def initialize(self):
        pass


class DependencyFactory(Generic[TDependency], DependencyCore):
    def __init__(self, dependency_factory: Union[Callable[[], TDependency], None]):
        super().__init__()
        self._dependency_factory: Callable[[], TDependency] = dependency_factory

    def create(self) -> TDependency:
        service = self._dependency_factory()
        self.application_builder._wire_dependency(service, self._dependency_name)
        service.logger.info(f"Creating service '{self._dependency_name}' from factory")
        service.initialize()
        return service


class Worker(Dependency):
    def run(self):
        raise NotImplementedError()


class ApplicationBuilder:
    def __init__(self, log_dir: str = os.path.join(os.getcwd(), "logs")):
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
        dependency._dependency_name = dependency.__class__.__name__ if name is None else name
        dependency._dependency_local_key = dependency.__class__.__name__ if key is None else key
        dependency._dependency_keys.append(dependency.__class__.__name__)
        for base_class in dependency.__class__.__bases__:
            self._add_base_classes(dependency._dependency_keys, base_class)
        dependency.application_builder = self
        dependency.logger = loguru.logger.bind(context=dependency._dependency_name)

    def _wire_dependency_factory(self, dependency_factory: DependencyFactory[TDependency], key: Union[str, None] = None, name: Union[str, None] = None):
        dependency_type = dependency_factory.__orig_class__.__args__[0]
        dependency_factory._dependency_name = dependency_type.__name__ if name is None else name
        dependency_factory._dependency_local_key = dependency_type.__name__ if key is None else key
        dependency_factory._dependency_keys.append(dependency_type.__name__)
        for base_class in dependency_type.__bases__:
            self._add_base_classes(dependency_factory._dependency_keys, base_class)
        dependency_factory.application_builder = self

    def _add_base_classes(self, keys: List[str], base_class: type):
        keys.append(base_class.__name__)
        for parent_class in base_class.__bases__:
            self._add_base_classes(keys, parent_class)

    def _initialize_service(self, dependency: Dependency):
        if not dependency._dependency_is_initialized:
            dependency.initialize()
            dependency._dependency_is_initialized = True
            dependency.logger.info("Service initialized")

    def _run_worker(self, worker: Worker):
        worker.logger.info("Worker started")
        worker.run()

    def add_configuration(self, key: str, value: Any):
        self.configurations[key] = value

    def add_service(self, service_type: type[TDependency], local_key: Union[str, None] = None, name: Union[str, None] = None):
        service = service_type()
        self._wire_dependency(service, local_key, name)
        self.services_to_initialize.append(service)
        for key in service._dependency_keys:
            if key not in self.services:
                self.services[key] = {}
            if service._dependency_local_key in self.services[key]:
                raise Exception(f"Service '{service._dependency_local_key}' already exists in '{local_key}'")
            self.services[key][service._dependency_local_key] = service

    def add_factory(self, factory_type: type[TDependency], local_key: Union[str, None] = None, name: Union[str, None] = None):
        def factory_type_factory() -> TDependency:
            return factory_type()
        factory = DependencyFactory[factory_type](factory_type_factory)
        self._wire_dependency_factory(factory, local_key, name)
        for key in factory._dependency_keys:
            if key not in self.factories:
                self.factories[key] = {}
            if factory._dependency_local_key in self.factories[key]:
                raise Exception(f"Factory '{factory._dependency_local_key}' already exists in '{local_key}'")
            self.factories[key][factory._dependency_local_key] = factory

    def add_worker(self, worker_type: type[TWorker], name: Union[str, None] = None):
        worker = worker_type()
        self._wire_dependency(worker, name)
        key = worker._dependency_keys[0]
        if key in self.workers:
            raise Exception(f"Worker '{key}' already exists")
        self.workers[key] = worker

    def get_configuration(self, key: str, default: Any = None) -> Any:
        if key in self.configurations:
            return self.configurations[key]
        if default is not None:
            return default
        raise Exception(f"Configuration '{key}' does not exist")

    def get_services(self, service_type: type[TDependency]) -> Dict[str, TDependency]:
        key = service_type.__name__
        if key not in self.services:
            raise Exception(f"Service '{key}' does not exist")
        services: Dict[str, TDependency] = {}
        for key, service in self.services[key].items():
            services[key] = service
        return services

    def get_service(self, service_type: type[TDependency], local_key: Union[str, None] = None) -> TDependency:
        key = service_type.__name__
        services = self.get_services(service_type)
        if local_key is None and len(services) > 1:
            raise Exception(f"Service '{key}' have more than one implementation: {[s._dependency_keys[0] for s in services.values()]}")
        if local_key is not None:
            if local_key not in services:
                raise Exception(f"Service '{local_key}' does not exist in '{key}'")
            return services[local_key]
        return next(iter(services.values()))

    def get_factories(self, factory_type: type[TDependency]) -> Dict[str, DependencyFactory[TDependency]]:
        key = factory_type.__name__
        if key not in self.factories:
            raise Exception(f"Factory '{key}' does not exist")
        factories: Dict[str, DependencyFactory[TDependency]] = {}
        for key, factory in self.factories[key].items():
            factories[key] = factory
        return factories

    def get_factory(self, factory_type: type[TDependency], local_key: Union[str, None] = None) -> DependencyFactory[TDependency]:
        key = factory_type.__name__
        factories = self.get_factories(factory_type)
        if local_key is None and len(factories) > 1:
            raise Exception(f"Factory '{key}' have more than one implementation: {[s._dependency_keys[0] for s in factories.values()]}")
        if local_key is not None:
            if local_key not in factories:
                raise Exception(f"Factory '{local_key}' does not exist in '{key}'")
            return factories[local_key]
        return next(iter(factories.values()))

    def run(self):
        loguru.logger.remove()
        loguru.logger.add(sys.stdout, colorize=True, format=self.log_format)
        loguru.logger.add(os.path.join(self.log_dir, "main_{time:YYYY-MM-DD}.log"), format=self.log_format,
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
