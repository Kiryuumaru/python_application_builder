from application_builder import ApplicationBuilder, Worker, ILogger
from typing import List
import time


class BaseHelloService:
    def say_hello(self, name: str) -> str:
        return f"Hello, {name}!"


class SimpleHelloService(BaseHelloService):
    def say_hello(self, name: str) -> str:
        return f"Simple Hello, {name}!"


class BonjourHelloService(BaseHelloService):
    def say_hello(self, name: str) -> str:
        return f"Bonjour Hello, {name}!"


class SimpleWorker(Worker):
    def __init__(self, logger: ILogger, hello_services: List[BaseHelloService]):
        super().__init__()
        self.logger = logger
        self.hello_services = hello_services

    def execute(self) -> None:
        self.logger.info("SimpleWorker started")
        for i in range(3):
            if self.is_stopping():
                break
            self.logger.info(f"SimpleWorker running iteration {i+1}")
            for service in self.hello_services:
                greeting = service.say_hello("World")
                self.logger.info(greeting)
            time.sleep(1)
        self.logger.info("SimpleWorker finished")
        

app = ApplicationBuilder()
app.add_scoped(BaseHelloService, SimpleHelloService)
app.add_scoped(BaseHelloService, BonjourHelloService)
app.add_worker(SimpleWorker)
app.run()