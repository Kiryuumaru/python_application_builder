
from typing import Union
from application.application_builder import Dependency

class BaseFoodService(Dependency):
    def __init__(self):
        self.name: Union[str, None] = None

    def make_food(self):
        self.logger.info(f"Making {self.name} food")
