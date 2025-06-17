
from typing import Union
from application.application_builder import Dependency

class BaseAnimalService(Dependency):

    def speak(self):
        raise NotImplementedError()
