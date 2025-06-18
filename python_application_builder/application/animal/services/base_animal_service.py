
from application_builder import Dependency

class BaseAnimalService(Dependency):

    def speak(self):
        raise NotImplementedError()
