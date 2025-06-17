from application.services.base_animal_service import BaseAnimalService
from application.application_builder import Worker


class MainWorker(Worker):
    
    def run(self):
        
        animal_services = self.get_services(BaseAnimalService)

        for animal_service in animal_services:
            animal_service.speak()
