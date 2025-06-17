
from application.services.base_animal_service import BaseAnimalService
from application.services.base_food_service import BaseFoodService


class DogService(BaseAnimalService):
    
    def speak(self):
        food_factory = self.get_factory(BaseFoodService)

        food1 = food_factory.create()
        food2 = food_factory.create()
        food3 = food_factory.create()

        food1.name = "bone"
        food2.name = "meat"
        food3.name = "dog treats"

        food1.make_food()
        food2.make_food()
        food3.make_food()

        self.logger.info(f"Dog is speaking with food: {food1.name}, {food2.name}, {food3.name}")
