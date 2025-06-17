
from application.services.base_animal_service import BaseAnimalService
from application.services.base_food_service import BaseFoodService

class CatService(BaseAnimalService):

    def speak(self):
        food_factory = self.get_factory(BaseFoodService)

        food1 = food_factory.create()
        food2 = food_factory.create()
        food3 = food_factory.create()
        
        food1.name = "salmon"
        food2.name = "chicken"
        food3.name = "cat treats"

        food1.make_food()
        food2.make_food()
        food3.make_food()

        self.logger.info(f"Cat is speaking with food: {food1.name}, {food2.name}, {food3.name}")
