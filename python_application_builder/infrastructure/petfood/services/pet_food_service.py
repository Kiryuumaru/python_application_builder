
from application.services.base_food_service import BaseFoodService

class PetFoodService(BaseFoodService):

    def initialize(self):
        super().initialize()
        self.logger.info(f"Initializing pet food from infra")
        self.name = "default"
