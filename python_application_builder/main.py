from application_builder import ApplicationBuilder
from workers.main_worker import MainWorker
from infrastructure.animal_samples.services.cat_service import CatService
from infrastructure.animal_samples.services.dog_service import DogService
from infrastructure.petfood.services.pet_food_service import PetFoodService


app_builder: ApplicationBuilder = ApplicationBuilder()

# Add services
app_builder.add_service(CatService, "special")
app_builder.add_service(DogService)

# Add factories
app_builder.add_factory(PetFoodService)

# Add workers
app_builder.add_worker(MainWorker)

app_builder.run()
