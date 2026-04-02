import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder, ServiceDescriptor, ServiceLifetime
from interfaces import IGreeter, IFarewell
from services import (EnglishGreeter, FrenchGreeter, SpanishGreeter,
                      EnglishFarewell, FrenchFarewell)
from workers import DemoWorker

app = ApplicationBuilder()

# Register English greeter
app.add_singleton(IGreeter, EnglishGreeter)

# TryAdd won't add because IGreeter already registered
app.try_add_singleton(IGreeter, FrenchGreeter)

# Add Spanish as a second registration (multi-binding)
app.add_singleton(IGreeter, SpanishGreeter)

# Register English farewell, then replace with French
app.add_singleton(IFarewell, EnglishFarewell)
app.replace(ServiceDescriptor(
    service_type=IFarewell,
    implementation_type=FrenchFarewell,
    lifetime=ServiceLifetime.SINGLETON,
))

# RemoveAll farewell (we don't inject it anyway — just demonstrating the API)
app.remove_all(IFarewell)

app.add_worker(DemoWorker)

app.run()
