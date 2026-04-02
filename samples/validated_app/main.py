import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IEmailService, ICache
from services import SmtpEmailService, RedisCache
from workers import AppWorker

app = ApplicationBuilder()

# Enable build-time validation — catches missing registrations early
app.set_validate_on_build(True)

# Enable scope validation — prevents scoped services from leaking via root provider
app.set_validate_scopes(True)

# Register all services (remove one to see ValidateOnBuild catch the error)
app.add_singleton(IEmailService, SmtpEmailService)
app.add_singleton(ICache, RedisCache)

app.add_worker(AppWorker)

app.run()
