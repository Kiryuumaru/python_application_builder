import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder, ILogger
from interfaces import IRepository
from services import SqlRepository, LoggingRepositoryDecorator, CachingRepositoryDecorator
from workers import QueryWorker

app = ApplicationBuilder()

# Register the base repository
app.add_singleton(IRepository, SqlRepository)

# Wrap with logging (applied first — innermost decorator)
app.decorate(IRepository, lambda provider, inner: LoggingRepositoryDecorator(
    inner, provider.get_required_service(ILogger).with_context("LoggingDecorator"),
))

# Wrap with caching (applied second — outermost decorator)
app.decorate(IRepository, lambda provider, inner: CachingRepositoryDecorator(
    inner, provider.get_required_service(ILogger).with_context("CachingDecorator"),
))

app.add_worker(QueryWorker)

app.run()
