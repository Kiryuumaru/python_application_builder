import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IMessageFormatter, IRoomRegistry, ISessionContext
from services import (
    TimestampFormatter, UsernameFormatter, InMemoryRoomRegistry, SessionContext,
)
from workers import ChatSimulatorWorker


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "Chat": {
        "SimulationIntervalSeconds": "1",
    }
})

app.add_singleton(IRoomRegistry, InMemoryRoomRegistry)
app.add_scoped(ISessionContext, SessionContext)

# Multiple formatters — injected as List[IMessageFormatter]
app.add_singleton(IMessageFormatter, TimestampFormatter)
app.add_singleton(IMessageFormatter, UsernameFormatter)

app.add_worker(ChatSimulatorWorker)
app.run()
