import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import INotificationSender
from services import EmailSender, SmsSender, SlackSender
from workers import NotificationWorker

app = ApplicationBuilder()

# Register the same interface with different keys
app.add_keyed_singleton(INotificationSender, "email", EmailSender)
app.add_keyed_singleton(INotificationSender, "sms", SmsSender)
app.add_keyed_singleton(INotificationSender, "slack", SlackSender)

app.add_worker(NotificationWorker)

app.run()
