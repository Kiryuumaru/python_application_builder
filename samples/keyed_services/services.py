from interfaces import INotificationSender


class EmailSender(INotificationSender):
    def send(self, message: str) -> str:
        return f"[EMAIL] {message}"


class SmsSender(INotificationSender):
    def send(self, message: str) -> str:
        return f"[SMS] {message}"


class SlackSender(INotificationSender):
    def send(self, message: str) -> str:
        return f"[SLACK] {message}"
