from interfaces import IGreeter, IFarewell


class EnglishGreeter(IGreeter):
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"


class FrenchGreeter(IGreeter):
    def greet(self, name: str) -> str:
        return f"Bonjour, {name}!"


class SpanishGreeter(IGreeter):
    def greet(self, name: str) -> str:
        return f"¡Hola, {name}!"


class EnglishFarewell(IFarewell):
    def farewell(self, name: str) -> str:
        return f"Goodbye, {name}!"


class FrenchFarewell(IFarewell):
    def farewell(self, name: str) -> str:
        return f"Au revoir, {name}!"
