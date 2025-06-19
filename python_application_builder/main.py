from application_builder import ApplicationBuilder, Worker, TimedWorker, IConfiguration, ILogger
import time

# Define a custom worker
class CustomWorker(Worker):
    def __init__(self, config: IConfiguration, logger: ILogger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.app_name = config.get("App:Name", "Unknown Application")
        self.current_time = config.get("System:CurrentTime", "Unknown Time")
        self.current_user = config.get("UserDetails:Name", "Unknown User")
    
    def execute(self) -> None:
        self.logger.info(f"Starting custom worker for {self.app_name}")
        self.logger.info(f"Current time: {self.current_time}")
        self.logger.info(f"Current user: {self.current_user}")
        
        counter = 0
        while not self.is_stopping():
            counter += 1
            self.logger.debug(f"Custom worker is running... (count: {counter})")
            
            if counter % 5 == 0:
                self.logger.info(f"Custom worker milestone: {counter} iterations")
            
            # Simulate some work
            time.sleep(2)
        
        self.logger.info("Custom worker is stopping")


class CustomTimedWorker(TimedWorker):
    def __init__(self, config: IConfiguration, logger: ILogger):
        super().__init__(interval_seconds=1)
        self.config = config
        self.logger = logger
        self.counter = 0
    
    def do_work(self) -> None:
        self.counter += 1
        self.logger.debug(f"Custom worker is running... (count: {self.counter})")
            
        if self.counter % 5 == 0:
            self.logger.info(f"Custom worker milestone: {self.counter} iterations")

def main():
    # Create the application builder
    app = ApplicationBuilder()
    
    # Add additional configuration
    app.add_configuration_dictionary({
        "App": {
            "Name": "Custom Service Demo"
        },
        "Logging": {
            "Level": "DEBUG"
        }
    })
    
    # Add our custom worker
    app.add_worker(CustomWorker)
    app.add_worker(CustomTimedWorker)
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main()