from application_builder import ApplicationBuilder, Worker, ILogger
import time


class SimpleWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self.logger = logger

    def execute(self) -> None:
        self.logger.info("SimpleWorker started")
        for i in range(3):
            if self.is_stopping():
                break
            self.logger.info(f"SimpleWorker running iteration {i+1}")
            time.sleep(1)
        self.logger.info("SimpleWorker finished")
        

app = ApplicationBuilder()
app.add_worker(SimpleWorker)
app.run()