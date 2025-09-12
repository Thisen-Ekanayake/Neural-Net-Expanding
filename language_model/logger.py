import sys
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_file=None):
        os.makedirs(log_dir, exist_ok=True)
        if log_file is None:
            log_file = datetime.now().strftime("log_%Y%m%d_%H%M%S.txt")
        self.log_path = os.path.join(log_dir, log_file)
        self.terminal = sys.stdout
        self.log = open(self.log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Usage:
# import logger
# log = logger.Logger()
# print = log.write
# print("Hello world\n")
