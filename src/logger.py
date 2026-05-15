import logging
import os
import sys
from datetime import datetime

# 1. Setup paths
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 2. Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 3. Create the format
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

# 4. Create File Handler and add to logger
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 5. Create Terminal Handler and add to logger
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    logging.info("FORCE LOG TEST")
    # This manually forces the file to save immediately
    for handler in logger.handlers:
        handler.flush()