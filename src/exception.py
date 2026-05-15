import sys # sys module in python provides various functions and variables used to manipulate different parts of the python runtime
from src.logger import logging
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    
    file_name=exc_tb.tb_frame.f_code.co_filename # 1. exc_tb holds the map of crash
    # 2.tb_frame: This enters the "Execution Frame." 3. f_code: This accesses the "Code Object" within that frame
    # 4.co_filename: This is the final piece of data contains the path and name of the Python file
    
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    ) # Added closing parenthesis here

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Added () to super and fixed the call
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        # This ensures that when you print the error, you see the detailed message
        return self.error_message

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e,sys)
            





    
    # This is the heavy lifter. When an error happens, this function returns a tuple (a fixed list) containing three pieces of information:
    # Type: The category of the error (e.g., ZeroDivisionError).
    # Value: The error message (e.g., "division by zero").
    # Traceback: A special object that contains the file name and line number where the error lived.
    # _, _, exc_tb: This is called unpacking.
    # The underscores (_) are a Python convention meaning, "I know there's data here, but I don't care about it right now."
    # The code is intentionally ignoring the Type and the Value because it only wants the third item: the Traceback (exc_tb).