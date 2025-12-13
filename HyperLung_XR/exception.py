import sys
import os

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()#helps to capture error info
    filename = exc_tb.tb_frame.f_code.co_filename  #check which file has error
    error_message="error occured and the file name is [{0}] and the linenumber is [{1}] and error is[{2}]".format(
    filename,exc_tb.tb_lineno,str(error))

    return error_message


class HyperLungException(Exception):  #inheriting exception class
    def __init__(self,error_message,error_detail:sys):  #to capture error related details
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message,error_detail=error_detail)


    def __str__(self) :                # Dunder function to represent in string form
        return self.error_message