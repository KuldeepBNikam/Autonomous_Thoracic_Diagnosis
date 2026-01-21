import os
import sys
import HyperLung_XR
from HyperLung_XR.exception import HyperLungException

class S3operation:

    def sync_folder_to_s3(self, folder: str,bucket_name: str,bucket_folder_name: str)->None:
        try:
            command: str = (f'aws s3 sync "{folder}" '
                            f'"s3://{bucket_name}/{bucket_folder_name}"')

            os.system(command)
        except Exception as e:
            raise HyperLungException(e, sys)
        

    def sync_folder_from_s3(self,folder: str,bucket_name: str,bucket_folder_name: str)->None:
        try:
            os.makedirs(folder, exist_ok=True)
            command: str = (f'aws s3 sync '
                            f'"s3://{bucket_name}/{bucket_folder_name}" ' 
                            f'"{folder}"')

            os.system(command)

        except Exception as e:
            raise HyperLungException(e,sys)
