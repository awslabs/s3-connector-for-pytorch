import os
import boto3
from botocore.exceptions import ClientError
import sys

BUCKET_NAME = 'iisaev-pytorch-benchmarks-results'
ROOT_FOLDER = './multirun'
DS_PREFIX = 's3iterabledataset'
FOLDER_PREFIX = 'results'

s3_client = boto3.client('s3')

def upload_file_to_s3(local_file_path, bucket_name, s3_file_key):
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_file_key)
        print(f"Uploaded {local_file_path} to {bucket_name}/{s3_file_key}")
    except ClientError as e:
        print(f"Error uploading {local_file_path} to {bucket_name}/{s3_file_key}: {e}")

def traverse_folders(folder_path, bucket_name, prefix, dataloader):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'collated_results.json':
                local_file_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(local_file_path))
                s3_file_key = f"{prefix}/{dataloader}_{parent_folder}_{file}"
                print(f"Uploading {local_file_path} to {bucket_name}/{s3_file_key}")
                upload_file_to_s3(local_file_path, bucket_name, s3_file_key)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python script.py ROOT_FOLDER BUCKET_NAME FOLDER_PREFIX DS_PREFIX")
        print("Example: python script.py ./multirun pytorch-benchmarks-results 20240810 s3iterabledataset")
        print("Note: ROOT_FOLDER is the root folder where the results are stored")
        print("Note: BUCKET_NAME is the S3 bucket name where the results will be uploaded")
        print("Note: FOLDER_PREFIX is the prefix for the folder where the results are stored")
        print("Note: DS_PREFIX is the prefix for the dataset loader")
        sys.exit(1)

    ROOT_FOLDER = sys.argv[1]
    BUCKET_NAME = sys.argv[2]
    FOLDER_PREFIX = sys.argv[3]
    DS_PREFIX = sys.argv[4]
    traverse_folders(ROOT_FOLDER, BUCKET_NAME, FOLDER_PREFIX, DS_PREFIX)