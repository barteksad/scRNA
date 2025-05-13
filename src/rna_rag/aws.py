import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET_NAME = "pmc-oa-opendata"
PMC_FOLDERS = [
    "oa_comm/{}/all/",
    "oa_noncomm/{}/all/",
    "author_manuscript/{}/all/",
    "phe_timebound/{}/all/",
]

def get_default_pubmed_s3_client():
    # Create S3 resource with unsigned configuration
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    return s3


def ls_folder(folder_prefix, s3_client=None, bucket_name: str = BUCKET_NAME) -> List[str]:
    """Remember to add '/' at the end of the folder_prefix."""
    if s3_client is None:
        s3_client = get_default_pubmed_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix, Delimiter="/")
    out = []
    if "CommonPrefixes" in response:
        print("Directories:")
        for prefix in response["CommonPrefixes"]:
            out.append(prefix["Prefix"])
    if "Contents" in response:
        for obj in response["Contents"]:
            out.append(obj["Key"])
    return out


def download_file(object_key, download_dir, s3_client=None, bucket_name: str = BUCKET_NAME):
    """
    Download a single file from an S3 bucket.
    """
    if s3_client is None:
        s3_client = get_default_pubmed_s3_client()

    local_file_path = os.path.join(download_dir, os.path.basename(object_key))
    s3_client.download_file(bucket_name, object_key, local_file_path)
    # print(f"Downloaded: {object_key} -> {local_file_path}")


def download_files_in_batch(
    file_keys: List[str], download_dir: str, max_threads=5, s3_client=None, bucket_name: str = BUCKET_NAME
):
    """
    Download multiple files from an S3 bucket in parallel using a thread pool.

    Args:
        file_keys (List[str]): List of S3 object keys to download.
        download_dir (str): Local directory to save the downloaded files.
        max_threads (int, optional): Maximum number of threads to use for downloading. Defaults to 5.
        s3_client (boto3.client, optional): Custom S3 client to use. If None, a default client is created. Defaults to None.
        bucket_name (str, optional): Name of the S3 bucket. Defaults to BUCKET_NAME.

    Returns:
        List[str]: List of file keys that failed to download.
    """
    if s3_client is None:
        s3_client = get_default_pubmed_s3_client()
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)  # Create download directory if it doesn't exist

    failed_files = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {
            executor.submit(download_file, file_key, download_dir,  s3_client, bucket_name): file_key
            for file_key in file_keys
        }

        for future in as_completed(futures):
            file_key = futures[future]
            try:
                future.result()  # Wait for the download to complete or raise an exception
            except botocore.exceptions.ClientError as e:
                print(f"Failed to download {file_key}: {e}")
                failed_files.append(file_key)
    return failed_files


def find_and_download_pmc_file(pmc_id: str, download_dir: str, file_format: str = "txt", s3_client=None, bucket_name: str = BUCKET_NAME) -> bool:
    """
    Check all possible folders for the given PMC ID and download the file if found.

    Args:
        pmc_id (str): The PMC ID to search for.
        download_dir (str): The local directory to save the downloaded file.
        file_format (str): The file format to search for (e.g., "txt" or "xml"). Defaults to "txt".
        s3_client (boto3.client, optional): Custom S3 client to use. Defaults to None.
        bucket_name (str, optional): S3 bucket name. Defaults to BUCKET_NAME.

    Returns:
        bool: True if the file was found and downloaded, False otherwise.
    """
    if s3_client is None:
        s3_client = get_default_pubmed_s3_client()

    for folder_template in PMC_FOLDERS:
        folder = folder_template.format(file_format)
        object_key = os.path.join(folder, f"{pmc_id}.{file_format}")
        try:
            # Check if the file exists
            s3_client.head_object(Bucket=bucket_name, Key=object_key)
            # File exists, download it
            download_file(object_key, download_dir, s3_client, bucket_name)
            return True  # File found and downloaded
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "404":
                print(f"Error accessing {object_key}: {e}")
            continue  # File not found, check next folder

    print(f"{pmc_id} not found in any known folder for format {file_format}.")
    return False  # File not found in any folder


def download_pmc_files_by_id_list(
    pmc_id_list: List[str], download_dir: str, file_format: str = "txt", max_threads=5, s3_client=None, bucket_name: str = BUCKET_NAME
) -> List[Optional[str]]:
    """
    Download files corresponding to a list of PMC IDs from possible locations in parallel.

    Args:
        pmc_id_list (List[str]): List of PMC IDs to download.
        download_dir (str): Local directory to save the downloaded files.
        file_format (str): The file format to search for (e.g., "txt" or "xml"). Defaults to "txt".
        max_threads (int, optional): Maximum number of threads to use for downloading. Defaults to 5.
        s3_client (boto3.client, optional): Custom S3 client to use. Defaults to None.
        bucket_name (str, optional): S3 bucket name. Defaults to BUCKET_NAME.

    Returns:
        List[Optional[str]]: List of filenames if download was successful, or None if not.
    """
    if s3_client is None:
        s3_client = get_default_pubmed_s3_client()
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)  # Create download directory if it doesn't exist

    result = [None] * len(pmc_id_list)  # Initialize result list with None

    with ThreadPoolExecutor(max_threads) as executor, tqdm(total=len(pmc_id_list), desc="Downloading") as pbar:
        futures = {
            executor.submit(find_and_download_pmc_file, pmc_id, download_dir, file_format, s3_client, bucket_name): idx
            for idx, pmc_id in enumerate(pmc_id_list)
        }

        for future in as_completed(futures):
            idx = futures[future]
            pmc_id = pmc_id_list[idx]
            try:
                if future.result():  # If True, download was successful
                    result[idx] = os.path.join(download_dir, f"{pmc_id}.{file_format}")
            except botocore.exceptions.ClientError as e:
                print(f"S3 ClientError while downloading {pmc_id}: {e}")
            except botocore.exceptions.BotoCoreError as e:
                print(f"General BotoCoreError while downloading {pmc_id}: {e}")
            except OSError as e:
                print(f"File system error while saving {pmc_id}: {e}")
            finally:
                pbar.update(1)  # Update progress bar by 1 step

    assert len(pmc_id_list) == len(result)
    return result
