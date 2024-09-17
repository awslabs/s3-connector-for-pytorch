import sys
from enum import Enum
from typing import List, Dict, Any
import json
import boto3

from html_result_generator import HtmlResultGenerator

# Using type aliases for better code readability
JsonData = List[Dict[str, Any]]
ExtractedData = List[Dict[str, Any]]


class GenerateForMode(Enum):
    """
    Enum class to represent the mode for data generation.
    """

    DATALOAD = "dataload"
    CHECKPOINT = "checkpoint"


s3_client = boto3.client("s3")


# Get a list of all object keys (file names) in the S3 bucket.
def get_object_keys(bucket_name: str, prefix: str) -> List[str]:
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    object_keys = [content["Key"] for content in response.get("Contents", [])]
    while response.get("IsTruncated", False):
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            ContinuationToken=response["NextContinuationToken"],
        )
        object_keys.extend([content["Key"] for content in response.get("Contents", [])])

    return [key for key in object_keys if key.endswith(".json")]


# Read the content of a JSON file from the S3 bucket.
def read_json_from_s3(bucket_name: str, object_key: str) -> JsonData:
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        json_data = response["Body"].read().decode("utf-8")
        return json.loads(json_data)
    except (json.JSONDecodeError, s3_client.exceptions.ClientError) as exc:
        print(f"Error reading JSON for {object_key}: {exc}")
        return []


# Extract relevant fields from a JSON file for checkpointing.
def extract_fields_checkpoint(json_data: JsonData) -> ExtractedData:
    extracted_data = []
    for entry in json_data:
        cfg = entry["cfg"]
        result = entry["result"]

        extracted_data.append(
            {
                "kind": cfg["training"]["kind"],
                "model": cfg["training"]["model"],
                "max_epochs": cfg["training"]["max_epochs"],
                "save_one_in": cfg["checkpoint"]["save_one_in"],
                "destination": cfg["checkpoint"]["destination"],
                "uri": cfg["checkpoint"]["uri"],
                "region": cfg["checkpoint"]["region"],
                "result": {
                    "model_size": result["model_size"],
                    "mean_time": result["mean_time"],
                    "throughput": result["throughput"]["mean"],
                    "utilization": result["utilization"],
                    "gpu_util_mean": (
                        result["utilization"]["gpu_util"]["mean"]
                        if "gpu_util" in result["utilization"]
                        else ""
                    ),
                    "gpu_mem_mean": (
                        result["utilization"]["gpu_mem"]["mean"]
                        if "gpu_mem" in result["utilization"]
                        else ""
                    ),
                    "cpu_util_mean": result["utilization"]["cpu_util"]["mean"],
                    "cpu_mem_mean": result["utilization"]["cpu_mem"]["mean"],
                },
            }
        )
    return extracted_data


# Extract relevant fields from a JSON file for data loading.
def extract_fields_dataloading(json_data: JsonData) -> ExtractedData:
    extracted_data = []
    for entry in json_data:
        cfg = entry["cfg"]
        result = entry["result"]

        prefix_uri_suffix = cfg["dataset"]["prefix_uri"].split("/")[-2]

        extracted_data.append(
            {
                "sharding": (
                    cfg["dataset"].get("sharding")
                    if cfg["dataset"].get("sharding")
                    else "None"
                ),
                "model": cfg["training"]["model"],
                "max_epochs": cfg["training"]["max_epochs"],
                "prefix_uri_suffix": prefix_uri_suffix,
                "batch_size": cfg["dataloader"]["batch_size"],
                "num_workers": cfg["dataloader"]["num_workers"],
                "dataloader_kind": cfg["dataloader"]["kind"],
                "result": {
                    "elapsed_time": result["elapsed_time"],
                    "throughput": result["throughput"],
                    "utilization": result["utilization"],
                    "cpu_util_mean": result["utilization"]["cpu_util"]["mean"],
                    "gpu_util_mean": (
                        result["utilization"]["gpu_util"]["mean"]
                        if "gpu_util" in result["utilization"]
                        else ""
                    ),
                    "cpu_mem_mean": result["utilization"]["cpu_mem"]["mean"],
                    "gpu_mem_mean": (
                        result["utilization"]["gpu_mem"]["mean"]
                        if "gpu_mem" in result["utilization"]
                        else ""
                    ),
                },
            }
        )
    return extracted_data


# Function to read all json files from S3
def load_data_from_s3(
    extract_fields_function: callable,
    bucket_name: str,
    prefix: str,
) -> ExtractedData:
    object_keys = get_object_keys(bucket_name, prefix)
    all_data = []

    for object_key in object_keys:
        json_data = read_json_from_s3(bucket_name, object_key)
        all_data.extend(extract_fields_function(json_data))
    return all_data


# Helper function to save results to json
def save_data_to_simple_json(all_data: ExtractedData, file_name: str) -> None:
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(all_data, file, indent=4)


# Get field names and captions for data loading.
def get_dataloading_fields() -> tuple[List[str], List[str], List[str]]:
    sort_key_names = [
        "sharding",
        "model",
        "max_epochs",
        "prefix_uri_suffix",
        "batch_size",
        "num_workers",
        "dataloader_kind",
    ]
    result_fields = [
        "elapsed_time",
        "throughput",
        "cpu_util_mean",
        "gpu_util_mean",
        "cpu_mem_mean",
        "gpu_mem_mean",
    ]
    all_fields_captions = [
        "Sharding",
        "Model",
        "Max Epochs",
        "Dataset",
        "Batch Size",
        "Num Workers",
        "Dataloader Kind",
        "Elapsed Time",
        "Throughput",
        "CPU Util Mean",
        "GPU Util Mean",
        "CPU Mem Mean",
        "GPU Mem Mean",
    ]
    return sort_key_names, result_fields, all_fields_captions


# Get field names and captions for checkpointing.
def get_checkpointing_fields() -> tuple[List[str], List[str], List[str]]:
    sort_key_names = ["kind", "model", "max_epochs", "save_one_in", "destination"]
    result_fields = [
        "model_size",
        "mean_time",
        "throughput",
        "cpu_util_mean",
        "gpu_util_mean",
        "cpu_mem_mean",
        "gpu_mem_mean",
    ]
    all_fields_captions = [
        "Kind",
        "Model",
        "Max Epochs",
        "Save one in",
        "Destination",
        "Model size",
        "Elapsed Time",
        "Throughput",
        "CPU Util Mean",
        "GPU Util Mean",
        "CPU Mem Mean",
        "GPU Mem Mean",
    ]
    return sort_key_names, result_fields, all_fields_captions


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python download_and_transform_results.py <GENERATE_FOR_MODE> <FILE_NAME> <BUCKET_NAME> "
            "<PREFIX>"
        )
        sys.exit(1)

    generate_for_mode_str = sys.argv[1]
    try:
        generate_for_mode = GenerateForMode(generate_for_mode_str)
    except ValueError:
        print(f"Invalid value for GENERATE_FOR_MODE: {generate_for_mode_str}")
        print("Allowed values: dataload, checkpoint")
        sys.exit(1)
    file_name = sys.argv[2]
    bucket_name = sys.argv[3]
    prefix = sys.argv[4]

    if generate_for_mode == GenerateForMode.DATALOAD:
        sort_key_names, result_fields, all_fields_captions = get_dataloading_fields()
        # The last field in sort_key_names is excluded from grouped_by_key_names because
        # we want to preserve the granularity of results at that level.
        grouped_by_key_names = sort_key_names[:-1]
        records = load_data_from_s3(
            lambda json_data: extract_fields_dataloading(json_data), bucket_name, prefix
        )
    else:
        sort_key_names, result_fields, all_fields_captions = get_checkpointing_fields()
        grouped_by_key_names = sort_key_names[:-1]
        records = load_data_from_s3(
            lambda json_data: extract_fields_checkpoint(json_data), bucket_name, prefix
        )

    html_generator = HtmlResultGenerator()
    html = html_generator.generate_html(
        records,
        sort_key_names,
        grouped_by_key_names,
        result_fields,
        all_fields_captions,
    )
    html_generator.save_to_file(html, f"{file_name}.html")

    save_data_to_simple_json(records, f"{file_name}.json")
