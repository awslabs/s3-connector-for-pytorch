import boto3
import json
from collections import defaultdict
import csv
import sys
from enum import Enum


class GenerateForMode(Enum):
    DATALOAD = "dataload"
    CHECKPOINT = "checkpoint"


s3_client = boto3.client("s3")


def get_object_keys(bucket_name, prefix):
    """
    Get a list of all object keys (file names) in the S3 bucket.
    """
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    object_keys = [content["Key"] for content in response.get("Contents", [])]
    while response["IsTruncated"]:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name, ContinuationToken=response["NextContinuationToken"]
        )
        object_keys.extend([content["Key"] for content in response.get("Contents", [])])

    # filter out the files that are not json files
    object_keys = [key for key in object_keys if key.endswith(".json")]
    # filter out the files that are not in the root folder
    object_keys = [key for key in object_keys if prefix in key]
    print(object_keys)
    return object_keys


def read_json_from_s3(bucket_name, object_key):
    """
    Read the content of a JSON file from the S3 bucket.
    """
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    json_data = response["Body"].read().decode("utf-8")
    try:
        json_data = json.loads(json_data)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for {object_key}: {e}")
        return ""


# Function to extract relevant fields from a JSON file
def extract_fields_chekpoint(json_data):
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
                    "gpu_util_mean": result["utilization"]["gpu_util"]["mean"],
                    "gpu_mem_mean": result["utilization"]["gpu_mem"]["mean"],
                    "cpu_util_mean": result["utilization"]["cpu_util"]["mean"],
                    "cpu_mem_mean": result["utilization"]["cpu_mem"]["mean"],
                },
            }
        )
    return extracted_data


# Function to extract relevant fields from a JSON file
def extract_fields_dataloading(json_data):
    extracted_data = []
    for entry in json_data:
        cfg = entry["cfg"]
        result = entry["result"]

        prefix_uri_suffix = cfg["dataset"]["prefix_uri"].split("/")[-2]

        extracted_data.append(
            {
                "sharding": cfg["dataset"].get("sharding")
                if cfg["dataset"].get("sharding")
                else "None",
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
                    "gpu_util_mean": result["utilization"]["gpu_util"]["mean"],
                    "cpu_mem_mean": result["utilization"]["cpu_mem"]["mean"],
                    "gpu_mem_mean": result["utilization"]["gpu_mem"]["mean"],
                },
            }
        )
    return extracted_data


# Helper function to count rows for rowspan
def count_rows(data):
    if "result" in data:
        return 1
    return sum(
        count_rows(sub_data) for sub_data in data.values() if sub_data != "result"
    )


# Helper function to generate merged table cells
def generate_table_rows(data, indent=0, col_spans=None):
    if col_spans is None:
        col_spans = [1] * 7

    html = ""
    if "result" in data:
        html += "<tr>"
        for i in range(indent):
            html += "<td></td>"
        for key, value in data["result"].items():
            html += f"<td>{value}</td>"
        html += "</tr>"
    else:
        for key, value in data.items():
            if key == "result":
                continue
            span_count = count_rows(value)
            html += f"<tr><td rowspan='{span_count}'><strong>{key}</strong></td></tr>"
            for sub_key, sub_value in value.items():
                html += f"<tr><td rowspan='{count_rows(sub_value)}'>{sub_key}</td>"
                html += generate_table_rows(sub_value, indent + 2, col_spans)
                html += "</tr>"
    return html


# Function to recursively set nested dictionaries with explicit field names
def set_nested_dict(d, keys, value, field_names):
    current_level = d
    for key, name in zip(keys, field_names):
        if name not in current_level:
            current_level[name] = {}
        if key not in current_level[name]:
            current_level[name][key] = {}
        current_level = current_level[name][key]
    current_level["result"] = value


def load_data_from_s3(extract_fields_function, bucket_name, prefix):
    object_keys = get_object_keys(bucket_name, prefix)
    all_data = []

    for object_key in object_keys:
        json_data = read_json_from_s3(bucket_name, object_key)
        all_data.extend(extract_fields_function(json_data))
    return all_data


def save_data_to_csv(all_data, file_name):
    # Get the keys from the first dictionary in all_data
    fieldnames = list(all_data[0].keys())

    # Open the CSV file for writing
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write the header row
        writer.writeheader()

        # Write each dictionary as a row in the CSV file
        for data in all_data:
            writer.writerow(data)


def save_data_to_simple_json(all_data, file_name):
    with open(file_name, "w") as f:
        json.dump(all_data, f, indent=4)


def save_data_to_json(all_data, file_name, field_names):
    # Create a nested dictionary for the hierarchical structure
    nested_dict = defaultdict(dict)

    # Populate the nested dictionary with the extracted data
    for data in all_data:
        # create list of keys from data values for fields in field_names
        keys = [data[field] for field in field_names]
        set_nested_dict(nested_dict, keys, data["result"], field_names)

    # Convert defaultdict to dict for JSON serialization
    hierarchical_data = json.loads(json.dumps(nested_dict))

    # Save the hierarchical structure to a JSON file
    with open(file_name, "w") as f:
        json.dump(hierarchical_data, f, indent=4)


def save_data_to_html(
    all_data, file_name, sort_keys, field_names, result_fields, all_fields_captions
):
    # Generate the HTML content
    html_content = (
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Benchmark Results</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
    </head>
    <body>
        <h1>Benchmark Results</h1>
        <table>
            <thead>
                <tr>
    """
        + "".join([f"<th>{field}</th>" for field in all_fields_captions])
        + """
                </tr>
            </thead>
            <tbody>
    """
    )

    # Sort the data by the specified fields
    sorted_data = sorted(
        all_data,
        key=lambda x: ([x[field] for field in sort_keys]),
    )

    # initialise array of span indexes with 1 at start for every rows in sorted_data for first 6 columns
    column_count = len(field_names)
    span_indexes = [1] * len(sorted_data) * column_count

    # iterate over sorted_dat in reverse order
    for i in range(len(sorted_data) - 1, 0, -1):
        if sorted_data[i][field_names[0]] == sorted_data[i - 1][field_names[0]]:
            span_indexes[(i - 1) * column_count] = span_indexes[i * column_count] + 1
            span_indexes[i * column_count] = 0

    for i in range(len(sorted_data) - 1, 0, -1):
        for j in range(1, column_count, 1):
            if (
                sorted_data[i][field_names[j]] == sorted_data[i - 1][field_names[j]]
                and span_indexes[i * column_count + j - 1] == 0
            ):
                span_indexes[(i - 1) * column_count + j] = (
                    span_indexes[i * column_count + j] + 1
                )
                span_indexes[i * column_count + j] = 0

    def get_cell(all_data, index, field_names, field, span_indexes):
        field_index = field_names.index(field)
        if index < len(all_data):
            if span_indexes[index * len(field_names) + field_index] == 0:
                return ""
            if span_indexes[index * len(field_names) + field_index] == 1:
                return f"<td>{data[field]}</td>"
            return f'<td rowspan="{span_indexes[index * len(field_names) + field_index]}">{data[field]}</td>'
        return ""

    # Add rows for each benchmark result
    for i in range(len(sorted_data)):
        data = sorted_data[i]
        html_content += "<tr>"
        for field in sort_keys[:-1]:
            html_content += get_cell(sorted_data, i, field_names, field, span_indexes)
        html_content += f"<td>{data[sort_keys[-1]]}</td>"

        for field in result_fields:
            html_content += f"<td>{data['result'][field]}</td>"
        html_content += "</tr>"

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Save the HTML content to a file
    with open(file_name, "w") as f:
        f.write(html_content)


def get_dataloading_fields():
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


def get_checkpointing_fields():
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
        grouped_by_key_names = sort_key_names[:-1]
        data = load_data_from_s3(
            lambda json_data: extract_fields_dataloading(json_data), bucket_name, prefix
        )
    else:
        sort_key_names, result_fields, all_fields_captions = get_checkpointing_fields()
        grouped_by_key_names = sort_key_names[:-1]
        data = load_data_from_s3(
            lambda json_data: extract_fields_chekpoint(json_data), bucket_name, prefix
        )

    save_data_to_html(
        data,
        f"{file_name}.html",
        sort_key_names,
        grouped_by_key_names,
        result_fields,
        all_fields_captions,
    )
    save_data_to_simple_json(data, f"{file_name}.json")
