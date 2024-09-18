RESULTS_BUCKET_NAME=$1
RESULTS_PREFIX=$2

./utils/prepare_nvme.sh
rm -r -f ./multirun
s3torch-benchmark -cd conf -m -cn lightning_checkpointing
python ./utils/upload_colated_results_to_s3.py "./multirun" "${RESULTS_BUCKET_NAME}" "${RESULTS_PREFIX}" "checkpoint"
