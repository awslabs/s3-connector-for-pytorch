DATALOADER=$1
PATH_TO_STORE_DATASETS=$2
BUCKET_NAME=$3
REGION_NAME=$4
RESULTS_BUCKET_NAME=$5
RESULTS_REGION_NAME=$6
RESULTS_PREFIX=$7

datasets=("100k_496x387_images_4Mb_shards" "100k_496x387_images_8Mb_shards" "100k_496x387_images_16Mb_shards" "100k_496x387_images_32Mb_shards" "100k_496x387_images_64Mb_shards" "100k_496x387_images_128Mb_shards" "100k_496x387_images_256Mb_shards" "10k_496x387_images")

./utils/generate_datasets_files.sh "${PATH_TO_STORE_DATASETS}" "${BUCKET_NAME}" "${REGION_NAME}" "${datasets[@]}"
./utils/prepare_nvme.sh
rm -r -f ./multirun
./utils/run_dataloading_benchmarks.sh "${DATALOADER}" "${datasets[@]}"
python ./utils/upload_colated_results_to_s3.py "./multirun" "${RESULTS_BUCKET_NAME}" "${RESULTS_PREFIX}" "${DATALOADER}"
