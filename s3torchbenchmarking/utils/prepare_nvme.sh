sudo umount ./nvme/
sudo rm -rf ./nvme/
sudo mkfs -t xfs -f /dev/nvme1n1
sudo mkdir ./nvme/
sudo mount /dev/nvme1n1 ./nvme/
sudo chmod 777 ./nvme/
