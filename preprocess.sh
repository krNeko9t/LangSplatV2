# python preprocess.py --dataset_path /mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse/0a7cc12c0e

CUDA_VISIBLE_DEVICES=6 python preprocess.py \
  --dataset_path /mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse/0a7cc12c0e_down \
  --resolution 2 \
  --max_images 2 \
#   --sam_autocast bf16 \