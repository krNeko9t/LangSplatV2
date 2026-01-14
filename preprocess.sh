# python preprocess.py --dataset_path /mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse/0a7cc12c0e

CUDA_VISIBLE_DEVICES=3 python preprocess.py \
  --dataset_path /mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse/0a7cc12c0e_down \
  --resolution 512 \
  --sam_points_per_side 24 \
  # --max_images 10 \
#   --sam_autocast bf16 \