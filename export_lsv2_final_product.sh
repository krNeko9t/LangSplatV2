CUDA_VISIBLE_DEVICES=6 python export_lsv2_final_product.py \
  --scene_id figurines \
  --ckpt_root /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/lerf_ovs \
  --ckpt_prefix figurines_0 \
  --checkpoint 10000 \
  --topk 4 \
  --output_dir /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/LangSplatV2/export_weight_book \
  --levels 1 2 3 \
  --queries elephant camera

# 输出 PLY 文件名已更新为：gaussians_with_weights64.ply（包含完整 3DGS 参数 + weight_0..63 额外属性）