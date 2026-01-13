# DATASET_ROOT_PATH=./scannetpp
DATASET_ROOT_PATH=/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse
DATASET_NAME=0a7cc12c0e
INDEX=0
TOPK=4

for level in 1 2 3
do
    python train.py \
        -s $DATASET_ROOT_PATH/$DATASET_NAME \
        -m output_pgsr/${DATASET_NAME}_${INDEX} \
        --start_ply $DATASET_ROOT_PATH/$DATASET_NAME/output/$DATASET_NAME/point_cloud/iteration_30000/point_cloud.ply \
        --feature_level ${level} \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk $TOPK \
        --iterations 100 \
        -r 2
done



