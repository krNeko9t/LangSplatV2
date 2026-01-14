DATASET_ROOT_PATH=/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_fuse
DATASET_NAME=0a7cc12c0e_down
# DATASET_ROOT_PATH=./lerf_ovs
# DATASET_NAME=teatime
INDEX=0
TOPK=4

for level in 1 2 3
do
    CUDA_VISIBLE_DEVICES=3 python train.py \
        -s $DATASET_ROOT_PATH/$DATASET_NAME \
        -m output/${DATASET_NAME}_${INDEX} \
        --start_checkpoint $DATASET_ROOT_PATH/$DATASET_NAME/output/$DATASET_NAME/chkpnt30000.pth \
        --feature_level ${level} \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk $TOPK \
        --iterations 10000 \
        --train_subset_first_n 1
done


        # --start_checkpoint $DATASET_ROOT_PATH/$DATASET_NAME/output/$DATASET_NAME/chkpnt30000.pth \
        # --start_checkpoint output/teatime_0_${level}/chkpnt2000.pth \








