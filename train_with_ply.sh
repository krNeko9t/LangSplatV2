DATASET_ROOT_PATH=./scannetpp
DATASET_NAME=0a7cc12c0e
# DATASET_ROOT_PATH=./lerf_ovs
# DATASET_NAME=teatime
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
        --iterations 10000 \
        -r 2
done


        # --start_checkpoint $DATASET_ROOT_PATH/$DATASET_NAME/output/$DATASET_NAME/chkpnt30000.pth \
        # --start_checkpoint output/teatime_0_${level}/chkpnt2000.pth \








