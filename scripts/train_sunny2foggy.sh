CUDA_VISIBLE_DEVICES=0,1 python train_net.py \
    --num-gpus 2 \
    --config configs/faster_rcnn_R101_cross_clipart.yaml \
    OUTPUT_DIR exps/sunny2foggy \
    DATASETS.TRAIN_LABEL "('daytime_sunny_train',)" \
    DATASETS.TRAIN_UNLABEL "('daytime_foggy_train',)" \
    DATASETS.TRAIN_TARGET_LIKE_LABEL "('target_like_sunny2foggy_train',)" \
    DATASETS.TEST "('daytime_sunny_test','daytime_foggy_test')" \
    DATASETS.ADD_CROPS.SRC.CROP_IMG_DIR "datasets/sdgod/daytime_sunny/VOC2007/crops" \
    DATASETS.ADD_CROPS.SRC.ANNO_FILE_PATH None \
    DATASETS.ADD_CROPS.SRC.TARGET_LIKE_CROP_IMG_DIR "datasets/target_like/sunny2foggy/VOC2007/crops" \
    DATASETS.ADD_CROPS.SRC.NUM_CROPS 5 \
    DATASETS.ADD_CROPS.TGT.CROP_IMG_DIR "datasets/target_like/sunny2foggy/VOC2007/crops" \
    DATASETS.ADD_CROPS.TGT.ANNO_FILE_PATH none \
    DATASETS.ADD_CROPS.TGT.NUM_CROPS 5 \
    MODEL.BACKBONE.NAME "build_resnet101_backbone" \
    MODEL.RESNETS.DEPTH 101 \
    MODEL.ROI_HEADS.NUM_CLASSES 20 \
    SEMISUPNET.Trainer "CropATeacherTrainer" \
    SEMISUPNET.BURN_UP_STEP 20000 \
    SEMISUPNET.CONTRASTIVE False \
    SOLVER.MAX_ITER 40000 \
    SOLVER.IMG_PER_BATCH_LABEL 16 \
    SOLVER.IMG_PER_BATCH_UNLABEL 16 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000