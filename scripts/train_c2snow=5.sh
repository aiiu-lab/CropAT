CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
    --num-gpus 4 \
    --config configs/faster_rcnn_VGG_cross_city.yaml \
    OUTPUT_DIR exps/c2snow=5 \
    DATASETS.TRAIN_UNLABEL "('cityscapes_c_snow=5_train',)" \
    DATASETS.TEST "('cityscapes_fine_instance_seg_val','cityscapes_c_snow=5_val')" \
    DATASETS.TRAIN_TARGET_LIKE_LABEL "('target_like_cityscapes_c_snow=5_train',)" \
    DATASETS.ADD_CROPS.SRC.CROP_IMG_DIR "datasets/cityscapes/leftImg8bit/crops" \
    DATASETS.ADD_CROPS.SRC.ANNO_FILE_PATH "datasets/cityscapes/annotations/cityscapes_train.json" \
    DATASETS.ADD_CROPS.SRC.TARGET_LIKE_CROP_IMG_DIR "datasets/target_like/c2frost=5/leftImg8bit/crops" \
    DATASETS.ADD_CROPS.SRC.NUM_CROPS 5 \
    DATASETS.ADD_CROPS.TGT.CROP_IMG_DIR "datasets/target_like/c2frost=5/leftImg8bit/crops" \
    DATASETS.ADD_CROPS.TGT.ANNO_FILE_PATH "datasets/cityscapes/annotations/cityscapes_train.json" \
    DATASETS.ADD_CROPS.TGT.NUM_CROPS 5 \
    SEMISUPNET.Trainer "CropATeacherTrainer" \
    SEMISUPNET.BURN_UP_STEP 20000 \
    SEMISUPNET.CONTRASTIVE False \
    SOLVER.MAX_ITER 40000 \
    SOLVER.IMG_PER_BATCH_LABEL 16 \
    SOLVER.IMG_PER_BATCH_UNLABEL 16 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000