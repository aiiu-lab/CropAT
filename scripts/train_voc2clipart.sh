CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --config configs/faster_rcnn_R101_cross_clipart_b4.yaml \
    OUTPUT_DIR exps/voc2clipart \
    DATASETS.TRAIN_LABEL "('voc_2007_trainval','voc_2012_trainval')" \
    DATASETS.TRAIN_TARGET_LIKE_LABEL "('target_like_voc2007_to_clipart_trainval','target_like_voc2012_to_clipart_trainval')" \
    DATASETS.TRAIN_UNLABEL "('Clipart1k_train',)" \
    DATASETS.TEST "('voc_2007_val','voc_2012_val','Clipart1k_test')" \
    DATASETS.ADD_CROPS.SRC.CROP_IMG_DIR "('datasets/VOC2007/crops', 'datasets/VOC2012/crops')" \
    DATASETS.ADD_CROPS.SRC.ANNO_FILE_PATH None \
    DATASETS.ADD_CROPS.SRC.TARGET_LIKE_CROP_IMG_DIR "('datasets/target_like/voc2007_to_clipart/crops', 'datasets/target_like/voc2012_to_clipart/crops')" \
    DATASETS.ADD_CROPS.TGT.CROP_IMG_DIR "('datasets/target_like/voc2007_to_clipart/crops', 'datasets/target_like/voc2012_to_clipart/crops')" \
    DATASETS.ADD_CROPS.TGT.ANNO_FILE_PATH None \
    DATASETS.ADD_CROPS.TGT.NUM_CROPS 5 \
    MODEL.BACKBONE.NAME "build_resnet101_backbone" \
    MODEL.RESNETS.DEPTH 101 \
    MODEL.ROI_HEADS.NUM_CLASSES 20 \
    SEMISUPNET.Trainer "CropATeacherTrainer" \
    SEMISUPNET.BURN_UP_STEP 60000 \
    SEMISUPNET.CONTRASTIVE False \
    SOLVER.MAX_ITER 100000 \
    SOLVER.IMG_PER_BATCH_LABEL 16 \
    SOLVER.IMG_PER_BATCH_UNLABEL 16 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000