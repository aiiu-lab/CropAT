# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from cropat.data.detection_utils import build_strong_augmentation

from .dataset_mapper import DatasetMapperTwoCropSeparate
from .crop_adder import CropAdder


class LabeledDataseWithInsertedCropstMapper(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg

        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        self.src_crop_adder = CropAdder(
            cfg=cfg,
            crop_img_dir=cfg.DATASETS.ADD_CROPS.SRC.CROP_IMG_DIR,
            anno_file_path=cfg.DATASETS.ADD_CROPS.SRC.ANNO_FILE_PATH,
            target_like_crop_img_dir=cfg.DATASETS.ADD_CROPS.SRC.TARGET_LIKE_CROP_IMG_DIR,
            num_crops=cfg.DATASETS.ADD_CROPS.SRC.NUM_CROPS,
            sample_class_method=cfg.DATASETS.ADD_CROPS.SRC.SAMPLE_CLASS_METHOD,
            mixup_ratio=cfg.DATASETS.ADD_CROPS.SRC.MIXUP_RATIO,
            ioa_threshold=cfg.DATASETS.ADD_CROPS.SRC.IOA_THRESHOLD,
            img_format='RGB'
        )

    def sample_mixup_ratio(self):
        return np.random.beta(0.5, 0.5, size=1).item()

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        image_weak_aug_with_crops = image_weak_aug.copy()

        if 'target_like_file_name' not in dataset_dict:
            raise RunTimeError('Need to load the target-like dataset')
        target_like_image = utils.read_image(dataset_dict['target_like_file_name'], format=self.img_format)
        target_like_image_weak_aug = transforms.apply_image(target_like_image)

        dataset_dict["target_like_image"] = torch.as_tensor(
            np.ascontiguousarray(target_like_image_weak_aug.transpose(2, 0, 1))
        )

        # insert crops into the source and target-like images
        gt_inst = bboxes_d2_format
        image_weak_aug_with_crops, target_like_image_weak_aug_with_crops, gt_inst, added_inst = self.src_crop_adder.add_crops(
            image_weak_aug.copy(),
            target_like_img=target_like_image_weak_aug.copy(),
            gt_inst=gt_inst
        )

        # mixup the source and target-like images
        mixup_ratio = self.sample_mixup_ratio()
        image_weak_aug_with_crops = (1 - mixup_ratio) * image_weak_aug_with_crops + mixup_ratio * target_like_image_weak_aug_with_crops
        image_weak_aug_with_crops = image_weak_aug_with_crops.astype("uint8")

        image_weak_aug_with_crops = Image.fromarray(image_weak_aug_with_crops, "RGB")
        image_strong_aug_with_crops = np.array(self.strong_augmentation(image_weak_aug_with_crops))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug_with_crops.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )

        dataset_dict['instances'] = gt_inst
        dataset_dict['added_instances'] = added_inst

        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)


class UnlabeledDatasetWithInsertedCropsMapper(DatasetMapper):

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg

        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT  # 'RGB'
        self.mask_on = cfg.MODEL.MASK_ON  # False
        self.mask_format = cfg.INPUT.MASK_FORMAT  # 'polygon'
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON  # False
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS  # False
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        self.tgt_crop_adder = CropAdder(
            cfg=cfg,
            crop_img_dir=cfg.DATASETS.ADD_CROPS.TGT.CROP_IMG_DIR,
            anno_file_path=cfg.DATASETS.ADD_CROPS.TGT.ANNO_FILE_PATH,
            target_like_crop_img_dir=None,
            num_crops=cfg.DATASETS.ADD_CROPS.TGT.NUM_CROPS,
            sample_class_method=cfg.DATASETS.ADD_CROPS.TGT.SAMPLE_CLASS_METHOD,
            mixup_ratio=cfg.DATASETS.ADD_CROPS.TGT.MIXUP_RATIO,
            ioa_threshold=cfg.DATASETS.ADD_CROPS.TGT.IOA_THRESHOLD,
            img_format='RGB'
        )

    def sample_mixup_ratio(self):
        return np.random.beta(0.5, 0.5, size=1).item()

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        image_weak_aug_with_crops = image_weak_aug.copy()

        # insert target-like crops into the target images
        image_weak_aug_with_crops, target_like_image_weak_aug_with_crops, gt_inst, added_inst = self.tgt_crop_adder.add_crops(
            image_weak_aug.copy(),
            target_like_img=None,
            gt_inst=bboxes_d2_format
        )
        image_weak_aug_with_crops = image_weak_aug_with_crops.astype("uint8")
        
        image_weak_aug_with_crops = Image.fromarray(image_weak_aug_with_crops, "RGB")
        image_strong_aug_with_crops = np.array(self.strong_augmentation(image_weak_aug_with_crops))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug_with_crops.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )

        dataset_dict['instances'] = gt_inst
        dataset_dict['added_instances'] = added_inst

        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)