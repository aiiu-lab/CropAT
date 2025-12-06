import os
import cv2
import copy
import time
import logging
import itertools
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops import roi_align
from torchvision.transforms.functional import to_pil_image
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators, inference_on_dataset, print_csv_format

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.solver.build import maybe_add_gradient_clipping

from cropat.data.build import (
    build_cropat_detection_sup_train_loader,
    build_cropat_detection_semisup_train_loader_two_crops
)
from cropat.data.dataset_mapper import DatasetMapperTwoCropSeparate
from cropat.data.dataset_mapper_add_crops import (
    LabeledDataseWithInsertedCropstMapper,
    UnlabeledDatasetWithInsertedCropsMapper
)
from cropat.engine.hooks import LossEvalHook
from cropat.engine.trainer import ATeacherTrainer
from cropat.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from cropat.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from cropat.solver.build import build_lr_scheduler
from cropat.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator


class CropATeacherTrainer(ATeacherTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg, semisup=False):
        if not semisup:
            mapper = LabeledDataseWithInsertedCropstMapper(cfg, True)
            return build_cropat_detection_sup_train_loader(cfg, mapper)
        else:
            src_mapper = DatasetMapperTwoCropSeparate(cfg, True)
            tgt_mapper = UnlabeledDatasetWithInsertedCropsMapper(cfg, True)
            return build_cropat_detection_semisup_train_loader_two_crops(cfg, src_mapper, tgt_mapper)

    def merge_label(self, label_data):
        for label_datum in label_data:
            if 'added_instances' not in label_datum:
                continue
            
            gt_inst = label_datum['instances']
            added_inst = label_datum['added_instances']

            if added_inst is None:
                label_datum['instances'] = gt_inst
            else:
                label_datum['instances'] = Instances.cat([gt_inst, added_inst.to(gt_inst.gt_boxes.device)])

        return label_data

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"

        start = time.perf_counter()
        data_time = time.perf_counter() - start

        self.optimizer.zero_grad()
        
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            data = next(self._trainer._data_loader_iter)
            label_data_q, label_data_k = data

            if self.cfg.VIS_LABEL_PERIOD > 0 and self.iter % self.cfg.VIS_LABEL_PERIOD == 0:
                img_save_dir = Path(self.cfg.OUTPUT_DIR) / 'labeled_img'
                img_save_dir.mkdir(exist_ok=True, parents=True)
                
                for label_data, name in zip([label_data_k, label_data_q], ['k', 'q']):
                    img_rgb = label_data[0]['image'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

                    gt_inst = label_data[0]['instances']
                    gt_inst.pred_boxes = gt_inst.gt_boxes
                    gt_inst.pred_classes = gt_inst.gt_classes

                    v = Visualizer(
                        img_rgb,
                        metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                        instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                    )
                    out = v.draw_instance_predictions(gt_inst)
                    img_save_path = img_save_dir / f'{self.iter}_{name}.png'
                    cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])

                    if 'target_like_image' in label_data[0]:
                        img_rgb = label_data[0]['target_like_image'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        if self.cfg.INPUT.FORMAT == 'BGR':
                            img_rgb = img_rgb[:, :, ::-1]

                        gt_inst = label_data[0]['instances']
                        gt_inst.pred_boxes = gt_inst.gt_boxes
                        gt_inst.pred_classes = gt_inst.gt_classes

                        v = Visualizer(
                            img_rgb,
                            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                            instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                        )
                        out = v.draw_instance_predictions(gt_inst.to(torch.device('cpu')))
                        img_save_path = img_save_dir / f'{self.iter}_{name}_target_like.png'
                        cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])
                    
                    if 'blended_image' in label_data[0]:
                        img_rgb = label_data[0]['blended_image'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        if self.cfg.INPUT.FORMAT == 'BGR':
                            img_rgb = img_rgb[:, :, ::-1]

                        gt_inst = label_data[0]['instances']
                        gt_inst.pred_boxes = gt_inst.gt_boxes
                        gt_inst.pred_classes = gt_inst.gt_classes

                        v = Visualizer(
                            img_rgb,
                            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                            instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                        )
                        out = v.draw_instance_predictions(gt_inst.to(torch.device('cpu')))
                        img_save_path = img_save_dir / f'{self.iter}_{name}_blended.png'
                        cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])
                    
                    if 'blended_image_with_crops' in label_data[0]:
                        img_rgb = label_data[0]['blended_image_with_crops'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        if self.cfg.INPUT.FORMAT == 'BGR':
                            img_rgb = img_rgb[:, :, ::-1]

                        gt_inst = label_data[0]['instances']
                        gt_inst.pred_boxes = gt_inst.gt_boxes
                        gt_inst.pred_classes = gt_inst.gt_classes

                        v = Visualizer(
                            img_rgb,
                            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                            instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                        )
                        out = v.draw_instance_predictions(gt_inst.to(torch.device('cpu')))
                        img_save_path = img_save_dir / f'{self.iter}_{name}_blended_crops.png'
                        cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])

            label_data_q.extend(label_data_k)
            record_dict, _, _, _, features = self.model(label_data_q, branch="supervised")
        
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())
        
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                self._update_teacher_model(keep_rate=0.00)

                del self._trainer.data_loader
                self._trainer.data_loader = self.build_train_loader(self.cfg, semisup=True)
                self._trainer._data_loader_iter_obj = None

            elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            
            data = next(self._trainer._data_loader_iter)
            try:
                label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            except:
                del self._trainer.data_loader
                self._trainer.data_loader = self.build_train_loader(self.cfg, semisup=True)
                self._trainer._data_loader_iter_obj = None

                data = next(self._trainer._data_loader_iter)
                label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

            if self.cfg.VIS_LABEL_PERIOD > 0 and self.iter % self.cfg.VIS_LABEL_PERIOD == 0:
                img_save_dir = Path(self.cfg.OUTPUT_DIR) / 'labeled_img'
                img_save_dir.mkdir(exist_ok=True, parents=True)
                
                for label_data, name in zip([label_data_k, label_data_q], ['k', 'q']):
                    img_rgb = label_data[0]['image'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    if self.cfg.INPUT.FORMAT == 'BGR':
                        img_rgb = img_rgb[:, :, ::-1]

                    gt_inst = label_data[0]['instances']
                    gt_inst.pred_boxes = gt_inst.gt_boxes
                    gt_inst.pred_classes = gt_inst.gt_classes

                    v = Visualizer(
                        img_rgb,
                        metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                        instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                    )
                    out = v.draw_instance_predictions(gt_inst)
                    img_save_path = img_save_dir / f'{self.iter}_{name}.png'
                    cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])

                    if 'target_like_image' in label_data[0]:
                        img_rgb = label_data[0]['target_like_image'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        if self.cfg.INPUT.FORMAT == 'BGR':
                            img_rgb = img_rgb[:, :, ::-1]

                        gt_inst = label_data[0]['instances']
                        gt_inst.pred_boxes = gt_inst.gt_boxes
                        gt_inst.pred_classes = gt_inst.gt_classes

                        v = Visualizer(
                            img_rgb,
                            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                            instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                        )
                        out = v.draw_instance_predictions(gt_inst.to(torch.device('cpu')))
                        img_save_path = img_save_dir / f'{self.iter}_target_like.png'
                        cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])
                    
            #         if 'blended_image_with_crops' in label_data[0]:
            #             img_rgb = label_data[0]['blended_image_with_crops'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            #             if self.cfg.INPUT.FORMAT == 'BGR':
            #                 img_rgb = img_rgb[:, :, ::-1]

            #             gt_inst = label_data[0]['instances']
            #             gt_inst.pred_boxes = gt_inst.gt_boxes
            #             gt_inst.pred_classes = gt_inst.gt_classes

            #             v = Visualizer(
            #                 img_rgb,
            #                 metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
            #                 instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
            #             )
            #             out = v.draw_instance_predictions(gt_inst.to(torch.device('cpu')))
            #             img_save_path = img_save_dir / f'{self.iter}_blended.png'
            #             cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])

            record_dict = {}

            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            with torch.no_grad():
                # if self.cfg.SEMISUPNET.PSEUDO_BBOX_BRANCH == 'unsup_data_weak':
                _, proposals_rpn_unsup_k, proposals_roih_unsup_k, _, _ = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
                
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD  # 0.8

            joint_proposal_dict = {}
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")

            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            unlabel_data_k = self.add_label(unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])
            unlabel_data_q = self.add_label(unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])

            unlabel_data_k = self.merge_label(unlabel_data_k)
            unlabel_data_q = self.merge_label(unlabel_data_q)
            
            if self.cfg.VIS_LABEL_PERIOD > 0 and self.iter % self.cfg.VIS_LABEL_PERIOD == 0:
                img_save_dir = Path(self.cfg.OUTPUT_DIR) / 'unlabeled_img'
                img_save_dir.mkdir(exist_ok=True, parents=True)

                for unlabel_data, name in zip([unlabel_data_k, unlabel_data_q], ['k', 'q']):
                    img_rgb = unlabel_data[0]['image'].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    if self.cfg.INPUT.FORMAT == 'BGR':
                        img_rgb = img_rgb[:, :, ::-1]

                    gt_inst = unlabel_data[0]['instances']
                    gt_inst.pred_boxes = gt_inst.gt_boxes
                    gt_inst.pred_classes = gt_inst.gt_classes

                    v = Visualizer(
                        img_rgb,
                        metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                        instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                    )
                    out = v.draw_instance_predictions(gt_inst.to(torch.device('cpu')))
                    img_save_path = img_save_dir / f'{self.iter}_{name}_pseudo+added.png'
                    cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])

                    if 'added_instances' in unlabel_data[0] and unlabel_data[0]['added_instances'] is not None:
                        added_inst = unlabel_data[0]['added_instances']
                        added_inst.pred_boxes = added_inst.gt_boxes
                        added_inst.pred_classes = added_inst.gt_classes

                        v = Visualizer(
                            img_rgb,
                            metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN_LABEL[0]),
                            instance_mode=ColorMode.SEGMENTATION  # Let instances of the same category have similar colors
                        )
                        out = v.draw_instance_predictions(added_inst.to(torch.device('cpu')))
                        img_save_path = img_save_dir / f'{self.iter}_{name}_added.png'
                        cv2.imwrite(str(img_save_path), out.get_image()[:, :, ::-1])

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            record_all_label_data, _, _, _, features = self.model(all_label_data, branch="supervised")
            record_dict.update(record_all_label_data)

            record_all_unlabel_data, _, _, _, _ = self.model(all_unlabel_data, branch="supervised_target")
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)

            for i_index in range(len(unlabel_data_k)):
                for k, v in unlabel_data_k[i_index].items():
                    label_data_k[i_index][k + "_unlabeled"] = v

            all_domain_data = label_data_k
            record_all_domain_data, _, _, _, _ = self.model(all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)
            
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())
            
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        metrics_dict["features_max"] = features.max()
        metrics_dict['total_loss/nan'] = torch.zeros(1)

        if torch.isnan(losses):
            print('Loss NaN!!!')
            metrics_dict['total_loss/nan'] = torch.ones(1)
        else:
            losses.backward()
            self.optimizer.step()

        self._write_metrics(metrics_dict)
