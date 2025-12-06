import cv2
import copy
import json
import numpy as np
from PIL import Image
from pathlib import Path

from pycocotools.coco import COCO

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomHorizontalFlip

import detectron2.data.detection_utils as utils
from detectron2.structures import Boxes, Instances, pairwise_iou, pairwise_ioa


class CropAdder:

    def __init__(
        self,
        cfg,
        crop_img_dir,
        anno_file_path,
        target_like_crop_img_dir,
        num_crops,
        sample_class_method='inv',
        mixup_ratio=0.5,
        ioa_threshold=0.9,
        img_format='RGB'
    ):
        self.cfg = cfg
        if isinstance(crop_img_dir, str):
            crop_img_dir = [crop_img_dir]
        self.crop_img_dir = [Path(p) for p in crop_img_dir]

        self.anno_file_path = None if anno_file_path is None else Path(anno_file_path)

        if target_like_crop_img_dir is None:
            self.target_like_crop_img_dir = None
        else:
            if isinstance(target_like_crop_img_dir, str):
                target_like_crop_img_dir = [target_like_crop_img_dir]
            self.target_like_crop_img_dir = [Path(p) for p in target_like_crop_img_dir]
        
        self.num_crops = num_crops
        self.sample_class_method = sample_class_method
        self.mixup_ratio = mixup_ratio
        self.ioa_threshold = ioa_threshold
        self.img_format = img_format

        self.cat_name2idx = self.get_cat_name2idx()
        self.cat_idx2name = {v: k for k, v in self.cat_name2idx.items()}
        self.crop_meta_data = self.load_crop_meta_data()
        self.bbox_stats = self.analysize_bbox()
        self.class_stats = self.analysize_class()
        self.class_probs = self.cal_class_probs()
        
        self.random_flip = RandomHorizontalFlip()

    def get_cat_name2idx(self):
        if self.anno_file_path is not None:
            coco = COCO(self.anno_file_path)
            cat_name2id = {cat_dict['name']: cat_dict['id'] for cat_dict in coco.cats.values()}

            min_id = min(cat_name2id.values())
            cat_name2idx = {cat_name: cat_id - min_id for cat_name, cat_id in cat_name2id.items()}
        else:
            cat_names_list = []
            for crop_img_dir in self.crop_img_dir:
                cat_names = sorted(list(p.name for p in crop_img_dir.iterdir() if p.is_dir()))

                if len(cat_names_list) == 0:
                    cat_names_list.append(cat_names)
                else:
                    assert cat_names_list[0] == cat_names, 'There are different category names.'

            cat_names = cat_names_list[0]
            cat_name2idx = {n: i for i, n in enumerate(cat_names)}

        return cat_name2idx

    def load_crop_meta_data(self):
        crop_meta_data_dir = {}
        
        for crop_img_dir_path in self.crop_img_dir:
            crop_meta_data = self._load_crop_meta_data(crop_img_dir_path)
            crop_meta_data_dir[crop_img_dir_path] = crop_meta_data
                
        return crop_meta_data_dir

    def _load_crop_meta_data(self, crop_img_dir_path):
        meta_data_path = crop_img_dir_path / 'crop_meta_data.json'

        with meta_data_path.open() as f:
            crop_meta_data = json.load(f)

        print('=== Original Crop Meta Data ===')
        for cat_name, infos in crop_meta_data.items():
            print(f'# {self.cat_name2idx[cat_name]} {cat_name}: {len(infos)}')
        
        def is_not_small(crop_info):
            area = crop_info['crop_img_size'][0] * crop_info['crop_img_size'][1]
            return area > 32 ** 2

        for cat_name, infos in crop_meta_data.items():
            infos = list(filter(is_not_small, infos))
            crop_meta_data[cat_name] = infos

        print('=== Filtered Crop Meta Data ===')
        for cat_name, infos in crop_meta_data.items():
            print(f'# {self.cat_name2idx[cat_name]} {cat_name}: {len(infos)}')

        return crop_meta_data

    def analysize_bbox(self):
        bbox_stats = {
            cat_name: {
                'x1_range': (),
                'y1_range': (),
                'h_range': (),
                'r_range': ()
            }
            for cat_name in self.cat_name2idx
        }

        for crop_img_dir_path, crop_meta_data in self.crop_meta_data.items():
            for cat_name, infos in crop_meta_data.items():
                x1_min = min((info['location'][0] for info in infos))
                x1_max = max((info['location'][0] for info in infos))
                if bbox_stats[cat_name]['x1_range']:
                    x1_min = min(x1_min, bbox_stats[cat_name]['x1_range'][0])
                    x1_max = min(x1_max, bbox_stats[cat_name]['x1_range'][1])
                bbox_stats[cat_name]['x1_range'] = x1_min, x1_max
                
                y1_min = min((info['location'][1] for info in infos))
                y1_max = max((info['location'][1] for info in infos))
                if bbox_stats[cat_name]['y1_range']:
                    y1_min = min(y1_min, bbox_stats[cat_name]['y1_range'][0])
                    y1_max = min(x1_max, bbox_stats[cat_name]['y1_range'][1])
                bbox_stats[cat_name]['y1_range'] = y1_min, y1_max

                h_min = min((info['crop_img_size'][0] for info in infos))
                h_max = max((info['crop_img_size'][0] for info in infos))
                if bbox_stats[cat_name]['h_range']:
                    h_min = min(h_min, bbox_stats[cat_name]['h_range'][0])
                    h_max = min(h_max, bbox_stats[cat_name]['h_range'][1])
                bbox_stats[cat_name]['h_range'] = h_min, h_max

                r_min = min((info['crop_img_size'][1] / info['crop_img_size'][0] for info in infos))
                r_max = max((info['crop_img_size'][1] / info['crop_img_size'][0] for info in infos))
                if bbox_stats[cat_name]['r_range']:
                    r_min = min(r_min, bbox_stats[cat_name]['r_range'][0])
                    r_max = min(r_max, bbox_stats[cat_name]['r_range'][1])
                bbox_stats[cat_name]['r_range'] = r_min, r_max
        
        return bbox_stats

    def analysize_class(self):
        class_stats = {cat_name: 0 for cat_name in self.cat_name2idx}

        for crop_img_dir_path, crop_meta_data in self.crop_meta_data.items():
            for cat_name, infos in crop_meta_data.items():
                class_stats[cat_name] += len(infos)

        return class_stats

    def cal_class_probs(self):
        if self.sample_class_method == 'uniform':
            class_probs = np.ones(len(self.cat_idx2name))

        elif self.sample_class_method == 'inv':
            cat_counts = []
            for cat_idx in range(len(self.cat_idx2name)):
                cat_name = self.cat_idx2name[cat_idx]
                cat_counts.append(self.class_stats[cat_name])

            cat_counts = np.array(cat_counts)
            class_probs = 1 / cat_counts

        elif self.sample_class_method == 'sqrt_inv':
            cat_counts = []
            for cat_idx in range(len(self.cat_idx2name)):
                cat_name = self.cat_idx2name[cat_idx]
                cat_counts.append(self.class_stats[cat_name])

            cat_counts = np.array(cat_counts)
            class_probs = 1 / np.sqrt(cat_counts)
        
        else:
            raise ValueError(f'Unsupport sampling method {self.sample_class_method}')

        class_probs /= class_probs.sum()
        print('=== Class Probs ===')
        print(f'Sample class method: {self.sample_class_method}')
        for cat_idx, cat_prob in enumerate(class_probs):
            cat_name = self.cat_idx2name[cat_idx]
            print(f'{cat_name}: {cat_prob:.4f}')
        print('===================')
        print()

        return class_probs

    def sample_crop_info_and_path(self, sampled_class_name, sampled_crop_idx):
        for dir_idx, (crop_img_dir_path, crop_meta_data) in enumerate(self.crop_meta_data.items()):
            num_crops = len(crop_meta_data[sampled_class_name])

            if num_crops <= sampled_crop_idx:
                sampled_crop_idx -= num_crops
                continue
            else:
                sampled_crop_info = crop_meta_data[sampled_class_name][sampled_crop_idx]

        sampled_crop_img_path = crop_img_dir_path / sampled_class_name / sampled_crop_info['file_name']

        return sampled_crop_info, sampled_crop_img_path, dir_idx

    def sample_class_idx(self):
        return np.random.choice(len(self.class_probs), p=self.class_probs)

    def read_crop_img(self, crop_img_path):
        return utils.read_image(str(crop_img_path), format=self.img_format)

    def sample_crop(self, sampled_class_idx, tgt_img_size):
        while True:
            sampled_class_name = self.cat_idx2name[sampled_class_idx]
            num_crops = sum(len(crop_meta_data[sampled_class_name]) for crop_meta_data in self.crop_meta_data.values())
            
            sampled_crop_idx = np.random.choice(num_crops)
            sampled_crop_info, sampled_crop_img_path, dir_idx = self.sample_crop_info_and_path(sampled_class_name, sampled_crop_idx)
            crop_img = self.read_crop_img(sampled_crop_img_path)  # (h, w, c), np.ndarray

            ori_img_h, ori_img_w = sampled_crop_info['ori_img_size']
            tgt_img_h, tgt_img_w = tgt_img_size

            if ori_img_h > 0 and ori_img_w > 0:
                break

        img_resize_scale = min(tgt_img_h / ori_img_h, tgt_img_w / ori_img_w)
        
        target_like_crop_img = None
        if self.target_like_crop_img_dir is not None:
            target_like_crop_img_dir = self.target_like_crop_img_dir[dir_idx]
            target_like_crop_img_path = target_like_crop_img_dir / sampled_class_name / sampled_crop_info['file_name']
            target_like_crop_img = self.read_crop_img(target_like_crop_img_path)
            
        return crop_img, target_like_crop_img, img_resize_scale

    def sample_crop_loc_and_shape(self, sampled_class_idx, img_resize_scale):        
        sampled_class_name = self.cat_idx2name[sampled_class_idx]

        x1_min, x1_max = self.bbox_stats[sampled_class_name]['x1_range']
        sampled_x1 = int(np.random.uniform(x1_min, x1_max) * img_resize_scale)
        
        y1_min, y1_max = self.bbox_stats[sampled_class_name]['y1_range']
        sampled_y1 = int(np.random.uniform(y1_min, y1_max) * img_resize_scale)
        
        sampled_scale = np.random.uniform(0.8, 1.2)

        return sampled_x1, sampled_y1, sampled_scale

    def add_crops(self, img, target_like_img, gt_inst):
        # img: (h, w, c) np.ndarray
        # target_like_img: (h, w, c) np.ndarray
        # gt_inst: Instance obj or None
        
        img = copy.deepcopy(img)
        target_like_img = copy.deepcopy(target_like_img)
        img_size = img.shape[:2]
        
        added_inst = None
        for _ in range(self.num_crops):
            sampled_class_idx = self.sample_class_idx()
            crop_img, target_like_crop_img, img_resize_scale = self.sample_crop(sampled_class_idx, img_size)
            crop_h, crop_w = crop_img.shape[:2]

            added_boxes = None
            for _ in range(5):
                sampled_x1, sampled_y1, sampled_scale = self.sample_crop_loc_and_shape(sampled_class_idx, img_resize_scale)
                resize_crop_img = cv2.resize(crop_img, (0, 0), fx=sampled_scale, fy=sampled_scale, interpolation=cv2.INTER_CUBIC)
                resized_crop_h, resized_crop_w = resize_crop_img.shape[:2]
                
                sampled_x2 = sampled_x1 + resized_crop_w
                sampled_y2 = sampled_y1 + resized_crop_h

                img_h, img_w = img_size
                start_x, start_y, end_x, end_y = 0, 0, resized_crop_w, resized_crop_h
                if sampled_x2 > img_w:
                    end_x = resized_crop_w - (sampled_x2 - img_w)
                    sampled_x2 = img_w
                if sampled_y2 > img_h:
                    end_y = resized_crop_h - (sampled_y2 - img_h)
                    sampled_y2 = img_h
                if sampled_x2 <= 0 or sampled_y2 <= 0:
                    continue
                if end_x <= 0 or end_y <= 0:
                    continue

                added_boxes = Boxes(torch.tensor([[sampled_x1, sampled_y1, sampled_x2, sampled_y2]]))

                if gt_inst is None or len(gt_inst) == 0:
                    break
                
                ioa_matrix_1 = pairwise_ioa(added_boxes, gt_inst.gt_boxes)  # (1, #gt_boxes)
                ioa_matrix_2 = pairwise_ioa(gt_inst.gt_boxes, added_boxes)  # (#gt_boxes, 1)
                
                if not (ioa_matrix_1 > self.ioa_threshold).any() and not (ioa_matrix_2 > self.ioa_threshold).any():
                    break

            random_flip_flag = np.random.rand() < 0.5
            if random_flip_flag:
                resize_crop_img = resize_crop_img[:, ::-1, :]  # randomly flip

            img[sampled_y1:sampled_y2, sampled_x1:sampled_x2, :] = self.mixup_ratio * resize_crop_img[start_y:end_y, start_x:end_x, :] \
                                                                + (1 - self.mixup_ratio) * img[sampled_y1:sampled_y2, sampled_x1:sampled_x2, :]

            if target_like_img is not None:
                resized_target_like_crop_img = cv2.resize(target_like_crop_img, (0, 0), fx=sampled_scale, fy=sampled_scale, interpolation=cv2.INTER_CUBIC)
                if random_flip_flag:
                    resized_target_like_crop_img = resized_target_like_crop_img[:, ::-1, :]  # randomly flip

                target_like_img[sampled_y1:sampled_y2, sampled_x1:sampled_x2, :] = self.mixup_ratio * resized_target_like_crop_img[start_y:end_y, start_x:end_x, :] \
                                                                + (1 - self.mixup_ratio) * target_like_img[sampled_y1:sampled_y2, sampled_x1:sampled_x2, :]
            
            if added_boxes is None:
                new_added_inst = None
            else:
                new_added_inst = Instances(
                    img_size,
                    gt_boxes=added_boxes,
                    gt_classes=torch.tensor([sampled_class_idx]),
                    scores=torch.ones([1]).float()
                )
            if added_inst is None:
                added_inst = new_added_inst
            elif new_added_inst is not None:
                added_inst = Instances.cat([added_inst, new_added_inst])
            
            if gt_inst is None:
                gt_inst = copy.deepcopy(added_inst)
            elif added_inst is not None:
                gt_inst = Instances.cat([gt_inst, added_inst])

        return img, target_like_img, gt_inst, added_inst
