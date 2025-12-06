import os
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision.datasets.vision import VisionDataset

from einops import rearrange


def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'cityscapes_foggy': {
            'train_img': root / 'cityscapes_foggy/leftImg8bit/train',
            'train_anno': root / 'cityscapes_foggy/annotations/foggy_cityscapes_train.json',
            'val_img': root / 'cityscapes_foggy/leftImg8bit/val',
            'val_anno': root / 'cityscapes_foggy/annotations/foggy_cityscapes_val.json',
        },
        'corrputed_cityscapes_bright=5': {
            'train_img': root / 'cityscapes_c/brightness-5/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes_c/brightness-5/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'corrputed_cityscapes_fog=5': {
            'train_img': root / 'cityscapes_c/fog-5/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes_c/fog-5/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'corrputed_cityscapes_frost=5': {
            'train_img': root / 'cityscapes_c/frost-5/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes_c/frsot-5/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'corrputed_cityscapes_snow=5': {
            'train_img': root / 'cityscapes_c/snow-5/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes_c/snow-5/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'voc2007': {
            'img_dir': root / 'VOC2007/JPEGImages',
            'train_imgs_file': root / 'VOC2007/ImageSets/Main/trainval.txt',
        },
        'voc2012': {
            'img_dir': root / 'VOC2012/JPEGImages',
            'train_imgs_file': root / 'VOC2012/ImageSets/Main/trainval.txt',
        },
        'clipart': {
            'img_dir': root / 'clipart/JPEGImages',
            'train_imgs_file': root / 'clipart/ImageSets/Main/train.txt'
        },
        'sdgod_daytime_sunny': {
            'img_dir': root / 'sdgod/daytime_sunny/VOC2007/JPEGImages',
            'train_imgs_file': root / 'sdgod/daytime_sunny/VOC2007/ImageSets/Main/train.txt',
        },
        'sdgod_daytime_foggy': {
            'img_dir': root / 'sdgod/daytime_foggy/VOC2007/JPEGImages',
            'train_imgs_file': root / 'sdgod/daytime_foggy/VOC2007/ImageSets/Main/train.txt',
        },
        'sdgod_night_sunny': {
            'img_dir': root / 'sdgod/night_sunny/VOC2007/JPEGImages',
            'train_imgs_file': root / 'sdgod/night_sunny/VOC2007/ImageSets/Main/train.txt',
        },
        'sdgod_night_rainy': {
            'img_dir': root / 'sdgod/night_rainy/VOC2007/JPEGImages',
            'train_imgs_file': root / 'sdgod/night_rainy/VOC2007/ImageSets/Main/train.txt',
        },
        'sdgod_dusk_rainy': {
            'img_dir': root / 'sdgod/dusk_rainy/VOC2007/JPEGImages',
            'train_imgs_file': root / 'sdgod/dusk_rainy/VOC2007/ImageSets/Main/train.txt',
        },
    }


class BaseDataset(Dataset):

    def __init__(self):
        pass

    def get_image(self, path):
        return Image.open(path).convert('RGB')
    
    def resize(self, img):
        return ImageOps.fit(img, (self.resolution, self.resolution), method=Image.Resampling.LANCZOS)

    def normalize(self, img):
        return 2 * torch.tensor(np.array(img)).float() / 255 - 1

    def rearrange(self, img):
        if img.ndim == 3:
            img = rearrange(img, "h w c -> c h w")

        elif img.ndim == 4:
            img = rearrange(img, "t h w c -> t c h w")

        return img

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


class CocoDataset(BaseDataset):

    def __init__(self, img_folder, ann_file, resolution):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.resolution = resolution

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    # def resize(self, img):
        # width, height = img.size
        # factor = self.resolution / max(width, height)
        # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        # width = int((width * factor) // 64) * 64
        # height = int((height * factor) // 64) * 64
        # img = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)

        # return img

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        path = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(self.img_folder, path)
        img = self.get_image(path)
        ori_size = img.size

        img = self.resize(img)
        img = self.normalize(img)
        img = self.rearrange(img)

        return img, path, ori_size


class DADataset(Dataset):
    def __init__(
        self,
        src_img_folder,
        src_ann_file,
        tgt_img_folder,
        tgt_ann_file,
        resolution
    ):

        # for source data
        self.source = CocoDataset(
            img_folder=src_img_folder,
            ann_file=src_ann_file,
            resolution=resolution
        )

        # for target data
        self.target = CocoDataset(
            img_folder=tgt_img_folder,
            ann_file=tgt_ann_file,
            resolution=resolution
        )

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, idx):
        source_img, source_img_path, _ = self.source[idx % len(self.source)]
        target_img, target_img_path, _ = self.target[idx % len(self.target)]

        return source_img, target_img, source_img_path, target_img_path


class VocDataset(BaseDataset):

    def __init__(
        self,
        img_folder,
        train_imgs_file,
        resolution
    ):
        self.img_folder = img_folder
        self.train_imgs_file = train_imgs_file
        self.resolution = resolution

        self.img_paths = self.read_imgs_file(img_folder, train_imgs_file)

    def read_imgs_file(self, img_folder, imgs_file):
        with open(imgs_file, 'r') as f:
            img_names = f.readlines()
        img_paths = [os.path.join(img_folder, n.replace('\n', '.jpg')) for n in img_names]

        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = self.get_image(img_path)
        ori_size = img.size
        img = self.resize(img)
        img = self.normalize(img)
        img = self.rearrange(img)

        return img, img_path, ori_size


class VocDADataset(BaseDataset):

    def __init__(
        self,
        src_img_folder_2007,
        src_train_imgs_file_2007,
        src_img_folder_2012,
        src_train_imgs_file_2012,
        tgt_img_folder,
        tgt_train_imgs_file,
        resolution
    ):
        self.src_img_folder_2007 = src_img_folder_2007
        self.src_train_imgs_file_2007 = src_train_imgs_file_2007
        self.src_img_folder_2012 = src_img_folder_2012
        self.src_train_imgs_file_2012 = src_train_imgs_file_2012
        self.tgt_img_folder = tgt_img_folder
        self.tgt_train_imgs_file = tgt_train_imgs_file
        self.resolution = resolution

        src_img_paths_2007 = self.read_imgs_file(src_img_folder_2007, src_train_imgs_file_2007)
        src_img_paths_2012 = self.read_imgs_file(src_img_folder_2012, src_train_imgs_file_2012) if src_img_folder_2012 else []
        self.src_img_paths = src_img_paths_2007 + src_img_paths_2012

        self.tgt_img_paths = self.read_imgs_file(tgt_img_folder, tgt_train_imgs_file)

    def read_imgs_file(self, img_folder, imgs_file):
        with open(imgs_file, 'r') as f:
            img_names = f.readlines()
        img_paths = [os.path.join(img_folder, n.replace('\n', '.jpg')) for n in img_names]

        return img_paths

    def __len__(self):
        return max(len(self.src_img_paths), len(self.tgt_img_paths))

    def __getitem__(self, idx):
        src_img_path = self.src_img_paths[idx % len(self.src_img_paths)]
        tgt_img_path = self.tgt_img_paths[idx % len(self.tgt_img_paths)]

        src_img = self.get_image(src_img_path)
        src_img = self.resize(src_img)
        src_img = self.normalize(src_img)
        src_img = self.rearrange(src_img)

        tgt_img = self.get_image(tgt_img_path)
        tgt_img = self.resize(tgt_img)
        tgt_img = self.normalize(tgt_img)
        tgt_img = self.rearrange(tgt_img)

        return src_img, tgt_img, src_img_path, tgt_img_path
        

def build_dataset(source_domain, target_domain, image_set, resolution, paired=False, do_resize=True, split_img=False):
    assert source_domain in ['cityscapes', 'voc2007+voc2012', 'daytime_sunny']

    paths = get_paths(root='../datasets')

    if source_domain == 'cityscapes':
        if target_domain is None:
            dataset = CocoDataset(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                resolution=resolution
            )

        else:
            dataset = DADataset(
                src_img_folder=paths[source_domain]['train_img'],
                src_ann_file=paths[source_domain]['train_anno'],
                tgt_img_folder=paths[target_domain]['train_img'],
                tgt_ann_file=paths[target_domain]['train_anno'],
                resolution=resolution
            )

        return dataset

    if source_domain == 'voc2007+voc2012':
        assert target_domain == 'clipart'

        dataset = VocDADataset(
            src_img_folder_2007=paths['voc2007']['img_dir'],
            src_train_imgs_file_2007=paths['voc2007']['img_dir'],
            src_img_folder_2012=paths['voc2012']['img_dir'],
            src_train_imgs_file_2012=paths['voc2012']['img_dir'],
            tgt_img_folder=paths[target_domain]['img_dir'],
            tgt_train_imgs_file=paths[target_domain]['train_imgs_file'],
            resolution=resolution
        )

        return dataset

    if source_domain in ['voc2007', 'voc2012']:
        assert target_domain is None

        dataset = VocDataset(
            img_folder=paths[source_domain]['img_dir'],
            train_imgs_file=paths[source_domain]['train_imgs_file'],
            resolution=resolution
        )

        return dataset

    if source_domain == 'sdgod_daytime_sunny':
        assert target_domain in ['sdgod_daytime_foggy', 'sdgod_night_sunny', 'sdgod_night_rainy', 'sdgod_dusk_rainy']

        dataset = VocDADataset(
            src_img_folder_2007=paths[source_domain]['img_dir'],
            src_train_imgs_file_2007=paths[source_domain]['train_imgs_file'],
            src_img_folder_2012=None,
            src_train_imgs_file_2012=None,
            tgt_img_folder=paths[target_domain]['img_dir'],
            tgt_train_imgs_file=paths[target_domain]['train_imgs_file'],
            resolution=resolution
        )

        return dataset
