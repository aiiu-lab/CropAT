# This verson fixes the bug that a crop may includes other object while no annotation for it

import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET


def main(args):
    voc_dir = Path(args.voc_dir)
    img_dir = voc_dir / 'JPEGImages'
    ann_dir = Path(args.ann_dir)
    crop_save_dir = voc_dir / 'crops'
    crop_save_dir.mkdir(exist_ok=True)

    with open(args.img_list_file, 'r') as f:
        img_names = f.readlines()
        img_names = [n.replace('\n', '') for n in img_names]
    
    crop_meta_data = {
        # 'person': [],
        # 'rider': [],
        # 'car': [],
        # 'truck': [],
        # 'bus': [],
        # 'motorcycle': [],
        # 'bicycle': []
    }

    for img_name in tqdm(img_names, desc='crop images...'):
        img_path = img_dir / f'{img_name}.jpg'
        img_array = np.array(Image.open(str(img_path)))

        ann_path = ann_dir / f'{img_name}.xml'
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            cat_name = obj.find('name').text
            cat_save_dir = crop_save_dir / cat_name
            cat_save_dir.mkdir(exist_ok=True)

            if cat_name not in crop_meta_data:
                crop_meta_data[cat_name] = []

            x1 = int(obj.find('bndbox').find('xmin').text)
            x2 = int(obj.find('bndbox').find('xmax').text)
            y1 = int(obj.find('bndbox').find('ymin').text)
            y2 = int(obj.find('bndbox').find('ymax').text)

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            crop_img = img_array[y1:y2, x1:x2, :]
            crop_img = Image.fromarray(crop_img)

            crop_file_name = f'{len(crop_meta_data[cat_name])}.jpg'
            crop_img_save_path = cat_save_dir / crop_file_name
            crop_img.save(str(crop_img_save_path))

            crop_meta_info = {
                'file_name': crop_file_name,
                'ori_img_size': img_array.shape[:2],
                'crop_img_size': (h, w),
                'location': (x1, y1, x2, y2)
            }
            crop_meta_data[cat_name].append(crop_meta_info)
    
    crop_meta_data_save_path = crop_save_dir / 'crop_meta_data.json'
    json_object = json.dumps(crop_meta_data, indent=4)
 
    with open(str(crop_meta_data_save_path), 'w') as f:
        f.write(json_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-dir', type=str, default='./sdgod/daytime_sunny/VOC2007')
    parser.add_argument('--ann-dir', type=str, default='./sdgod/daytime_sunny/VOC2007/Annotations')
    parser.add_argument('--img-list-file', type=str, default='./sdgod/daytime_sunny/VOC2007/ImageSets/Main/train.txt')
    args = parser.parse_args()

    main(args)
