import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from pycocotools.coco import COCO


def main(args):
    img_dir = Path(args.img_dir)
    ann_path = Path(args.ann_path)

    coco = COCO(str(ann_path))
    crop_save_dir = img_dir.parents[0] / 'crops'
    
    crop_meta_data = {}
    for cat_id, cat_info in coco.cats.items():
        cat_name = cat_info['name']
        cat_crop_save_dir = crop_save_dir / cat_name
        cat_crop_save_dir.mkdir(exist_ok=True, parents=True)

        crop_meta_data[cat_name] = []

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids, desc='crop images...'):
        img_file_name = coco.loadImgs(img_id)[0]['file_name']
        img_path = img_dir / img_file_name
        img_array = np.array(Image.open(str(img_path)))

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ids=ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            cat_name = coco.cats[cat_id]['name']

            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h

            crop_img = img_array[y1:y2, x1:x2, :]
            crop_img = Image.fromarray(crop_img)

            crop_file_name = f'{len(crop_meta_data[cat_name])}.png'
            crop_img_save_path = crop_save_dir / cat_name / crop_file_name
            crop_img.save(str(crop_img_save_path))

            crop_meta_info = {
                'file_name': crop_file_name,
                'ori_img_size': img_array.shape[:2],
                'crop_img_size': (h, w),
                'location': (x1, y1, x2, y2)
            }
            crop_meta_data[cat_name].append(crop_meta_info)

    # crop_meta_data: {
    #     class name: [
    #         {
    #             'file_name': 0.png,
    #             'ori_img_size': [1024, 2048],
    #             'crop_img_size': [61, 26],
    #             'location': [x1, y1, x2, y2]
    #         }
    #     ]
    # }
    
    crop_meta_data_save_path = crop_save_dir / 'crop_meta_data.json'
    json_object = json.dumps(crop_meta_data, indent=4)
 
    with open(str(crop_meta_data_save_path), 'w') as f:
        f.write(json_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='datasets/ip2p/c2fc_vgg_0706_general_large_cfg_text=10/leftImg8bit/train')
    parser.add_argument('--ann-path', type=str, default='datasets/cityscapes/annotations/cityscapes_train.json')
    args = parser.parse_args()

    main(args)
