import numpy as np

import torch


class Mosaic():

    def __init__(self, p, r):
        self.p = p
        self.r = r

    def __call__(self, label_data):
        for i, single_label_data in enumerate(label_data):
            assert 'translated_image' in single_label_data
            if np.random.rand(1) > self.p:
                label_data[i] = self.null_transform(single_label_data)
            else:
                label_data[i] = self.transform_single_data(single_label_data)

        return label_data

    def null_transform(self, single_label_data):
        src_img = single_label_data['image']  # (C, H, W)
        _, h, w = src_img.shape
        single_label_data['mosaic_mask'] = torch.zeros(h, w)

        return single_label_data

    def transform_single_data(self, single_label_data):
        src_img = single_label_data['image']  # (C, H, W)
        translated_img = single_label_data['translated_image']  # (C, H, W)
        domain_label = single_label_data['translated_domain_label']
        
        _, h, w = src_img.shape

        transformed_src_img = src_img.clone().detach()
        mask = torch.zeros(h, w).long()

        x_c, y_c = self.sample_center(h, w)
        sampled_patch_idx = self.sample_patch_idx()

        for patch_idx in range(4):
            # do not swap
            if patch_idx not in sampled_patch_idx:
                continue

            # swap
            x_min, y_min, x_max, y_max = 0, 0, w, h
            if patch_idx == 0:
                x_min, y_min, x_max, y_max = 0, 0, x_c, y_c
            elif patch_idx == 1:
                x_min, y_min, x_max, y_max = x_c, 0, w, y_c
            elif patch_idx == 2:
                x_min, y_min, x_max, y_max = 0, y_c, x_c, h
            elif patch_idx == 3:
                x_min, y_min, x_max, y_max = x_c, y_c, w, h

            src_patch = src_img[:, y_min:y_max, x_min:x_max].clone().detach()
            translated_patch = translated_img[:, y_min:y_max, x_min:x_max].clone().detach()

            transformed_src_img[:, y_min:y_max, x_min:x_max] = (1 - self.r) * src_patch + self.r * translated_patch
            mask[y_min:y_max, x_min:x_max] = 1

        single_label_data['image'] = transformed_src_img
        single_label_data['mosaic_mask'] = torch.where(mask == 0, mask, domain_label)

        return single_label_data

    def sample_center(self, h, w):
        mu, sigma = 0.5, 0.1

        x_c = np.clip(
            int((np.random.randn(1) * sigma + mu) * w),
            int(0.1 * w),
            int(0.9 * w)
        )
        y_c = np.clip(
            int((np.random.randn(1) * sigma + mu) * h),
            int(0.1 * h),
            int(0.9 * h)    
        )

        return x_c, y_c

    def sample_patch_idx(self):
        return np.random.choice(4, size=2, replace=False).tolist()