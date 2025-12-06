import cv2
from pathlib import Path


def draw_boxes_and_save(data, gt_inst, pseudo_inst, added_inst, class_names, output_dir, img_file_name):
    cv2_img = data['image'].permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()

    if gt_inst is not None:
        for gt_box, cls_idx in zip(gt_inst.gt_boxes.tensor, gt_inst.gt_classes):
            box_loc = gt_box.cpu().numpy().astype(int)
            cls_name = class_names[cls_idx]
            img = draw_boxes(cv2_img, box_loc, cls_name, is_gt_box=True)

    if pseudo_inst is not None:
        for pseudo_box, cls_idx in zip(pseudo_inst.gt_boxes.tensor, pseudo_inst.gt_classes):
            box_loc = pseudo_box.cpu().numpy().astype(int)
            cls_name = class_names[cls_idx]
            img = draw_boxes(cv2_img, box_loc, cls_name, is_pseudo_box=True)

    if added_inst is not None:
        for added_box, cls_idx in zip(added_inst.gt_boxes.tensor, added_inst.gt_classes):
            box_loc = added_box.cpu().numpy().astype(int)
            cls_name = class_names[cls_idx]
            img = draw_boxes(cv2_img, box_loc, cls_name, is_added_box=True)

    img_save_dir = Path(output_dir) / 'gt+pseudo+added_label_images'
    img_save_dir.mkdir(exist_ok=True, parents=True)
    img_save_path = img_save_dir / img_file_name

    cv2.imwrite(str(img_save_path), cv2_img)


def draw_boxes(img, box_loc, cls_name, is_gt_box=False, is_pseudo_box=False, is_added_box=False,):
    box_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    text_color = (255, 255, 255)
    color = (0, 255, 255) if is_gt_box else (255, 0, 0)

    if is_gt_box:
        color = (0, 255, 0)  # green for gt
    elif is_pseudo_box:
        color = (255, 0, 0)  # blue for pseudo labels
    elif is_added_box:
        color = (0, 0, 255)  # red for added labels
    else:
        raise ValueError

    x1, y1, x2, y2 = box_loc

    # bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=box_thickness)

    # class
    text_size, _ = cv2.getTextSize(cls_name, font, font_scale, font_thickness)
    text_w, text_h = text_size

    upper_left = (x1, y1 - text_h) if is_gt_box else (x2 - text_w, y1 - text_h)
    lower_right = (x1 + text_w, y1) if is_gt_box else (x2, y1)
    text_pos = (x1, y1) if is_gt_box else (x2 - text_w, y1)

    cv2.rectangle(img, upper_left, lower_right, color, -1)
    cv2.putText(img, cls_name, text_pos, font, font_scale, text_color, font_thickness)
    
    return img