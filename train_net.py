#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from cropat import add_ateacher_config
from cropat.engine.trainer import ATeacherTrainer, BaselineTrainer
from cropat.engine.trainer_cropat import CropATeacherTrainer

# hacky way to register
from cropat.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from cropat.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from cropat.modeling.meta_arch.resnet import build_resnet101_backbone  # noqa
from cropat.modeling.proposal_generator.rpn import PseudoLabRPN
from cropat.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import cropat.data.datasets.builtin

from cropat.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    elif cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "CropATeacherTrainer":
        Trainer = CropATeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer in ["ateacher", "CropATeacherTrainer"]:
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            teacher_res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            student_res = Trainer.test(cfg, ensem_ts_model.modelStudent)
            res = {
                **teacher_res,
                **student_res
            }

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
