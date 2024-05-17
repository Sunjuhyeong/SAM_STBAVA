import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import MS3Dataset_SAM
from torchvggish import vggish

from utils import pyutils
from utils.system import setup_logging
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask

import torch.nn as nn

import yaml

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device, model_type='vggish', prompt_dim=256):
        super(audio_extractor, self).__init__()
        self.model_type = model_type
        self.prompt_dim = prompt_dim
        self.audio_backbone = vggish.VGGish(cfg, device)

    def init_random_params(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(
                    m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward_vggish(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea.unsqueeze(1)

    def forward(self, audio):
        audio_fea = self.forward_vggish(audio)

        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_name", default="MS3_SAM", type=str, help="the MS3 setting"
    )
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--config', default="sam_avs_adapter.yaml")
    parser.add_argument("--log_dir", default="./test_logs", type=str)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument(
        "--test_weights", type=str, default="./best_model.pth", help="path of trained model"
    )
    parser.add_argument(
        "--save_pred_mask",
        action="store_true",
        default=True,
        help="save predited masks or not",
    )
    
    args = parser.parse_args()

    from model import STBAVA

    # torch.multiprocessing.set_start_method("spawn")
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
        
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(
        args.log_dir, "{}".format(time.strftime(prefix + "_%Y%m%d-%H%M%S"))
    )
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, "scripts")
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, "log.txt"))
    logger = logging.getLogger(__name__)
    logger.info("==> Config: {}".format(cfg))
    logger.info("==> Arguments: {}".format(args))
    logger.info("==> Experiment: {}".format(args.session_name))

    # Model
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.config = config
    model = STBAVA.Decoder(config=args.config, depth=args.depth)
    model.cuda()

    logger.info(
        "==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device, model_type='vggish')
    audio_backbone.cuda()
    print("==> Total params of audio backbone: %.2fM" % (sum(p.numel() for p in audio_backbone.parameters()) / 1e6))

    # Data
    test_dataset = MS3Dataset_SAM("test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    avg_meter_miou = pyutils.AverageMeter("miou")
    avg_meter_F = pyutils.AverageMeter("F_score")

    model.load_state_dict(torch.load(args.test_weights), strict=False)
    print(f"load from {args.test_weights}")
    
    model.eval()
    audio_backbone.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            (
                imgs,
                audio,
                mask,
                video_name,
            ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape

            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B * frame, mask.shape[-2], mask.shape[-1])
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            audio_feature = audio_backbone(audio)

            output = model(
                imgs, audio_feature=audio_feature
            )

            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, "pred_masks")
                save_mask(output.squeeze(1), mask_save_path, video_name)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({"miou": miou})
            
            F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
            avg_meter_F.add({"F_score": F_score})
            
            print("n_iter: {}, iou: {}, F_score: {}".format(n_iter, miou, F_score))

        miou = avg_meter_miou.pop("miou")
        F_score = avg_meter_F.pop("F_score")
        print("test miou:", miou.item())
        print("test F_score:", F_score)
        logger.info("test miou: {}, F_score: {}".format(miou.item(), F_score))
