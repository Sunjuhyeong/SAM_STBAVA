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
from loss import IoULoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
from utils.utility import logger, mask_iou

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--config', default="./sam_avs_adapter.yaml")
    parser.add_argument("--log_dir", default="./train_logs", type=str)
    parser.add_argument('--use_adapter', action='store_true', default=True)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument(
        "--sam_weights", type=str, default="sam_sandbox", help="path of trained model"
    )
    parser.add_argument(
        "--train_weights", type=str, default="", help="path of trained model"
    )

    args = parser.parse_args()

    from model import STBAVA

    torch.multiprocessing.set_start_method("spawn")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
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

    scripts_to_save = [
        "config.py",
        "dataloader.py",
        "./model/STBAVA.py",
        "./model/SAMA_Enc.py",
        "loss.py",
    ]
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

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
    writer = SummaryWriter(args.log_dir)

    # Model
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.config = config
    model = STBAVA.Decoder(config=args.config, depth=args.depth, use_4gpus=True)
    model.cuda()

    model.load_state_dict(torch.load(os.path.join(args.sam_weights, "sama_ms3_best.pth")), strict=False)
    prompt_path = os.path.join(args.sam_weights, "prompt_encoder.pth")
    mask_decoder_path = os.path.join(args.sam_weights, "mask_decoder.pth") 
    pe_layer_path = os.path.join(args.sam_weights, "pe_layer.pth")
    model.prompt_encoder.load_state_dict(torch.load(prompt_path))
    model.mask_decoder.load_state_dict(torch.load(mask_decoder_path))
    model.pe_layer.load_state_dict(torch.load(pe_layer_path))
    for i in range(32):
        if i < 9: 
            model.image_encoder.blocks[i].to("cuda:3")
        elif 9 <= i < 18:
            model.image_encoder.blocks[i].to("cuda:2")
        elif 18 <= i < 27:
            model.image_encoder.blocks[i].to("cuda:1")
        else:
            model.image_encoder.blocks[i].to("cuda:0")
    logger.info(
        "==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6)
    )
    if os.path.exists(args.train_weights):
        model.load_state_dict(torch.load(args.train_weights), strict=False)
        print("load latest weights")

    for name, para in model.named_parameters():        
        if "prompt_generator" in name:
            para.requires_grad_(True)
        elif "mask_decoder" in name:
            para.requires_grad_(True)
        else:
            para.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device, model_type='vggish')
    audio_backbone.cuda()
    print("==> Total params of audio backbone: %.2fM" % (sum(p.numel() for p in audio_backbone.parameters()) / 1e6))

    # Data
    train_dataset = MS3Dataset_SAM("train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    test_dataset = MS3Dataset_SAM("val")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)

    if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
        audio_backbone_params = audio_backbone.parameters()
        optimizer_audio = torch.optim.Adam(audio_backbone_params, args.lr * 10)
    else:
        audio_backbone.eval()

    avg_meter_total_loss = pyutils.AverageMeter("total_loss")
    avg_meter_iou_loss = pyutils.AverageMeter("iou_loss")

    avg_meter_miou = pyutils.AverageMeter("miou")
    avg_meter_F = pyutils.AverageMeter("F_score")

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0

    for epoch in range(args.max_epoches):
        model.train()
        if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
            audio_backbone.train()

        for n_iter, batch_data in enumerate(train_dataloader):
            (
                imgs,
                audio,
                mask,
                video_name,
            ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5 or 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()

            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)

            mask_num = 5
            mask = mask.view(B * mask_num, 1, mask.shape[-2], mask.shape[-1])
            audio = audio.view(
                -1, audio.shape[2], audio.shape[3], audio.shape[4]
            )  # [B*T, 1, 96, 64]
            audio_feature = audio_backbone(audio)  # [B*T, 128]
            output = model(
                imgs, audio_feature=audio_feature
            )  # [bs*5, 1, 224, 224]
            
            loss, loss_dict = IoULoss(output, mask)

            avg_meter_total_loss.add({"total_loss": loss.item()})
            avg_meter_iou_loss.add({"iou_loss": loss_dict["iou_loss"]})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR:
                optimizer_audio.zero_grad()
                optimizer_audio.step()
            writer.add_scalar("train loss",
                            loss.item(), global_step)
            global_step += 1
            if (global_step - 1) % 20 == 0:
                train_log = (
                    "Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, lr: %.5f"
                    % (
                        global_step - 1,
                        max_step,
                        avg_meter_total_loss.pop("total_loss"),
                        avg_meter_iou_loss.pop("iou_loss"),
                        optimizer.param_groups[0]["lr"],
                    )
                )
                logger.info(train_log)

        # Validation:
        model.eval()
        audio_backbone.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(test_dataloader):
                (
                    imgs,
                    audio,
                    mask,
                    video_name,
                ) = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, mask.shape[-2], mask.shape[-1])
                audio = audio.view(
                    -1, audio.shape[2], audio.shape[3], audio.shape[4]
                )
                audio_feature = audio_backbone(audio)
                output = model(
                    imgs, audio_feature=audio_feature
                )  # [bs*5, 1, 224, 224]
                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({"miou": miou})

            miou = avg_meter_miou.pop("miou")
            if miou > max_miou:
                model_save_path = os.path.join(
                    checkpoint_dir, "%s_best.pth" % (args.session_name)
                )
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info("save best model to %s" % model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = "Epoch: {}, Miou: {}, maxMiou: {}".format(
                epoch, miou, max_miou
            )
            # print(val_log)
            logger.info(val_log)
            writer.add_scalar("val miou",
                miou.item(), epoch)

    logger.info("best val Miou {} at peoch: {}".format(max_miou, best_epoch))
    writer.close()
