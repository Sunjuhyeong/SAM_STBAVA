import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
import yaml
from config import cfg
from dataloader import S4Dataset_SAM
from torchvggish import vggish

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
from utils.utility import logger, mask_iou, Eval_Fmeasure

import torch.nn as nn

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
    parser.add_argument("--session_name", default="S4_SAM_decoder", type=str, help="the S4 setting")

    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--log_dir', default='test_logs', type=str)
    parser.add_argument("--test_weights", type=str, default='', help='path of trained model')
    parser.add_argument('--config', default="sam_avs_adapter.yaml")
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument(
        "--sam_weights", type=str, default="sam_sandbox", help="path of trained model"
    )

    args = parser.parse_args()

    from model import STBAVA
    print('==> Use SAM decoder as the visual backbone...')

    torch.multiprocessing.set_start_method('spawn')

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
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['test_STBAVA.py', 'config.py', 'dataloader.py', './model/STBAVA.py', './model/SAMA_Enc.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.config = config
    model = STBAVA.Decoder(depth=args.depth, config=args.config)
    model.cuda()

    for i in range(32):
        if i < 9: 
            model.image_encoder.blocks[i].to("cuda:3")
        elif 9 <= i < 18:
            model.image_encoder.blocks[i].to("cuda:2")
        elif 18 <= i < 27:
            model.image_encoder.blocks[i].to("cuda:1")
        else:
            model.image_encoder.blocks[i].to("cuda:0")

    if os.path.exists(args.test_weights):
        model.load_state_dict(torch.load(args.test_weights), strict=False)
        print("load pre_trained weights")

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    test_dataset = S4Dataset_SAM('val')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=False)

    audio_backbone.eval()
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            (
                imgs,
                audio,
                mask,
                category,
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

            miou = mask_iou(output.squeeze(1).squeeze(1), mask)
            avg_meter_miou.add({"miou": miou})
            
            F_score = Eval_Fmeasure(output.squeeze(1).squeeze(1), mask, log_dir)
            avg_meter_F.add({"F_score": F_score})
            
            print("n_iter: {}, iou: {}, F_score: {}".format(n_iter, miou, F_score))

        miou = avg_meter_miou.pop("miou")
        F_score = avg_meter_F.pop("F_score")
        print("test miou:", miou.item())
        print("test F_score:", F_score)
        logger.info("test miou: {}, F_score: {}".format(miou.item(), F_score))
