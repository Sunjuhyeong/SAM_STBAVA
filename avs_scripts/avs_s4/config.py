from easydict import EasyDict as edict

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4
cfg.LAMBDA_1 = 50

##############################
# TRAIN
cfg.TRAIN = edict()
# TRAIN.SCHEDULER
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "../../pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "../../pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False

###############################
# DATA
path_root = 'nas3'
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = f"../../avsbench_data/Single-source/s4_meta_data.csv"
cfg.DATA.DIR_IMG = f"../../avsbench_data/Single-source/s4_data/visual_frames"
cfg.DATA.DIR_AUDIO_WAV = f"../../avsbench_data/Single-source/s4_data/audio_wav"
cfg.DATA.DIR_AUDIO_LOG_MEL = f"../../avsbench_data/Single-source/s4_data/audio_log_mel"
cfg.DATA.DIR_MASK = f"../../avsbench_data/Single-source/s4_data/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)
###############################


if __name__ == "__main__":
    print(cfg)
