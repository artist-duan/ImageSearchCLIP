#!/usr/bin/env python3


IMAGE_PATH = "/media/xrgaze/test2t/DUAN-HUAIWEI-MATE40PRO"
IMAGE_FEATURE_PATH = "clip_feature"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".ico",
}


HOST = "0.0.0.0"
PORT = 7680
PRE_LOAD = True
MAX_SPLIT_SIZE = 4096

CLIP_MODEL_PATH = "./checkpoints/clip/ViT-L-14-336px.pt"
CLIP_MODEL = "ViT-L/14@336px"
FEATURE_LENGTH = 768
# CLIP_MODEL_PATH = "./checkpoints/clip/ViT-B-32.pt"
# CLIP_MODEL = "ViT-B/32"
# FEATURE_LENGTH = 512
POSITIVE_PROMPT_THRESHOLD = 0.22
NEGATIVE_PROMPT_THRESHOLD = 0.18
IMAGE_THRESHOLD = 0.6


# https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14-336.pt
CN_CLIP_MODEL_PATH = "./checkpoints/clip"
CN_CLIP_MODEL = "ViT-L-14-336"
CN_FEATURE_LENGTH = 768
CN_POSITIVE_PROMPT_THRESHOLD = 0.22
CN_NEGATIVE_PROMPT_THRESHOLD = 0.18
CN_IMAGE_THRESHOLD = 0.6
