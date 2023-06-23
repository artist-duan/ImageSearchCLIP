#!/usr/bin/env python3
import time
import clip
import torch
import logging
from PIL import Image
import cn_clip.clip as clip_cn
from functools import lru_cache
from typing import Any, Union, List


class CLIP_Model:
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            self.config.CLIP_MODEL_PATH, device=self.device
        )
        self.model.eval()

    def image_feature(self, image):
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except:
                return None, None

        image_size = image.size
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image).detach().cpu().numpy()
        return image_feature, image_size

    def text_feature(self, text: Union[str, List[str]]):
        with torch.no_grad():
            if isinstance(text, str):
                text = clip.tokenize([text]).to(self.device)
            elif isinstance(text, list):
                text = clip.tokenize(text).to(self.device)
            else:
                raise NotImplementedError

            text_feature = self.model.encode_text(text).detach().cpu().numpy()
        return text_feature


class CNCLIP_Model:
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip_cn.load_from_name(
            self.config.CN_CLIP_MODEL,
            device=self.device,
            download_root=self.config.CN_CLIP_MODEL_PATH,
        )
        self.model.eval()

    def image_feature(self, image):
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except:
                return None, None

        image_size = image.size
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image).detach().cpu().numpy()
        return image_feature, image_size

    def text_feature(self, text: Union[str, List[str]]):
        with torch.no_grad():
            if isinstance(text, str):
                text = clip_cn.tokenize([text]).to(self.device)
            elif isinstance(text, list):
                text = clip_cn.tokenize(text).to(self.device)
            else:
                raise NotImplementedError

            text_feature = self.model.encode_text(text).detach().cpu().numpy()
        return text_feature


@lru_cache(maxsize=1)
def get_model(config) -> CLIP_Model:
    tic = time.time()
    model = CLIP_Model(config)
    toc = time.time()
    logging.info(
        f"load CLIP-Model[{config.CLIP_MODEL}][{model.device}] using {toc - tic:.3f} seconds."
    )

    tic = time.time()
    cn_model = CNCLIP_Model(config)
    toc = time.time()
    logging.info(
        f"load CN-CLIP-Model[{config.CN_CLIP_MODEL}][{cn_model.device}] using {toc - tic:.3f} seconds."
    )
    return model, cn_model
