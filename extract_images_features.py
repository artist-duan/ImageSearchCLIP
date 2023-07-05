#!/usr/bin/env python3
import os
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import *
import config as cfg
from models.clip_model import get_model as get_clip_model
from models.ocr_model import get_model as get_ocr_model


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--eval", action="store_true")
    # args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    root_path = cfg.IMAGE_PATH
    feature_path = cfg.IMAGE_FEATURE_PATH

    """ model """
    clip_model, cnclip_model = get_clip_model(cfg)
    ocr_model = get_ocr_model()

    """ traverse image path """
    images = traverse_path(root_path, extensions=cfg.IMAGE_EXTENSIONS)
    extracted_images = traverse_path(
        os.path.join(root_path, feature_path), extensions={".pkl"}, complete=False
    )
    extracted_images = set(extracted_images)
    logging.info(
        f"number of images: {len(images)}; number of extracted images: {len(extracted_images)}."
    )

    """ extract feature for all image """
    for image in tqdm(images, total=len(images), desc="extract image feature"):
        # skip extracted image
        feature_file = image.split("/")[-1].split(".")[0] + ".pkl"
        if feature_file in extracted_images:
            continue
        extension = os.path.splitext(image)[1].lower()[1:]

        # extract image feature
        image_feature, image_size = clip_model.image_feature(image)
        if image_feature is None or image_size is None:
            logging.info(f"skip [{image}], file not exist.")
            continue
        cn_image_feature, image_size = cnclip_model.image_feature(image)
        if cn_image_feature is None or image_size is None:
            logging.info(f"skip [{image}], file not exist.")
            continue

        # extract ocr
        texts, boxes, scores = ocr_model.det_ocr(image)

        # save info
        stat = os.stat(image)
        image_st_mtime = datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        info = {
            "filename": image,
            "extension": extension,
            "height": image_size[1],
            "width": image_size[0],
            "filesize": stat.st_size,
            "date": image_st_mtime,
            "feature": image_feature,
            "cn_feature": cn_image_feature,
            "ocr": texts,
        }

        save_path = os.path.dirname(image).replace(
            root_path, os.path.join(root_path, feature_path)
        )
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, feature_file), "wb") as fw:
            pickle.dump(info, fw)


if __name__ == "__main__":
    main()
