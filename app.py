#!/usr/bin/env python3
import os
import time
import uuid
import pickle
import torch
import logging
import numpy as np
import gradio as gr
from PIL import Image
from tqdm import tqdm
from pympler import asizeof
from functools import lru_cache
from typing import List, Tuple, Any

from utils import *
import config as cfg
from models.clip_model import get_model


class Demo:
    def __init__(self, config):
        self.config = config
        self.feature_dim = config.FEATURE_LENGTH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.datas = self.get_datas(
            os.path.join(self.config.IMAGE_PATH, self.config.IMAGE_FEATURE_PATH)
        )
        self.model = get_model(config)

    @lru_cache(maxsize=1)
    def get_datas(self, path):
        paths = traverse_path(path, extensions={".pkl"}, complete=True)

        if self.config.PRE_LOAD:
            datas = []
            for path in paths:
                with open(path, "rb") as fr:
                    info = pickle.load(fr)
                datas.append(info)
            memory = asizeof.asizeof(datas) / (1024 * 1024)
            logging.info(
                f"preload all information, number-{len(datas)}, size-{memory:.3f} MB."
            )
            return datas
        return paths

    def cosine_similarity(self, query_feature, features):
        logging.info(
            f"query_feature-{query_feature.shape}, query_feature-{features.shape}."
        )
        query_feature = query_feature / np.linalg.norm(
            query_feature, axis=1, keepdims=True
        )
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        scores = query_feature @ features.T
        return scores[0]

    def search_nearest(self, query_feature, options={}):
        topk = options["topk"]
        minimum_height = options["minimum_height"]
        minimum_width = options["minimum_width"]

        infos, features, scores = [], [], []
        for info in tqdm(self.datas, total=len(self.datas), desc="matching"):
            if isinstance(info, str):
                with open(info, "rb") as fr:
                    info = pickle.load(fr)

            condition = (len(options["extension"]) <= 0) or (
                info["extension"] in options["extension"]
            )
            condition = (
                condition
                and (info["height"] >= minimum_height)
                and (info["width"] >= minimum_width)
            )
            if not condition:
                continue

            if not os.path.exists(info["filename"]):
                continue

            features.append(info["feature"][0])
            infos.append(
                [
                    info["filename"],
                    info["height"],
                    info["width"],
                    info["filesize"],
                    info["extension"],
                    info["date"],
                ]
            )
            if len(features) >= self.config.MAX_SPLIT_SIZE:
                features = np.array(features, dtype=np.float32)
                scores.append(self.cosine_similarity(query_feature, features))
                features = []

        if len(features) > 0:
            features = np.array(features, dtype=np.float32)
            scores.append(self.cosine_similarity(query_feature, features))

        if len(scores) == 0:
            return [], []

        scores = np.concatenate(scores, axis=0)
        topk_idx = np.argsort(scores)[::-1][:topk]
        topk_infos = [infos[idx] for idx in topk_idx]
        topk_scores = [float(scores[idx]) for idx in topk_idx]
        return topk_infos, topk_scores

    def search_nearest_text_image():
        pass

    def search_nearest_pntext():
        pass

    def search_nearest_pntext_image():
        pass

    def vis_results(self, infos: List[str], scores: List[float]):
        results = []
        for info, score in zip(infos, scores):
            filename, h, w, size, ext, date = info
            s = "score={:.5f}\n".format(score)
            s += filename + "\n"
            s += f"{w}x{h} size={size}\ndate={date}"
            results.append((filename, s))
        return results

    def search_image(
        self,
        positive_query,
        negative_query,
        image_query,
        topk,
        minimum_width,
        minimum_height,
        extension,
    ):
        # TODO:
        # prompt + image
        # negative prompt

        positive_feature, negative_feature, image_feature = None, None, None
        if len(positive_query) != 0:
            positive_feature = self.model.text_feature(positive_query)
        if len(negative_query) != 0:
            negative_feature = self.model.text_feature(negative_query)
        if isinstance(image_query, Image.Image):
            image_feature, _ = self.model.image_feature(image_query)

        options = {
            "topk": int(topk),
            "minimum_width": minimum_width,
            "minimum_height": minimum_height,
            "extension": extension,
        }

        if positive_feature is None and image_feature is None:
            assert False, "No enough query input."

        elif positive_feature is None:
            infos, scores = self.search_nearest(image_feature, options=options)

        elif image_feature is None:
            if negative_feature is None:
                infos, scores = self.search_nearest(positive_feature, options=options)
            else:
                infos, scores = self.search_nearest_pntext(
                    positive_feature, negative_feature
                )

        else:
            if negative_feature is None:
                infos, scores = self.search_nearest_text_image(
                    positive_feature, image_feature
                )
            else:
                infos, scores = self.search_nearest_pntext_image(
                    positive_feature, negative_feature, image_feature
                )

        print(len(infos), len(scores))
        return self.vis_results(infos, scores)

    def server(self):
        # build gradio app
        with gr.Blocks() as demo:
            heading = gr.Markdown("# Image Search Using CLIP")
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_textbox = gr.Textbox(lines=4, label="Prompt")
                    negative_prompt_textbox = gr.Textbox(
                        lines=4, label="Negative Prompt"
                    )
                    button = gr.Button("Search").style(size="lg")
                with gr.Column(scale=2):
                    input_image = gr.Image(label="Image", type="pil")

            with gr.Accordion("Search options", open=False):
                extension_choice = gr.CheckboxGroup(
                    ["jpg", "png", "gif"],
                    label="extension",
                    info="choose extension for search",
                )
                with gr.Row():
                    topk = gr.Number(value=16, label="topk")
                    minimum_width = gr.Number(value=0, label="minimum_width")
                    minimun_height = gr.Number(value=0, label="minimum_height")

            gallery = gr.Gallery(label="results").style(grid=4)

            button.click(
                self.search_image,
                inputs=[
                    prompt_textbox,
                    negative_prompt_textbox,
                    input_image,
                    topk,
                    minimum_width,
                    minimun_height,
                    extension_choice,
                ],
                outputs=[gallery],
            )

        demo.launch(server_name=cfg.HOST, server_port=cfg.PORT)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app = Demo(cfg)
    app.server()
