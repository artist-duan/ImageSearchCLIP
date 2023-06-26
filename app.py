#!/usr/bin/env python3
import os
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
        self.model, self.cn_model = get_model(config)

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

    def cosine_similarity(self, query_feature, features, negative=False):
        query_feature = query_feature / np.linalg.norm(
            query_feature, axis=1, keepdims=True
        )
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        scores = query_feature @ features.T
        logging.info(
            f"query_feature-{query_feature.shape}, query_feature-{features.shape}, scores-{scores.shape}."
        )
        if negative:
            scores = scores.max(axis=0)
        else:
            scores = scores.min(axis=0)
        return scores

    def search_nearest(
        self,
        image_feature=None,
        positive_feature=None,
        negative_feature=None,
        options={},
        is_cn=False,
    ):
        topk = options["topk"]
        minimum_height = options["minimum_height"]
        minimum_width = options["minimum_width"]

        infos, features = [], []
        image_scores, positive_scores, negative_scores = [], [], []
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

            if is_cn:
                features.append(info["cn_feature"][0])
            else:
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
                if image_feature is not None:
                    image_scores.append(self.cosine_similarity(image_feature, features))
                if positive_feature is not None:
                    positive_scores.append(
                        self.cosine_similarity(positive_feature, features)
                    )
                if negative_feature is not None:
                    negative_scores.append(
                        self.cosine_similarity(negative_feature, features, True)
                    )
                features = []

        if len(features) > 0:
            features = np.array(features, dtype=np.float32)
            if image_feature is not None:
                image_scores.append(self.cosine_similarity(image_feature, features))
            if positive_feature is not None:
                positive_scores.append(
                    self.cosine_similarity(positive_feature, features)
                )
            if negative_feature is not None:
                negative_scores.append(
                    self.cosine_similarity(negative_feature, features)
                )

        if len(image_scores):
            image_scores = np.concatenate(image_scores, axis=0)
        if len(positive_scores):
            positive_scores = np.concatenate(positive_scores, axis=0)
        if len(negative_scores):
            negative_scores = np.concatenate(negative_scores, axis=0)

        if len(image_scores) == 0 and len(positive_scores) == 0:
            return [], []
        elif len(image_scores) == 0:
            sort_idx = np.argsort(positive_scores)[::-1]
            image_scores = np.ones_like(positive_scores)
        elif len(positive_scores) == 0:
            sort_idx = np.argsort(image_scores)[::-1]
            positive_scores = np.ones_like(image_scores)
        else:
            sort_idx = np.argsort(image_scores)[::-1]

        if len(negative_scores) == 0:
            negative_scores = np.zeros_like(positive_scores)

        image_threshold = (
            self.config.CN_IMAGE_THRESHOLD if is_cn else self.config.IMAGE_THRESHOLD
        )
        positive_prompt_threshold = (
            self.config.CN_POSITIVE_PROMPT_THRESHOLD
            if is_cn
            else self.config.POSITIVE_PROMPT_THRESHOLD
        )
        negative_prompt_threshold = (
            self.config.CN_NEGATIVE_PROMPT_THRESHOLD
            if is_cn
            else self.config.NEGATIVE_PROMPT_THRESHOLD
        )
        sort_infos, sort_scores = [], []
        for idx in sort_idx:
            i_score = float(image_scores[idx])
            p_score = float(positive_scores[idx])
            n_score = float(negative_scores[idx])
            if (
                i_score < image_threshold
                or p_score < positive_prompt_threshold
                or n_score >= negative_prompt_threshold
            ):
                continue
            sort_infos.append(infos[idx])
            sort_scores.append(i_score * p_score)

        return sort_infos[:topk], sort_scores[:topk]

    def vis_results(self, infos: List[str], scores: List[float]):
        results = []
        for info, score in zip(infos, scores):
            filename, h, w, size, ext, date = info
            s = "score={:.5f}\n".format(score)
            s += filename + "\n"
            s += f"{w}x{h} size={size}\ndate={date}"
            results.append((filename, s))
        return results

    def is_chinese(self, text):
        for t in text:
            if "\u4e00" <= t <= "\u9fa5":
                return True
        return False

    def text_feature(self, query, is_cn=False):
        if is_cn:
            query = query.split("；")
            query = [q.strip() for q in query if len(q.strip())]
            feature = self.cn_model.text_feature(query)
        else:
            query = query.split(";")
            query = [q.strip() for q in query if len(q.strip())]
            feature = self.model.text_feature(query)
        return feature

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
        positive_feature, negative_feature, image_feature = None, None, None

        is_cn = False
        if len(positive_query) != 0 and self.is_chinese(positive_query):
            if len(negative_query) == 0 or self.is_chinese(negative_query):
                is_cn = True
            else:
                assert False, "Keep the same language."

        if len(positive_query) != 0:
            positive_feature = self.text_feature(positive_query, is_cn)

        if len(negative_query) != 0:
            negative_feature = self.text_feature(negative_query, is_cn)

        if isinstance(image_query, Image.Image):
            if is_cn:
                image_feature, _ = self.cn_model.image_feature(image_query)
            else:
                image_feature, _ = self.model.image_feature(image_query)

        options = {
            "topk": int(topk),
            "minimum_width": minimum_width,
            "minimum_height": minimum_height,
            "extension": extension,
        }

        if positive_feature is None and image_feature is None:
            assert False, "No enough query input."
        else:
            infos, scores = self.search_nearest(
                image_feature,
                positive_feature,
                negative_feature,
                options=options,
                is_cn=is_cn,
            )
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
                    topk = gr.Number(value=32, label="topk")
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

        demo.launch(server_name=cfg.HOST, server_port=cfg.PORT, debug=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app = Demo(cfg)
    app.server()
