#!/usr/bin/env python3
import os
import re
import pickle
import torch
import logging
import numpy as np
import gradio as gr
from PIL import Image
from tqdm import tqdm
import Levenshtein as lev
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

    def search_ocr(self, positive_ocr, negative_ocr, image_ocrs):
        def string_similarity(dst_texts, src_texts):
            scores = []
            for dst in dst_texts:
                score = -1
                for src in src_texts:
                    dst, src = dst.lower(), src.lower()
                    if dst == src or dst in src or src in dst:
                        score = 1
                    distance = lev.distance(dst, src)
                    score = max(score, 1 - distance / max(len(dst), len(src)))
                scores.append(score)
            return scores

        if len(image_ocrs) == 0 and len(positive_ocr) == 0:
            return 0
        score = 0
        pos_scores = string_similarity(positive_ocr, image_ocrs)
        if len(pos_scores) and min(pos_scores) >= self.config.POSITIVE_TEXT_THRESHOLD:
            score = 1
        neg_scores = string_similarity(negative_ocr, image_ocrs)
        if len(neg_scores) and max(neg_scores) >= self.config.NEGATIVE_TEXT_THRESHOLD:
            score = -1
        return score

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
        positive_ocr=[],
        negative_ocr=[],
        options={},
        is_cn=False,
    ):
        topk = options["topk"]
        minimum_height, minimum_width = (
            options["minimum_height"],
            options["minimum_width"],
        )

        infos, features = [], []
        image_scores, positive_scores, negative_scores, ocr_scores = [], [], [], []
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

            ocr_score = self.search_ocr(positive_ocr, negative_ocr, info["ocr"])
            ocr_scores.append(ocr_score)

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
        ocr_scores = np.array(ocr_scores, dtype=np.float32)

        if (
            len(image_scores) == 0
            and len(positive_scores) == 0
            and (ocr_scores == 0).all()
        ):
            return [], []

        elif len(image_scores) != 0:
            sort_idx = np.argsort(image_scores)[::-1]
            positive_scores = (
                np.ones_like(image_scores)
                if len(positive_scores) == 0
                else positive_scores
            )
            ocr_scores = (
                np.ones_like(image_scores) if (ocr_scores == 0).all() else ocr_scores
            )

        elif len(positive_scores) != 0:
            sort_idx = np.argsort(positive_scores)[::-1]
            image_scores = np.ones_like(positive_scores)
            ocr_scores = (
                np.ones_like(positive_scores) if (ocr_scores == 0).all() else ocr_scores
            )

        elif not (ocr_scores == 0).all():
            sort_idx = np.argsort(ocr_scores)[::-1]
            image_scores = np.ones_like(ocr_scores)
            positive_scores = np.ones_like(ocr_scores)

        else:
            pass

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
            o_score = float(ocr_scores[idx])
            if (
                i_score < image_threshold
                or p_score < positive_prompt_threshold
                or n_score >= negative_prompt_threshold
                or o_score != 1
            ):
                continue
            sort_infos.append(infos[idx])
            sort_scores.append(i_score * p_score * o_score)

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

    def extract_ocr(self, text):
        if "<text>" in text:
            pattern = r"<text>(.*)"
            match = re.search(pattern, text)
        elif "<文本>" in text:
            pattern = r"<文本>(.*)"
            match = re.search(pattern, text)
        else:
            match = None

        if match:
            return match.group(1)
        else:
            return None

    def text_feature(self, query, is_cn=False):
        querys = query.split("；") if is_cn else query.split(";")
        ocr_text, query = [], []
        for q in querys:
            q = q.strip()
            if len(q):
                o = self.extract_ocr(q)
                if o is None:
                    query.append(q)
                else:
                    ocr_text.append(o)
        if is_cn:
            feature = self.cn_model.text_feature(query) if len(query) > 0 else None
        else:
            feature = self.model.text_feature(query) if len(query) > 0 else None
        return feature, ocr_text

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
        positive_text, negative_text = [], []

        is_cn = False
        if len(positive_query) != 0 and self.is_chinese(positive_query):
            if len(negative_query) == 0 or self.is_chinese(negative_query):
                is_cn = True
            else:
                assert False, "Keep the same language."

        if len(positive_query) != 0:
            positive_feature, positive_text = self.text_feature(positive_query, is_cn)

        if len(negative_query) != 0:
            negative_feature, negative_text = self.text_feature(negative_query, is_cn)

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

        if (
            positive_feature is None
            and image_feature is None
            and len(positive_text) == 0
        ):
            assert False, "No enough query input."
        else:
            infos, scores = self.search_nearest(
                image_feature,
                positive_feature,
                negative_feature,
                positive_text,
                negative_text,
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
