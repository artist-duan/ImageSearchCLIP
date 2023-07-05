#!/usr/bin/env python3
import cv2
import torch
import logging
import numpy as np
from functools import lru_cache
from paddleocr import PaddleOCR
from PIL import Image, ImageFont, ImageDraw


# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
class PaddleOCR_Model:
    def __init__(self) -> None:
        # need to run only once to download and load model into memory
        self.model = PaddleOCR(use_angle_cls=True)
        # self.model = PaddleOCR(use_angle_cls=True, lang='ch')
        # self.model = PaddleOCR(use_angle_cls=True, lang='en')

    def validate_str(self, s):
        if len(s) > 2 and all(ord(c) < 128 for c in s):  # english
            return True
        elif len(s) > 1 and any(ord(c) >= 128 for c in s):  # chinese
            return True
        else:
            return False

    def det_ocr(self, image):
        results = self.model.ocr(image, cls=True)
        boxes, texts, scores = [], [], []
        for res in results:
            for line in res:
                if not self.validate_str(line[1][0]):
                    continue
                boxes.append(line[0])
                texts.append(line[1][0])
                scores.append(line[1][1])
        return texts, boxes, scores

    def vis_ocr(self, image, texts, boxes, scores):
        for i in range(len(texts)):
            text, box, score = texts[i], boxes[i], scores[i]
            box = np.array(box, dtype=np.int32)
            for j in range(4):
                cv2.line(
                    image,
                    (box[j % 4, 0], box[j % 4, 1]),
                    (box[(j + 1) % 4, 0], box[(j + 1) % 4, 1]),
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            str_ = f"{i+1}.{text}, {score:3f}"
            # cv2.putText(image, str_, (box[0, 0], box[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, lineType=cv2.LINE_AA)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("NotoSansCJK-Bold.ttc", 15)
            draw.text((box[0, 0], box[0, 1]), str_, font=font, fill=(255, 0, 0, 0))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image


@lru_cache(maxsize=1)
def get_model() -> PaddleOCR_Model:
    model = PaddleOCR_Model()
    logging.info(f"load PaddleOCR-Model.")
    return model


if __name__ == "__main__":
    model = get_model()

    if 0:
        path = "./examples/en_ocr.jpg"
        # path = './examples/ch_ocr.jpg'
        image = cv2.imread(path)

        # h, w = image.shape[:2]
        # center = (w/2, h/2)
        # rotation_matrix = cv2.getRotationMatrix2D(center, np.random.randint(0,180)-90, 1.0)
        # abs_cos = abs(rotation_matrix[0,0])
        # abs_sin = abs(rotation_matrix[0,1])
        # w = int(h * abs_sin + w * abs_cos)
        # h = int(h * abs_cos + w * abs_sin)
        # rotation_matrix[0, 2] += w/2 - center[0]
        # rotation_matrix[1, 2] += h/2 - center[1]
        # image = cv2.warpAffine(image, rotation_matrix, (w, h))

        texts, boxes, scores = model.det_ocr(image)
        image = model.vis_ocr(image, texts, boxes, scores)
        cv2.imwrite(path + ".res.png", image)
        cv2.imshow("ocr", image)
        cv2.waitKey(0)

    if 1:
        print(model.validate_str("ab123"))
        print(model.validate_str("13"))
        print(model.validate_str("a1"))
        print(model.validate_str("哈哈"))
        print(model.validate_str("哈哈哈哈哈哈"))
        print(model.validate_str("哈"))
