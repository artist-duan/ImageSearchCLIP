# ImageSearchCLIP
image search using CLIP feature.

refer to [clip-image-search](https://github.com/atarss/clip-image-search.git).

## 1. Installation
```
pip3 install -r requirements.txt
```
```
clip: https://github.com/openai/CLIP
```

## 1. Extract image feature

modify the IMAGE_PATH in [config.py](config.py) and run

```
python3 extract_images_features.py
```

## 2. Demo 
```
python3 app.py
```

- positive prompt
![](./statics/text_query.png)

- negative prompt: must be premised on positive/image prompt
![](./statics/negative_query.png)

- image prompt
![](./statics/image_query.png)


## 3. Hyperparameters
- image feature: 768
- image enter
  - filename, extension, height, width, filesize, createtime, clipfeature
  - memory-3.5kB/per(clip/cnclip feature)
  - time-2.7w/1h
- image-image threshold: 0.6
- text-image threshold: p-0.22/n-0.18
- cn text-image threshold: p-0.22/n-0.18
- match: cosine


## 4. TODO
- [x] extract images feature(image enter)
- [x] prompt(positive/negative)/image search
- [x] positive/negative word-segmentation use ";"(e.g. yellow; person;)
- [x] chinese: 中文效果没有英文效果好
- [ ] match algorithm
<!-- - [ ] OCR/chinese-OCR -->