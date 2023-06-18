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
  - memory-1.9kB/per(clip feature)
  - time-2.7w/1h
- text-image threshold: 0.2
- image-image threshold: 0.6
- match: cosine


## 4. TODO
- [x] extract images feature(image enter)
- [x] prompt(positive/negative)/image search
- [ ] negative word segmentation
- [ ] match algorithm
- [ ] Chinese
<!-- - [ ] OCR/chinese-OCR -->