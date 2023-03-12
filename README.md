<h1 align="center">sketchy vision</h1>

<br>

<div align="center">
    <a href="https://github.com/SkalskiP">
        <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/github.png" width="4%"/>
    </a>
    <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/transparent.png" width="3%"/>
    <a href="https://twitter.com/skalskip92">
        <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/twitter.png" width="4%"/>
    </a>
    <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/transparent.png" width="3%"/>
    <a href="https://linkedin.com/in/piotr-skalski-36b5b4122">
        <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/linkedin.png" width="4%"/>
    </a>
    <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/transparent.png" width="3%"/>
    <a href="https://kaggle.com/skalskip">
        <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/kaggle.png" width="4%"/>
    </a>
    <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/transparent.png" width="3%"/>
    <a href="https://skalskip.medium.com/">
        <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/medium.png" width="4%" />
    </a>
    <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/transparent.png" width="3%"/>
    <a href="https://youtu.be/AWjKfjDGiYE">
        <img src="https://raw.githubusercontent.com/SkalskiP/SkalskiP/master/icons/youtube.png" width="4%" />
    </a>
</div>

## 👋 hello

Each week I create sketches covering key Computer Vision concepts. If you want to learn more about CV stick around!

<br>

![title](https://i.imgur.com/cf4UyDs.png)

<details close>
<summary>👆 click to read code snippet</summary>

```python
def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_a[:, None] + area_b - area_inter)
```

</details>

![title](https://i.imgur.com/6jmAmlX.jpg)

<details close>
<summary>👆 click to read code snippet</summary>

```python
def non_max_suppression(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]
```

</details>
