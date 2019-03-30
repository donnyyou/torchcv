### Data Format for GANs

The raw data will be processed by generator shell scripts. There will be two subdirs('train' & 'val')


```
train or val dir {
    imageA: contains the A images for train or val.
    imageB: contains the B images for train or val.
    labelA.json: contains the A json files for train or val.(optional)
    labelB.json: contains the B json files for train or val.(optional)
}
```

The json format for GAN below.

The label json format for Image Classification below.

```
[
    {
        "image_path": "imageA/image_name",
        "label": class_num,
        "bbox": [x_left_up, y_left_up, x_right_bottom, y_right_bottom],
        "kpts": [
            [x, y, visible],
             ...
        ]
    },
    {
        ...
    }
]
```
