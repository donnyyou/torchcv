### Data Format for Semantic Segmentation

The raw data will be processed by generator shell scripts. There will be two subdirs('train' & 'val')

```
train or val dir {
    image: contains the images for train or val.
    label: contains the label png files(mode='P') for train or val.
    mask: contains the mask png files(mode='P') for train or val.
}
```


### Data Format for Instance Segmentation

The raw data will be processed by generator shell scripts. There will be two subdirs('train' & 'val')

```
train or val dir {
    image: contains the images for train or val.
    json: contains the json files for train or val.
}
```

The json format for Instance Segmentation below.

```
{
    "width": 640,
    "height": 480,
    "objects": [
        {
            "bbox": [x_left_up, y_left_up, x_right_bottom, y_right_bottom],
            "label": class_num,
            "segm": [[polygon1], [...], ...] or rle
        },
        {
            ...
        }
    ]
}
```
