### Data Format for Image Classification

The raw data will be processed by generator shell scripts. There will be two subdirs('train' & 'val')

```
train or val dir {
    image: contains the images for train or val.
    label.json: the label file for train or val.
}
```

The label json format for Image Classification below.

```
[
    {
        "image_path": "image/image_name",
        "label": class_num
    },
    {
        ...
    }
]
```
