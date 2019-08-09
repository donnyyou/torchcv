### Data Format for Tracking

The raw data will be processed by generator shell scripts. There will be two subdirs('train' & 'val')

```
train or val dir {
    image: contains the images for train or val.
    json: contains the json files for train or val.
}

image dir contains sequences sub-directories. And each sub-directory contains all the frames of the sequence.
json dir contains sequences sub-directories. And each sub-directory contains all the jsons of the sequence.
```

The json format for Tracking below. visible=-1, invisible & unlabeled; visible=0, invisible but labeled; visible=1, visible and labeled.

```
{
    "width": 640,
    "height": 480,
    "objects": [
        {
            "id": id_num,
            "bbox": [x_left_up, y_left_up, x_right_bottom, y_right_bottom],
            "keypoints": [
                [x, y, visible],
                 ...
             ]
        },
        {
            ...
        }
    ]
}
```

