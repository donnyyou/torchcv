### Data Format for Semantic Segmentation

The raw data will be processed by generator shell scripts. There will be two subdirs('train' & 'val')

```
train or val dir {
    image: contains the images for train or val.
    label: contains the label png files(mode='P') for train or val.
    mask: contains the mask png files(mode='P') for train or val.
}
```
