## FasterRCNN-Train
caffe FasterRCNN train code, use [cc5.0](https://github.com/dlunion/CC5.0)
code is written by c++

The offical Faster R-CNN code written in python is [here](https://github.com/rbgirshick/py-faster-rcnn)

## implement
implement four special layer below. ~~The whole train code is not tested yet.~~ 
The whole code is already tested.

- anchor target layer
- roi data layer
- proposal layer
- proposal target layer

## How to use
Download voc data or your own data. And put the path in config.datapath and config.xmlpath.

Visualization code is already added, so the result can be seen while training.

All the configuration can be adjusted in the common.h.

Just change the code as you want.