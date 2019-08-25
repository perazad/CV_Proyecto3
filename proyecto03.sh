#!/bin/bash

TRACKERIDX=8
BTFLYIDX=2
CAMIDX=0
DEPLOY=nets/tracker.prototxt
CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel
CAFFE_MODEL2=nets/models/trained_model/tracker_train.caffemodel

build/proyecto03 $TRACKERIDX $BTFLYIDX $CAMIDX $DEPLOY $CAFFE_MODEL
