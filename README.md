# CV_Proyecto3

Installation instructions:

git clone https://github.com/perazad/CV_Proyecto3

cd CV_Proyecto3

cd build

chmod 777 proyecto03

cd ..

chmod 777 proyecto03.sh

download https://drive.google.com/open?id=1l-LvoaAW-jem_pdMoKKQ3fvUvjx02AWH

unzip butterflies_videos.zip

download https://drive.google.com/open?id=1wwF24RBhVrCrhpTbT7OYBtrMtB7DNwly

cd nets/models/

unzip models.zip

Execution instructions:

./proyecto03.sh

Parameters modification:

In proyecto03.sh you can change several parameters

TRACKERIDX=8(default) //0-BOOSTING, 1-MIL, 2-KCF, 3-TLD, 4-MEDIANFLOW, 5-GOTURN(Code), 6-MOSSE, 7-CSRT, 8-ALL

BTFLYIDX=2(default) //Index of butterfly to track.

CAMIDX=0(default) //Index of camera 0, 1, 2

CAFFE_MODEL=nets/models/pretrained_model/tracker.caffemodel   //Pretrained model using Imagenet, Alov databases and 500k iterations

CAFFE_MODEL2=nets/models/trained_model/tracker_train.caffemodel   //Trained model using butterflies only videos and images and 2K iterations

Build Instructions:

cd build

cmake ..

make

Prerequisites:

OpenCV 3.4.1 or higher

Caffe
