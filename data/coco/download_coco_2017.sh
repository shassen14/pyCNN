#!/usr/bin/env bash

# This will check this file's current directory, attempt to download,
# abstract, and create directories for the COCO dataset
SCRIPT_DIR=$(realpath $(dirname $0))

TRAIN_ZIP="coco_train2017.zip"
TRAIN_DIR="train2017"
VAL_ZIP="coco_val2017.zip"
VAL_DIR="val2017"
ANN_ZIP="coco_ann2017.zip"
ANN_DIR="annotations"

# Training data
if [ ! -d $SCRIPT_DIR/$TRAIN_DIR ] && [ ! -f $SCRIPT_DIR/$TRAIN_ZIP ]; then
  echo "2017 COCO training zip downloading."
  wget http://images.cocodataset.org/zips/train2017.zip -O $SCRIPT_DIR/$TRAIN_ZIP
  echo "Unzipping training data to $SCRIPT_DIR/$TRAIN_DIR."
  unzip $SCRIPT_DIR/$TRAIN_ZIP -d $SCRIPT_DIR
elif [ ! -d $SCRIPT_DIR/$TRAIN_DIR ] && [ -f $SCRIPT_DIR/$TRAIN_ZIP ]; then
  echo "2017 COCO training zip already downloaded. Unzipping training data to $SCRIPT_DIR/$TRAIN_DIR"
  unzip $SCRIPT_DIR/$TRAIN_ZIP -d $SCRIPT_DIR
else
  echo "2017 COCO training dataset already acquired."
fi

# Validation data
if [ ! -d $SCRIPT_DIR/$VAL_DIR ] && [ ! -f $SCRIPT_DIR/$VAL_ZIP ]; then
  echo "2017 COCO validation zip downloading."
  wget http://images.cocodataset.org/zips/val2017.zip -O $SCRIPT_DIR/$VAL_ZIP
  echo "Unzipping validation data to $SCRIPT_DIR/$VAL_DIR"
  unzip $SCRIPT_DIR/$VAL_ZIP -d $SCRIPT_DIR
elif [ ! -d $SCRIPT_DIR/$VAL_DIR ] && [ -f $SCRIPT_DIR/$VAL_ZIP ]; then
  echo "2017 COCO validation zip already downloaded. Unzipping validation data to $SCRIPT_DIR/$VAL_DIR"
  unzip $SCRIPT_DIR/$VAL_ZIP -d $SCRIPT_DIR
else
  echo "2017 COCO validation dataset already acquired."
fi

# Annotation data
if [ ! -d $SCRIPT_DIR/$ANN_DIR ] && [ ! -f $SCRIPT_DIR/$ANN_ZIP ]; then
  echo "2017 COCO annotation zip downloading."
  wget http://images.cocodataset.org/zips/annotations_trainval2017.zip -O $SCRIPT_DIR/$ANN_ZIP
  echo "Unzipping annotation data to $SCRIPT_DIR/$ANN_DIR"
  unzip $SCRIPT_DIR/$ANN_ZIP -d $SCRIPT_DIR
elif [ ! -d $SCRIPT_DIR/$ANN_ZIP ] && [ -f $SCRIPT_DIR/$ANN_ZIP ]; then
  echo "2017 COCO annotation zip already downloaded. Unzipping annotation data to $SCRIPT_DIR/$ANN_DIR"
  unzip $SCRIPT_DIR/$ANN_ZIP -d $SCRIPT_DIR
else
  echo "2017 COCO annotation dataset already acquired."
fi
