#!/usr/bin/env bash

# This will check this file's current directory, and attempt to
# delete zip files given their supposed names
SCRIPT_DIR=$(realpath $(dirname $0))

TRAIN_ZIP="coco_train2017.zip"
VAL_ZIP="coco_val2017.zip"
ANN_ZIP="coco_ann2017.zip"

# Training data
if [ -f $SCRIPT_DIR/$TRAIN_ZIP ]; then
  echo "Deleting $SCRIPT_DIR/$TRAIN_ZIP."
  rm $SCRIPT_DIR/$TRAIN_ZIP
  echo "Deleted."
else
  echo "2017 COCO training dataset zip doesn't exist."
fi

# Validation data
if [ -f $SCRIPT_DIR/$VAL_ZIP ]; then
  echo "Deleting $SCRIPT_DIR/$VAL_ZIP."
  rm $SCRIPT_DIR/$VAL_ZIP
  echo "Deleted."
else
  echo "2017 COCO validation dataset zip doesn't exist."
fi

# Annotation data
if [ -f $SCRIPT_DIR/$ANN_ZIP ]; then
  echo "Deleting $SCRIPT_DIR/$ANN_ZIP."
  rm $SCRIPT_DIR/$ANN_ZIP
  echo "Deleted."
else
  echo "2017 COCO annotations zip doesn't exist."
fi
