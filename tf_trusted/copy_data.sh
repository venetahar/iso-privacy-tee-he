#/bin/bash

TF_TRUSTED_INSALL_PATH=$1

cd $(dirname $0)/..

cp -R \
  ./tf_trusted/*.py \
  ./models \
  ./data \
  $TF_TRUSTED_INSALL_PATH/tf_trusted_custom_op/

