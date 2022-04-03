#!/bin/bash

cd ~/fbsource/fbcode || exit

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [local|cloud]" >&2
  exit 1
fi

location=$1

if [ $location = "local" ];then
    echo "Testing the job locally..."

    buck run @mode/dev-nosan -c fbcode.enable_gpu_sections=true on_device_ai/Tools/experimental/swin_transformer:main -- \
        --cfg on_device_ai/Tools/experimental/swin_transformer/configs/swin_mobile_patch4_window7_224.yaml \
        --exp-name swin_transformers \
        --batch-size 32 \
        --amp-opt-level O0 \
        --accumulation-steps 3

elif [ $location = "cloud" ];then
    echo "Submitting the job to cloud..."

    ## submit the job to cloud
    bento console --file on_device_ai/Tools/experimental/swin_transformer/launch_fblearner.py -- --\
        --main main \
        --cfg on_device_ai/Tools/experimental/swin_transformer/configs/swin_mobile_patch4_window7_224.yaml \
        --exp-name swin_transformers_mobile_patch8_res224_window14_mlp1_dim32_2_3_4_2_lr5e-4_epoch300_clstoken_nomix_nov3head_t1.5 \
        --batch-size 150 \
        --amp-opt-level O0 \
        --accumulation-steps 1 \
        --num-nodes 4 \
        --n-gpu-per-node 8 \
        --entitlement "gpu_prod" 
else
    echo "Do not recognize location ${location}"

fi

exit 0


