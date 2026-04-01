#!/bin/bash
# run_single_instance.sh

# Run a single instance of the RNN to replicate initial figures of spatial selectivity
# based on the parameters found for AutoencoderPred_LN in Figure 1.

python3 trainNet.py \
    --savefolder 'replicate_fig1/' \
    --env 'MiniGrid-Empty-16x16-v0' \
    --lr 0.002 \
    --sparsity 0.5 \
    --noisestd 0.03 \
    --dropout 0.15 \
    --numepochs 80 \
    --ntimescale 2.0 \
    --hiddensize 500 \
    --seqdur 600 \
    --bias_lr 0.1 \
    --trainBias \
    --pRNNtype 'AutoencoderPred_LN' \
    --actenc 'Onehot' \
    --namext 'Onehot' \
    -s 102 \
    --no-saveTrainData

echo "Training completed. You can now run the analysis script."
