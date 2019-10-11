#!/bin/bash

source activate bert-tf 

CMRC_DIR="../CMRC/"

PATH_TO_BERT="../chinese_wwm_ext_L-12_H-768_A-12"
MODEL_DIR="experiments/chinese_wwm_ext"
python ./baseline/run_cmrc2018_drcd_baseline.py \
    --vocab_file=${PATH_TO_BERT}/vocab.txt \
    --bert_config_file=${PATH_TO_BERT}/bert_config.json \
    --init_checkpoint=${PATH_TO_BERT}/bert_model.ckpt \
    --do_train \
    --do_predict \
    --train_file $CMRC_DIR/cmrc2018_train.json \
    --predict_file $CMRC_DIR/cmrc2018_dev.json \
    --train_batch_size=4 \
    --num_train_epochs=2 \
    --max_seq_length=512 \
    --doc_stride=128 \
    --learning_rate=3e-5 \
    --save_checkpoints_steps=1000 \
    --output_dir=${MODEL_DIR} \
    --do_lower_case=False \
    --use_tpu=False
