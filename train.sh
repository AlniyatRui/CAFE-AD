#!/bin/bash

export NUPLAN_MAPS_ROOT="/data/workspace/zhangjunrui/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="exp/cafe_ad"
export PYTHONPATH=$PYTHONPATH:$(pwd)
# export NUPLAN_DATA_ROOT="/data/workspace/zhangjunrui/nuplan/dataset"

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=4 \
  scenario_builder=nuplan cache.cache_path=/data/workspace/zhangjunrui/Pluto/Datasets/cache_train_100K cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=4 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  model.use_hidden_proj=true +custom_trainer.use_contrast_loss=true \
  