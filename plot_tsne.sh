#!/bin/bash
python plot_tsne.py \
    --task hopper-medium-expert-v2 \
    --dataset-path /public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl \
    --policy "MOBILE=log/hopper-medium-expert-v2/mobile&penalty_coef=1.5&rollout_length=5/seed_456&timestamp_26-0119-074428/model/policy.pth" \
    --deterministic \
    --device cuda:7