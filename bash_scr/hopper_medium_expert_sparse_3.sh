 for seed in 42 123 456
do
    echo "Running MOBILE with seed $seed"
    python train.py \
    --dataset-path "/public/d4rl/sparse_datasets/hopper_medium_expert_sparse_73.pkl" \
    --device cuda:5 \
    --seed $seed \
    --task hopper-medium-expert-v2  
done

