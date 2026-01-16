 for seed in 42 123 456
do
    echo "Running MOBILE with seed $seed"
    python train.py \
    --dataset-path "/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl" \
    --device cuda:6 \
    --seed $seed \
    --task walker2d-medium-expert-v2  
done

