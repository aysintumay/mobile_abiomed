 for seed in 42 123 456
do
    echo "Running MOBILE with seed $seed"
    python train.py \
    --dataset-path "/public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_78.pkl" \
    --device cuda:3 \
    --seed $seed \
    --task halfcheetah-medium-expert-v2  
done

