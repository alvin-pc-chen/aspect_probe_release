python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/linear_example.log \
    probe \
    --results_path ../model_outputs/results/linear_example.csv \
    --heatmap_path ../model_outputs/heatmaps/linear_example.png \
    --train_dir ../data/experiment_ready/train \
    --test_dir ../data/experiment_ready/test \
    --exp_name linear_example \
    --device mps \
    --probe linear \
    --epochs 32 \
    --train_batch_size 16 \
    --learn_rate 0.01 \
    --test_batch_size 16 \
