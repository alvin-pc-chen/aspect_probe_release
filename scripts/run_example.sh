python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/multi_32_epochs_2.log \
    probe \
    --results_path ../model_outputs/results/multi_32_epochs_2.csv \
    --heatmap_path ../model_outputs/heatmaps/multi_32_epochs_2.png \
    --train_dir ../data/experiment_ready/train \
    --test_dir ../data/experiment_ready/test \
    --exp_name multi_32_epochs_2 \
    --device mps \
    --probe multi \
    --epochs 32 \
    --train_batch_size 16 \
    --learn_rate 0.01 \
    --test_batch_size 16 \
