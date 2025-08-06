python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/mbert_linear_16.log \
    probe \
    --results_path ../model_outputs/results/mbert_linear_16.csv \
    --heatmap_path ../model_outputs/heatmaps/mbert_linear_16.png \
    --train_dir ../data/experiment_ready/modernbert/train \
    --test_dir ../data/experiment_ready/modernbert/test \
    --exp_name mbert_linear_16 \
    --device mps \
    --probe linear \
    --epochs 16 \
    --train_batch_size 16 \
    --learn_rate 0.001 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/mbert_linear_32.log \
    probe \
    --results_path ../model_outputs/results/mbert_linear_32.csv \
    --heatmap_path ../model_outputs/heatmaps/mbert_linear_32.png \
    --train_dir ../data/experiment_ready/modernbert/train \
    --test_dir ../data/experiment_ready/modernbert/test \
    --exp_name mbert_linear_32 \
    --device mps \
    --probe linear \
    --epochs 32 \
    --train_batch_size 16 \
    --learn_rate 0.001 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/mbert_multi_16.log \
    probe \
    --results_path ../model_outputs/results/mbert_multi_16.csv \
    --heatmap_path ../model_outputs/heatmaps/mbert_multi_16.png \
    --train_dir ../data/experiment_ready/modernbert/train \
    --test_dir ../data/experiment_ready/modernbert/test \
    --exp_name mbert_multi_16 \
    --device mps \
    --probe multi \
    --epochs 16 \
    --train_batch_size 16 \
    --learn_rate 0.001 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/mbert_multi_32.log \
    probe \
    --results_path ../model_outputs/results/mbert_multi_32.csv \
    --heatmap_path ../model_outputs/heatmaps/mbert_multi_32.png \
    --train_dir ../data/experiment_ready/modernbert/train \
    --test_dir ../data/experiment_ready/modernbert/test \
    --exp_name mbert_multi_32 \
    --device mps \
    --probe multi \
    --epochs 32 \
    --train_batch_size 16 \
    --learn_rate 0.001 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/bert_linear_16.log \
    probe \
    --results_path ../model_outputs/results/bert_linear_16.csv \
    --heatmap_path ../model_outputs/heatmaps/bert_linear_16.png \
    --train_dir ../data/experiment_ready/bert/train \
    --test_dir ../data/experiment_ready/bert/test \
    --exp_name bert_linear_16 \
    --device mps \
    --probe linear \
    --epochs 16 \
    --train_batch_size 16 \
    --learn_rate 0.01 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/bert_linear_32.log \
    probe \
    --results_path ../model_outputs/results/bert_linear_32.csv \
    --heatmap_path ../model_outputs/heatmaps/bert_linear_32.png \
    --train_dir ../data/experiment_ready/bert/train \
    --test_dir ../data/experiment_ready/bert/test \
    --exp_name bert_linear_32 \
    --device mps \
    --probe linear \
    --epochs 32 \
    --train_batch_size 16 \
    --learn_rate 0.01 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/bert_multi_16.log \
    probe \
    --results_path ../model_outputs/results/bert_multi_16.csv \
    --heatmap_path ../model_outputs/heatmaps/bert_multi_16.png \
    --train_dir ../data/experiment_ready/bert/train \
    --test_dir ../data/experiment_ready/bert/test \
    --exp_name bert_multi_16 \
    --device mps \
    --probe multi \
    --epochs 16 \
    --train_batch_size 16 \
    --learn_rate 0.01 \
    --test_batch_size 16 \

python ../src/run.py \
    --log_level INFO \
    --log_file ../model_outputs/logs/bert_multi_32.log \
    probe \
    --results_path ../model_outputs/results/bert_multi_32.csv \
    --heatmap_path ../model_outputs/heatmaps/bert_multi_32.png \
    --train_dir ../data/experiment_ready/bert/train \
    --test_dir ../data/experiment_ready/bert/test \
    --exp_name bert_multi_32 \
    --device mps \
    --probe multi \
    --epochs 32 \
    --train_batch_size 16 \
    --learn_rate 0.01 \
    --test_batch_size 16 \