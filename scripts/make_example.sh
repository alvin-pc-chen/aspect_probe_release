python ../src/run.py make \
    -o ../data/original/sitent-ambiguous_train.tsv \
    -c ../data/cleaned/train.tsv \
    -m google-bert/bert-large-uncased \
    --hidden_dir ../data/hidden_layers/bert/train/ \
    --experiment_dir ../data/experiment_ready/bert/train/ \
    --device mps

python ../src/run.py make \
    -o ../data/original/sitent-ambiguous_test.tsv \
    -c ../data/cleaned/test.tsv \
    -m google-bert/bert-large-uncased \
    --hidden_dir ../data/hidden_layers/bert/test/ \
    --experiment_dir ../data/experiment_ready/bert/test/ \
    --device mps

python ../src/run.py make \
    -o ../data/original/sitent-ambiguous_train.tsv \
    -c ../data/cleaned/train.tsv \
    -m answerdotai/ModernBERT-large \
    --hidden_dir ../data/hidden_layers/modernbert/train/ \
    --experiment_dir ../data/experiment_ready/modernbert/train/ \
    --device mps

python ../src/run.py make \
    -o ../data/original/sitent-ambiguous_test.tsv \
    -c ../data/cleaned/test.tsv \
    -m answerdotai/ModernBERT-large \
    --hidden_dir ../data/hidden_layers/modernbert/test/ \
    --experiment_dir ../data/experiment_ready/modernbert/test/ \
    --device mps