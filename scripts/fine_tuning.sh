cd ..
export TOKENIZERS_PARALLELISM=false

timestamp=$(date "+%Y%m%d-%H%M%S")
OUTPUT=./logs/fine_tuning/training_${timestamp}
mkdir -p $OUTPUT

deepspeed \
    --hostfile ./config/hostfile \
    ./run_fine_tuning.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
    --query_column poss_docstring \
    --passage_column snippet_wo_comment \
    --label_column similarity \
    --ibn_w 0.98 \
    --cosine_w 0.02 \
    --mix_data \
    --max_length 1024 \
    --in_batch_num 8 \
    --pooling_strategy last \
    --pooling_layer -1 \
    --lr 5e-4 \
    --train_data_path hf_dataset \
    --train_batch_size 80 \
    --multi_node \
    --epochs 10 \
    --output_dir $OUTPUT \
    --deepspeed \
    --deepspeed_config ./config/fine_tuning_ds_config.json \
    &> $OUTPUT/fine_tuning.log

