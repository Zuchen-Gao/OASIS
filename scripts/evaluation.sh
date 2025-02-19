cd ..
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_name_or_path=$1
short_message=$2
use_valid=$3
if [ "$short_message" == "" ]; then
    short_message=$(basename "$model_name_or_path")
fi
if [ "$use_valid" -eq 1 ]; then
    valid_arg="--use_valid=true"
else
    valid_arg="--use_valid=false"
fi

timestamp=$(date "+%Y%m%d-%H%M%S")
eval_batch_size=2

log_path="./logs"

python run_evaluation.py \
    --model_name_or_path $model_name_or_path \
    --eval_batch_size $eval_batch_size \
    $valid_arg \
    &> ${log_path}/nl2code/${short_message}_${timestamp}result.log