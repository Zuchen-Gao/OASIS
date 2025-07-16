cd ..
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_name_or_path=$1
short_message=$2

timestamp=$(date "+%Y%m%d-%H%M%S")
eval_batch_size=16

log_path="./logs"
mkdir -p ${log_path}/code2code

python run_evaluation_code2code.py \
    --model_name_or_path $model_name_or_path \
    --eval_batch_size $eval_batch_size \
    &> ${log_path}/code2code/${short_message}_${timestamp}result.log
