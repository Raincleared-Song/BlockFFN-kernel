base_file="data/spec_bench/model_answer/blockffn-v2/baseline.jsonl"
kernel_file="data/spec_bench/model_answer/blockffn-v2/kernel.jsonl"

echo "=== kernel vs. baseline ==="
python evaluation/spec_bench/speed.py \
    --file-path $kernel_file \
    --base-path $base_file \
    $@