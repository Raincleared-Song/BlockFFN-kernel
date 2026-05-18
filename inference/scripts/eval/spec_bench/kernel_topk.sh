export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
Model_id="blockffn-v2"
Bench_name="spec_bench"
export BLOCKFFN_ROUTER_TOPK=${BLOCKFFN_ROUTER_TOPK:-36}

python3 evaluation/inference_baseline.py \
    $@ \
    --cuda-graph \
    --model-id $Model_id/kernel_topk_${BLOCKFFN_ROUTER_TOPK} \
    --memory-limit 0.5 \
    --bench-name $Bench_name \
    --dtype "bfloat16" \
    --use-kernel \
