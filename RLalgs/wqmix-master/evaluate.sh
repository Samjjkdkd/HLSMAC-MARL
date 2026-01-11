CUDA_VISIBLE_DEVICES=0 python3 src/main.py \
--config=qatten \
--env-config=sc2te \
with \
evaluate=True \
runner=episode \
batch_size_run=1 \
use_tensorboard=False \
save_replay=True \
local_results_path="eval_results" \
checkpoint_path="/data1/zhouyanju/HLSMAC-MARL/RLalgs/wqmix-master/models/experiments/adcc/adcc_bs128_cr__2025-12-14_21-22-38" \
env_args.map_name=adcc_te