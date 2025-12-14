CUDA_VISIBLE_DEVICES=5 python3 src/main.py \
--config=qatten \
--env-config=sc2te \
with \
evaluate=True \
use_tensorboard=False \
save_replay=True \
local_results_path="eval_results" \
checkpoint_path="/data1/zhouyanju/SC2_RL/wqmix-master/results/models/qatten_bs128_gmzz__2025-11-29_23-12-04" \
env_args.map_name=gmzz_te 