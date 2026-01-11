CUDA_VISIBLE_DEVICES=6 \
python3 src/main.py \
--config=qatten \
--env-config=sc2te \
with \
runner="parallel" \
batch_size_run=8 \
training_iters=8 \
buffer_size=40000 \
epsilon_anneal_time=1000000 \
t_max=2050000 \
n_step=1 \
lambda=0.9 \
name=wwjz_resq_bs128_lambda9 \
use_resq=true \
env_args.map_name=wwjz_te