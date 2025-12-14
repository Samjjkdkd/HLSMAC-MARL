CUDA_VISIBLE_DEVICES=6 \
python3 src/main.py \
--config=qatten \
--env-config=sc2te \
with \
epsilon_anneal_time=100000 \
t_max=205000 \
name=qatten_bs128_cr_dhls \
env_args.map_name=dhls_te \