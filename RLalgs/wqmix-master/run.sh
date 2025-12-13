CUDA_VISIBLE_DEVICES=7 \
python3 src/main.py \
--config=ow_qmix \
--env-config=sc2te \
with \
w=0.5 \
central_mixer=atten \
epsilon_anneal_time=1000000 \
name=ow_qmix_bs32_fkwz \
env_args.map_name=fkwz_te \