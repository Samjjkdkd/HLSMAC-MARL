import json
import os
import sys
import matplotlib.pyplot as plt

# ===============================
#   命令行输入 run_id
#   用法:
#   python plot_info_cmd.py 3
# ===============================
if len(sys.argv) < 2:
    print("用法: python plot_info_cmd.py <run_id>")
    print("例如: python plot_info_cmd.py 3")
    sys.exit(1)

RUN_ID = sys.argv[1]

BASE_DIR = "/data1/zhouyanju/SC2_RL/wqmix-master/results/sacred"

# ===============================

run_dir = os.path.join(BASE_DIR, RUN_ID)
info_path = os.path.join(run_dir, "info.json")

print("读取:", info_path)
if not os.path.exists(info_path):
    print(f"错误: info.json 不存在: {info_path}")
    sys.exit(1)

with open(info_path, "r") as f:
    info = json.load(f)

print("\ninfo.json 的 key 列表：")
for k in info.keys():
    print(" ", k)

# 输出目录
out_dir = os.path.join(run_dir, f"plots_run_{RUN_ID}")
os.makedirs(out_dir, exist_ok=True)
print(f"\n所有图会保存在: {out_dir}")

plotted = []

for key, val in info.items():
    if key.endswith("_T"):
        continue

    t_key = key + "_T"
    if t_key not in info:
        continue

    x = info[t_key]
    y = info[key]

    if not (isinstance(x, list) and isinstance(y, list)):
        continue
    if len(x) == 0 or len(y) == 0:
        continue
    if len(x) != len(y):
        print(f"[跳过] {key}: x,y 长度不一致 ({len(x)} vs {len(y)})")
        continue

    plt.figure()
    try:
        plt.plot(x, y)
    except Exception as e:
        print(f"[跳过] 画 {key} 时出错：{e}")
        plt.close()
        continue

    plt.xlabel("Timestep")
    plt.ylabel(key)
    plt.title(f"{key} (run {RUN_ID})")
    plt.grid(True)
    plt.tight_layout()

    safe_key = key.replace("/", "_")
    out_path = os.path.join(out_dir, f"{safe_key}_run_{RUN_ID}.png")
    plt.savefig(out_path)
    plt.close()

    plotted.append(key)
    print(f"[已画图] {key} -> {out_path}")

print("\n共生成图像数量:", len(plotted))

if plotted:
    print("主要关注以下指标（若存在）：")
    for k in ["reward_mean", "test_reward_mean", "battle_won_mean", "test_battle_won_mean",
              "loss", "qmix_loss", "central_loss", "epsilon", "ep_length_mean"]:
        if k in plotted:
            print("  -", k)
else:
    print("没有成功画出任何图，请检查 info.json 内容。")