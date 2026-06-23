# 13 已完成：PPO Squashed EP4 Multi-Seed

## 本节目标

验证 12 得到的候选 baseline 是否稳定。

候选配置：

- observation normalization
- tanh-squashed Gaussian policy
- update epochs: `4`
- total timesteps: `3000000`
- seeds: `0/1/2`

本节的核心问题不是继续追求单 seed 更高分，而是判断这个配置是否能作为后续 SAC/off-policy 对比的可靠 PPO baseline。

## 实验结果

| seed | evaluation mean return | evaluation std | tail rolling return | tail rolling episode length | tail value loss | tail entropy | tail approx KL | tail PPO clip fraction | tail action clip fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 716.011 | 115.490 | 617.978 | 121.450 | 144.237 | 26.640 | 0.1038 | 0.4198 | 0.0000 |
| 1 | 899.806 | 190.234 | 708.255 | 147.175 | 92.481 | 25.340 | 0.0701 | 0.4067 | 0.0000 |
| 2 | 861.418 | 122.380 | 729.721 | 147.185 | 82.554 | 25.902 | 0.0783 | 0.4448 | 0.0000 |

三 seed evaluation mean return：

- mean: `825.745`
- std across seed means: `79.160`

## 关键观察

1. 多 seed 没有崩塌

   seed `0/1/2` 的 evaluation mean return 都明显高于早期 baseline，没有出现某个 seed 固定低分或策略完全失败。

2. 动作裁剪问题被稳定解决

   三个 seed 的 tail `action_clip_fraction` 都是 `0.0000`。这说明 tanh-squashed Gaussian policy 不是只在 seed 0 上偶然有效，而是稳定解决了 raw Gaussian action 被环境边界裁剪的问题。

3. `update_epochs=4` 是当前合理选择

   相比 11 中 `update_epochs=10` 的 squashed policy，EP4 显著降低了 approx KL 和 PPO clip fraction，同时保留了更高 return。它的本质是减少同一批 rollout 被重复训练的次数，避免 PPO update 过猛。

4. 当前 baseline 仍不是“完美策略”

   PPO clip fraction 仍在 `0.40` 左右，evaluation episode return 也有波动。但对当前项目阶段来说，它已经满足“可复现、可解释、可作为后续多智能体对照组”的要求。

## Stage 2 结论

Stage 2 可以收束。

最终单智能体 baseline 配置定为：

```bash
python src/train_ppo.py \
  --seed <seed> \
  --total-timesteps 3000000 \
  --rollout-steps 2048 \
  --batch-size 256 \
  --update-epochs 4 \
  --run-name ppo_long_obsnorm_squash_ep4_seed<seed> \
  --normalize-observations \
  --squash-actions
```

保留的关键改进：

- observation normalization：改善输入尺度，降低 critic 拟合压力。
- tanh-squashed Gaussian policy：让采样动作天然落在环境动作范围内，并用 log_prob 修正保持 PPO 数学一致。
- update epochs tuning：把 `10` 降到 `4`，缓解 squashed policy 下 PPO update 过猛的问题。

放弃或暂不作为主线的方向：

- target KL early stopping：机制有效，但当前不是最强 baseline 的核心配置。
- action log std clamp：`0.5` 和 `1.0` 都导致策略退化。
- reward/return scaling、entropy schedule、value clipping、orthogonal initialization、vectorized rollout：不是废弃，只是当前已有足够清晰的 Stage 2 baseline，先进入 Stage 3。

## 下一节入口

后续进入 `notes/14_stage3_sac_entry.md`。

下一阶段先引入 SAC 作为 off-policy 强基线，与当前 PPO final baseline 对比 return、episode length 和视频行为；多智能体扩展放到更后面。
