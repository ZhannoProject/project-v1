# EEG 去伪影实验日志 · 数据索引

整理日期 2026-11-16。返回[实验日志正文](experiment-log-2026-11-16.md)。
每行链接指向原始记录。训练曲线、被试逐项数据、每 hop 时间序列和环境记录保留在原文件。“—”表示未记录或不适用。

## A. 模型训练与验证

项目 runs/ 有 103 个含 best.pt 的目录。指标来自 history.json 最低验证损失的 epoch。stale 行保留历史数值供追溯，此时权重与历史不齐。interrupted 行显示已有 checkpoint 进度。验证核验列使用 GUI 对旧协议的来源一致性检查。256* 是旧记录缺少显式训练采样率时 GUI 按数据原生采样率作的推定。

### 常规训练（47 条）

| 运行 / 原始历史 | 训练日期¹ | 伪影 | seed | 宽度 | 参数 | 内部归一化 | 输入归一化 | Hz | 损失 | 已记 epoch / 最佳 / 最后 | RRMSE_t | RRMSE_f | CC | 状态 / 旧协议验证核验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ctrl_mla_both_bi](../runs/ctrl_mla_both_bi/history.json) | 2026-09-22* | BOTH | — | 64 | 192633 | batchnorm | epoch | 256* | l1_combined | 40 / 40 / 40 | 0.7722 | 0.6951 | 0.6493 | complete / — |
| [ctrl_mla_both_causal](../runs/ctrl_mla_both_causal/history.json) | 2026-09-22* | BOTH | — | 64 | 192633 | batchnorm | epoch | 256* | l1_combined | 40 / 34 / 40 | 0.7632 | 0.6895 | 0.6605 | complete / — |
| [tcn_bi_both](../runs/tcn_bi_both/history.json) | 2026-09-25* | BOTH | 42 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6093 | 0.5876 | 0.7931 | complete / 通过 |
| [tcn_bi_both_250hz_s54](../runs/tcn_bi_both_250hz_s54/history.json) | 2026-11-10 | BOTH | 54 | 64 | 117068 | ChannelNorm | epoch | 250 | mse | 200 / 182 / 200 | 0.6173 | 0.5915 | 0.7871 | complete / — |
| [tcn_bi_both_s54](../runs/tcn_bi_both_s54/history.json) | 2026-10-21 | BOTH | 54 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 182 / 200 | 0.6111 | 0.5881 | 0.7908 | complete / 通过 |
| [tcn_bi_both_s67](../runs/tcn_bi_both_s67/history.json) | 2026-10-13 | BOTH | 67 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6105 | 0.5800 | 0.7913 | complete / 通过 |
| [tcn_bi_emg](../runs/tcn_bi_emg/history.json) | 2026-09-25* | EMG | 42 | 106 | 320672 | batchnorm | epoch | 256* | mse | 150 / 148 / 150 | 0.5284 | 0.4797 | 0.8426 | complete / — |
| [tcn_bi_eog](../runs/tcn_bi_eog/history.json) | 2026-09-25* | EOG | 42 | 64 | 118358 | batchnorm | epoch | 256* | mse | 150 / 149 / 150 | 0.3715 | 0.4366 | 0.9246 | complete / — |
| [tcn_both](../runs/tcn_both/history.json) | 2026-10-13 | BOTH | 67 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 155 / 200 | 0.7240 | 0.6579 | 0.6922 | complete / 通过 |
| [tcn_both_s42](../runs/tcn_both_s42/history.json) | 2026-10-21 | BOTH | 42 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7205 | 0.6607 | 0.6979 | complete / 通过 |
| [tcn_both_s67](../runs/tcn_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 118358 | batchnorm | epoch | 250 | mse | 200 / 184 / 200 | 0.7262 | 0.6602 | 0.6902 | complete / 通过 |
| [tcn_channelnorm_both_250hz](../runs/tcn_channelnorm_both_250hz/history.json) | 2026-11-05 | BOTH | 54 | 64 | 117068 | ChannelNorm | epoch | 250 | mse | 200 / 182 / 200 | 0.7286 | 0.6612 | 0.6887 | complete / — |
| [tcn_channelnorm_both_250hz)_s67](../runs/tcn_channelnorm_both_250hz%29_s67/history.json) | 2026-11-05 | BOTH | 67 | 64 | 117068 | ChannelNorm | epoch | 250 | mse | 200 / 184 / 200 | 0.7290 | 0.6617 | 0.6880 | complete / — |
| [tcn_emg](../runs/tcn_emg/history.json) | 2026-09-25* | EMG | 42 | 106 | 320672 | batchnorm | epoch | 256* | mse | 150 / 133 / 150 | 0.6173 | 0.5411 | 0.7815 | complete / — |
| [tcn_eog](../runs/tcn_eog/history.json) | 2026-09-25* | EOG | 42 | 64 | 118358 | batchnorm | epoch | 256* | mse | 150 / 132 / 150 | 0.4954 | 0.4980 | 0.8641 | complete / — |
| [tcn_mha_both](../runs/tcn_mha_both/history.json) | 2026-05-20* | BOTH | — | — | 169081 | batchnorm | epoch | 256* | mse | 150 / 140 / 150 | 0.7050 | 0.6464 | 0.7140 | complete / — |
| [tcn_mha_emg](../runs/tcn_mha_emg/best.pt) | 2026-05-21* | EMG | — | — | 169081 | batchnorm | epoch | 256* | — | — / — / 146 | — | — | — | interrupted / — |
| [tcn_mha_eog](../runs/tcn_mha_eog/history.json) | 2026-05-22* | EOG | — | — | 169081 | batchnorm | epoch | 256* | — | 150 / 149 / 31 | 0.4479 | 0.4511 | 0.8888 | stale / — |
| [tcn_mha_par_bi_emg](../runs/tcn_mha_par_bi_emg/history.json) | 2026-09-24* | EMG | — | 64 | 339545 | batchnorm | epoch | 256* | l1_combined | 100 / 35 / 100 | 0.6491 | 0.5663 | 0.7607 | complete / — |
| [tcn_mha_par_both](../runs/tcn_mha_par_both/history.json) | 2026-05-20* | BOTH | — | — | 239577 | batchnorm | epoch | 256* | mse | 100 / 100 / 100 | 0.7131 | 0.6537 | 0.7066 | complete / — |
| [tcn_mha_par_emg](../runs/tcn_mha_par_emg/history.json) | 2026-05-20* | EMG | — | — | 239577 | batchnorm | epoch | 256* | mse | 100 / 97 / 100 | 0.6113 | 0.5355 | 0.7859 | complete / — |
| [tcn_mha_par_eog](../runs/tcn_mha_par_eog/best.pt) | 2026-05-20* | EOG | — | — | 239577 | batchnorm | epoch | 256* | — | — / — / 77 | — | — | — | interrupted / — |
| [tcn_mla_bi_both](../runs/tcn_mla_bi_both/history.json) | 2026-09-25* | BOTH | 42 | 64 | 192633 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6160 | 0.5783 | 0.7874 | complete / 通过 |
| [tcn_mla_bi_both_s67](../runs/tcn_mla_bi_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 192633 | batchnorm | epoch | 250 | mse | 200 / 184 / 200 | 0.6183 | 0.5757 | 0.7840 | complete / 通过 |
| [tcn_mla_bi_emg](../runs/tcn_mla_bi_emg/history.json) | 2026-09-25* | EMG | 42 | 64 | 266073 | batchnorm | epoch | 256* | mse | 150 / 149 / 150 | 0.5318 | 0.4700 | 0.8378 | complete / — |
| [tcn_mla_bi_eog](../runs/tcn_mla_bi_eog/history.json) | 2026-09-25* | EOG | 42 | 64 | 192633 | batchnorm | epoch | 256* | mse | 150 / 149 / 150 | 0.3774 | 0.4075 | 0.9198 | complete / — |
| [tcn_mla_both](../runs/tcn_mla_both/history.json) | 2026-10-13 | BOTH | 67 | 64 | 192633 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.7106 | 0.6420 | 0.7064 | complete / 通过 |
| [tcn_mla_both_s67](../runs/tcn_mla_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 192633 | batchnorm | epoch | 250 | mse | 200 / 184 / 200 | 0.7130 | 0.6439 | 0.7042 | complete / 通过 |
| [tcn_mla_emg](../runs/tcn_mla_emg/history.json) | 2026-09-25* | EMG | 42 | 64 | 192633 | batchnorm | epoch | 256* | mse | 150 / 139 / 150 | 0.6135 | 0.5342 | 0.7834 | complete / — |
| [tcn_mla_eog](../runs/tcn_mla_eog/history.json) | 2026-09-25* | EOG | 42 | 64 | 192633 | batchnorm | epoch | 256* | mse | 1 / 1 / 1 | 1.7888 | 3.9794 | 0.0161 | complete / — |
| [tcn_mla_par_bi_both](../runs/tcn_mla_par_bi_both/history.json) | 2026-09-24* | BOTH | — | 64 | 319257 | batchnorm | epoch | 256* | l1_combined | 100 / 40 / 31 | 0.7637 | 0.6882 | 0.6613 | stale / — |
| [tcn_mla_par_bi_emg](../runs/tcn_mla_par_bi_emg/history.json) | 2026-09-25* | EMG | 42 | 64 | 319257 | batchnorm | epoch | 256* | mse | 150 / 142 / 150 | 0.5298 | 0.4702 | 0.8391 | complete / — |
| [tcn_only_eog](../runs/tcn_only_eog/history.json) | 2026-05-13* | EOG | — | — | 164637 | 未记录 | epoch | 256* | — | 100 / 96 / 100 | 0.5506 | 0.5449 | 0.8262 | complete / — |
| [tcn_se_bi_both](../runs/tcn_se_bi_both/history.json) | 2026-10-21 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.5956 | 0.5745 | 0.7999 | complete / 通过 |
| [tcn_se_bi_both_250hz_s42](../runs/tcn_se_bi_both_250hz_s42/history.json) | 2026-11-11 | BOTH | 42 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 179 / 200 | 0.6075 | 0.5861 | 0.7931 | complete / — |
| [tcn_se_bi_both_250hz_s54](../runs/tcn_se_bi_both_250hz_s54/history.json) | 2026-11-10 | BOTH | 54 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 182 / 200 | 0.6088 | 0.5787 | 0.7913 | complete / — |
| [tcn_se_bi_both_250hz_s67](../runs/tcn_se_bi_both_250hz_s67/history.json) | 2026-11-11 | BOTH | 67 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 186 / 200 | 0.6065 | 0.5700 | 0.7926 | complete / — |
| [tcn_se_bi_both_s54](../runs/tcn_se_bi_both_s54/history.json) | 2026-10-21 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.5967 | 0.5753 | 0.7983 | complete / 通过 |
| [tcn_se_bi_both_s67](../runs/tcn_se_bi_both_s67/history.json) | 2026-10-21 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.5928 | 0.5610 | 0.8005 | complete / 通过 |
| [tcn_se_both](../runs/tcn_se_both/history.json) | 2026-10-21 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7082 | 0.6453 | 0.7107 | complete / 通过 |
| [tcn_se_both_250hz](../runs/tcn_se_both_250hz/history.json) | 2026-11-04 | BOTH | 42 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 168 / 200 | 0.7139 | 0.6509 | 0.7061 | complete / — |
| [tcn_se_both_250hz_s54](../runs/tcn_se_both_250hz_s54/history.json) | 2026-11-04 | BOTH | 54 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 182 / 200 | 0.7172 | 0.6527 | 0.7009 | complete / — |
| [tcn_se_both_250hz_s67](../runs/tcn_se_both_250hz_s67/history.json) | 2026-11-04 | BOTH | 67 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 184 / 200 | 0.7151 | 0.6423 | 0.7023 | complete / — |
| [tcn_se_both_s54](../runs/tcn_se_both_s54/history.json) | 2026-10-21 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 177 / 200 | 0.7121 | 0.6509 | 0.7055 | complete / 通过 |
| [tcn_se_both_s67](../runs/tcn_se_both_s67/history.json) | 2026-10-21 | BOTH | 67 | 48 | 70420 | batchnorm | epoch | 256* | mse | 200 / 193 / 200 | 0.7133 | 0.6466 | 0.7040 | complete / 通过 |
| [tcn_se_both_s67_h64](../runs/tcn_se_both_s67_h64/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.7110 | 0.6416 | 0.7059 | complete / 通过 |
| [tcn_se_eog_250hz_s54](../runs/tcn_se_eog_250hz_s54/history.json) | 2026-11-05 | EOG | 54 | 64 | 122548 | ChannelNorm | epoch | 250 | mse | 200 / 181 / 200 | 0.4768 | 0.4781 | 0.8737 | complete / — |

### SE 有限前瞻（27 条）

| 运行 / 原始历史 | 训练日期¹ | 伪影 | seed | 宽度 | 参数 | 内部归一化 | 输入归一化 | Hz | 损失 | 已记 epoch / 最佳 / 最后 | RRMSE_t | RRMSE_f | CC | 状态 / 旧协议验证核验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [tcn_se_la124_both](../runs/tcn_se_la124_both/history.json) | 2026-10-23 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6029 | 0.5808 | 0.7967 | complete / 通过 |
| [tcn_se_la124_both_s54](../runs/tcn_se_la124_both_s54/history.json) | 2026-10-23 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6037 | 0.5801 | 0.7955 | complete / 通过 |
| [tcn_se_la124_both_s67](../runs/tcn_se_la124_both_s67/history.json) | 2026-10-23 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6009 | 0.5665 | 0.7970 | complete / 通过 |
| [tcn_se_la16_both](../runs/tcn_se_la16_both/history.json) | 2026-10-22 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6264 | 0.5932 | 0.7804 | complete / 通过 |
| [tcn_se_la16_both_s54](../runs/tcn_se_la16_both_s54/history.json) | 2026-10-22 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 181 / 200 | 0.6275 | 0.5995 | 0.7787 | complete / 通过 |
| [tcn_se_la16_both_s67](../runs/tcn_se_la16_both_s67/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6264 | 0.5905 | 0.7796 | complete / 通过 |
| [tcn_se_la186_both](../runs/tcn_se_la186_both/history.json) | 2026-10-23 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6656 | 0.6131 | 0.7436 | complete / 通过 |
| [tcn_se_la186_both_s54](../runs/tcn_se_la186_both_s54/history.json) | 2026-10-23 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6684 | 0.6161 | 0.7408 | complete / 通过 |
| [tcn_se_la186_both_s67](../runs/tcn_se_la186_both_s67/history.json) | 2026-10-23 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 198 / 200 | 0.6677 | 0.6088 | 0.7411 | complete / 通过 |
| [tcn_se_la1_both](../runs/tcn_se_la1_both/history.json) | 2026-10-22 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6866 | 0.6288 | 0.7323 | complete / 通过 |
| [tcn_se_la1_both_s54](../runs/tcn_se_la1_both_s54/history.json) | 2026-10-22 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 177 / 200 | 0.6900 | 0.6301 | 0.7280 | complete / 通过 |
| [tcn_se_la1_both_s67](../runs/tcn_se_la1_both_s67/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6889 | 0.6230 | 0.7278 | complete / 通过 |
| [tcn_se_la2_both](../runs/tcn_se_la2_both/history.json) | 2026-10-22 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6786 | 0.6238 | 0.7393 | complete / 通过 |
| [tcn_se_la2_both_s54](../runs/tcn_se_la2_both_s54/history.json) | 2026-10-22 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 177 / 200 | 0.6821 | 0.6254 | 0.7353 | complete / 通过 |
| [tcn_se_la2_both_s67](../runs/tcn_se_la2_both_s67/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6813 | 0.6172 | 0.7346 | complete / 通过 |
| [tcn_se_la32_both](../runs/tcn_se_la32_both/history.json) | 2026-10-22 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6136 | 0.5879 | 0.7900 | complete / 通过 |
| [tcn_se_la32_both_s54](../runs/tcn_se_la32_both_s54/history.json) | 2026-10-22 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6144 | 0.5937 | 0.7884 | complete / 通过 |
| [tcn_se_la32_both_s67](../runs/tcn_se_la32_both_s67/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6112 | 0.5820 | 0.7907 | complete / 通过 |
| [tcn_se_la4_both](../runs/tcn_se_la4_both/history.json) | 2026-10-22 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6746 | 0.6212 | 0.7427 | complete / 通过 |
| [tcn_se_la4_both_s54](../runs/tcn_se_la4_both_s54/history.json) | 2026-10-22 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 177 / 200 | 0.6776 | 0.6245 | 0.7388 | complete / 通过 |
| [tcn_se_la4_both_s67](../runs/tcn_se_la4_both_s67/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6768 | 0.6172 | 0.7386 | complete / 通过 |
| [tcn_se_la62_both](../runs/tcn_se_la62_both/history.json) | 2026-10-23 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6040 | 0.5780 | 0.7960 | complete / 通过 |
| [tcn_se_la62_both_s54](../runs/tcn_se_la62_both_s54/history.json) | 2026-10-23 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6060 | 0.5850 | 0.7941 | complete / 通过 |
| [tcn_se_la62_both_s67](../runs/tcn_se_la62_both_s67/history.json) | 2026-10-23 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6039 | 0.5706 | 0.7955 | complete / 通过 |
| [tcn_se_la8_both](../runs/tcn_se_la8_both/history.json) | 2026-10-22 | BOTH | 42 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.6533 | 0.6084 | 0.7599 | complete / 通过 |
| [tcn_se_la8_both_s54](../runs/tcn_se_la8_both_s54/history.json) | 2026-10-22 | BOTH | 54 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 199 / 200 | 0.6564 | 0.6138 | 0.7557 | complete / 通过 |
| [tcn_se_la8_both_s67](../runs/tcn_se_la8_both_s67/history.json) | 2026-10-22 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 256* | mse | 200 / 184 / 200 | 0.6559 | 0.6068 | 0.7559 | complete / 通过 |

### TCN 感受野（24 条）

| 运行 / 原始历史 | 训练日期¹ | 伪影 | seed | 宽度 | 参数 | 内部归一化 | 输入归一化 | Hz | 损失 | 已记 epoch / 最佳 / 最后 | RRMSE_t | RRMSE_f | CC | 状态 / 旧协议验证核验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [tcn_rf125_both](../runs/tcn_rf125_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7205 | 0.6607 | 0.6979 | complete / 通过 |
| [tcn_rf125_both_s54](../runs/tcn_rf125_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 182 / 200 | 0.7244 | 0.6626 | 0.6920 | complete / 通过 |
| [tcn_rf125_both_s64](../runs/tcn_rf125_both_s64/history.json) | 2026-10-27 | BOTH | 64 | 64 | 118358 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7250 | 0.6709 | 0.6928 | complete / 通过 |
| [tcn_rf13_both](../runs/tcn_rf13_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 42698 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7678 | 0.7194 | 0.6535 | complete / 通过 |
| [tcn_rf13_both_s54](../runs/tcn_rf13_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 42698 | batchnorm | epoch | 256* | mse | 200 / 131 / 200 | 0.7743 | 0.7249 | 0.6460 | complete / 通过 |
| [tcn_rf13_both_s64](../runs/tcn_rf13_both_s64/history.json) | 2026-10-27 | BOTH | 64 | 64 | 42698 | batchnorm | epoch | 256* | mse | 200 / 152 / 200 | 0.7721 | 0.7302 | 0.6492 | complete / 通过 |
| [tcn_rf13_both_s67](../runs/tcn_rf13_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 42698 | batchnorm | epoch | 256* | mse | 200 / 151 / 200 | 0.7723 | 0.7135 | 0.6472 | complete / 通过 |
| [tcn_rf253_both](../runs/tcn_rf253_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 143578 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7127 | 0.6494 | 0.7059 | complete / 通过 |
| [tcn_rf253_both_s54](../runs/tcn_rf253_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 143578 | batchnorm | epoch | 256* | mse | 200 / 182 / 200 | 0.7159 | 0.6515 | 0.7011 | complete / 通过 |
| [tcn_rf253_both_s64](../runs/tcn_rf253_both_s64/history.json) | 2026-10-27 | BOTH | 64 | 64 | 143578 | batchnorm | epoch | 256* | mse | 184 / 174 / 184 | 0.7154 | 0.6561 | 0.7022 | complete / — |
| [tcn_rf29_both](../runs/tcn_rf29_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 67918 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7422 | 0.6906 | 0.6769 | complete / 通过 |
| [tcn_rf29_both_s54](../runs/tcn_rf29_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 67918 | batchnorm | epoch | 256* | mse | 200 / 173 / 200 | 0.7474 | 0.6970 | 0.6702 | complete / 通过 |
| [tcn_rf29_both_s64](../runs/tcn_rf29_both_s64/history.json) | 2026-10-27 | BOTH | 64 | 64 | 67918 | batchnorm | epoch | 256* | mse | 200 / 152 / 200 | 0.7469 | 0.6977 | 0.6724 | complete / 通过 |
| [tcn_rf29_both_s67](../runs/tcn_rf29_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 67918 | batchnorm | epoch | 256* | mse | 200 / 155 / 200 | 0.7461 | 0.6862 | 0.6712 | complete / 通过 |
| [tcn_rf509_both](../runs/tcn_rf509_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 168798 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7079 | 0.6464 | 0.7111 | complete / 通过 |
| [tcn_rf509_both_s54](../runs/tcn_rf509_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 168798 | batchnorm | epoch | 256* | mse | 200 / 181 / 200 | 0.7082 | 0.6456 | 0.7092 | complete / 通过 |
| [tcn_rf5_both](../runs/tcn_rf5_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 17478 | batchnorm | epoch | 256* | mse | 200 / 111 / 200 | 0.8536 | 0.9242 | 0.5566 | complete / 通过 |
| [tcn_rf5_both_s54](../runs/tcn_rf5_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 17478 | batchnorm | epoch | 256* | mse | 200 / 131 / 200 | 0.8582 | 0.9300 | 0.5483 | complete / 通过 |
| [tcn_rf5_both_s64](../runs/tcn_rf5_both_s64/history.json) | 2026-10-27 | BOTH | 64 | 64 | 17478 | batchnorm | epoch | 256* | mse | 200 / 96 / 200 | 0.8611 | 0.9364 | 0.5462 | complete / 通过 |
| [tcn_rf5_both_s67](../runs/tcn_rf5_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 17478 | batchnorm | epoch | 256* | mse | 200 / 105 / 200 | 0.8621 | 0.9378 | 0.5466 | complete / 通过 |
| [tcn_rf61_both](../runs/tcn_rf61_both/history.json) | 2026-10-27 | BOTH | 42 | 64 | 93138 | batchnorm | epoch | 256* | mse | 200 / 174 / 200 | 0.7287 | 0.6699 | 0.6902 | complete / 通过 |
| [tcn_rf61_both_s54](../runs/tcn_rf61_both_s54/history.json) | 2026-10-27 | BOTH | 54 | 64 | 93138 | batchnorm | epoch | 256* | mse | 200 / 182 / 200 | 0.7338 | 0.6697 | 0.6836 | complete / 通过 |
| [tcn_rf61_both_s64](../runs/tcn_rf61_both_s64/history.json) | 2026-10-27 | BOTH | 64 | 64 | 93138 | batchnorm | epoch | 256* | mse | 200 / 187 / 200 | 0.7327 | 0.6813 | 0.6860 | complete / 通过 |
| [tcn_rf61_both_s67](../runs/tcn_rf61_both_s67/history.json) | 2026-10-30 | BOTH | 67 | 64 | 93138 | batchnorm | epoch | 256* | mse | 124 / 118 / 124 | 0.7329 | 0.6644 | 0.6844 | complete / — |

### 蒸馏（3 条）

| 运行 / 原始历史 | 训练日期¹ | 伪影 | seed | 宽度 | 参数 | 内部归一化 | 输入归一化 | Hz | 损失 | 已记 epoch / 最佳 / 最后 | RRMSE_t | RRMSE_f | CC | 状态 / 旧协议验证核验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [tcn_bi_both_kd](../runs/tcn_bi_both_kd/history.json) | 2026-11-03 | BOTH | 42 | 32 | 30518 | batchnorm | epoch | 256 | mse | 200 / 199 / 200 | 0.6163 | 0.5987 | 0.7882 | complete / — |
| [tcn_se_both_kd_s54](../runs/tcn_se_both_kd_s54/history.json) | 2026-11-03 | BOTH | 54 | 32 | 31978 | batchnorm | epoch | 256 | mse | 200 / 155 / 200 | 0.7185 | 0.6569 | 0.6999 | complete / — |
| [tcn_se_both_kd_s67](../runs/tcn_se_both_kd_s67/history.json) | 2026-10-29 | BOTH | 67 | 64 | 123838 | batchnorm | epoch | 250 | mse | 200 / 193 / 200 | 0.7182 | 0.6516 | 0.6992 | complete / 通过 |

### 其他控制（2 条）

| 运行 / 原始历史 | 训练日期¹ | 伪影 | seed | 宽度 | 参数 | 内部归一化 | 输入归一化 | Hz | 损失 | 已记 epoch / 最佳 / 最后 | RRMSE_t | RRMSE_f | CC | 状态 / 旧协议验证核验 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [tcn_mla_r16_emg](../runs/tcn_mla_r16_emg/history.json) | 2026-09-24* | EMG | 42 | 64 | 192633 | batchnorm | epoch | 256* | mse | 150 / 148 / 150 | 0.6130 | 0.5317 | 0.7836 | complete / — |
| [tcn_mla_r8_both](../runs/tcn_mla_r8_both/history.json) | 2026-09-22* | BOTH | — | 64 | 189465 | batchnorm | epoch | 256* | l1_combined | 100 / 77 / 100 | 0.7580 | 0.6858 | 0.6647 | complete / — |

¹ 训练日期以 GUI 记录的时间为准，日期带 * 表示以文件时间估计。
外层工作区另有一份[tcn_only_eog 早期试跑](../../runs/tcn_only_eog/history.json)，30 epoch，最佳 epoch 30，RRMSE_t 0.6553，RRMSE_f 0.6663，CC 0.7555。seed、采样率和训练配置在该历史 JSON 中均未记录。

## B. EEG-IO 实测与回放

27 份结果文件含 38 条模型结果，共 708 条被试×模型×设置记录。同一批被试反复使用。指标取各文件保存的被试中位数，计数列另给各被试匹配峰的总数。每个文件内另有被试编号、配置、指标支持数及可用的配对统计。“离线 500/125+125”代表 500 点窗口和两侧 125 点边距。历史 ±4 记录保留。

| 保存时间 / 原文件 | 模型 | 内部归一化 | 目标与模式 | 被试 | Hz／hop | 滤波 | 幅度比 | 背景 RMSE µV | 计数减少 % | 峰数 输入→输出 | 抑制 dB | 保真 dB | legacy 残留率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [20261030 163815](../eegio_tests/eegio_20261030_163815.json) | tcn_se_both | 未记录 | host · legacy | 2 | 256／32 | 因果 | — | — | — | — | 14.15 | 1.84 | 0.136 |
| [20261030 163815](../eegio_tests/eegio_20261030_163815.json) | tcn_se_bi_both | 未记录 | host · legacy | 2 | 256／32 | 因果 | — | — | — | — | 16.72 | 1.42 | 0.060 |
| [20261104 111752](../eegio_tests/eegio_20261104_111752.json) | tcn_se_both | batchnorm | host · legacy | 4 | 256／32 | 零相位 | — | — | — | — | 13.57 | 1.63 | 0.242 |
| [20261104 112304](../eegio_tests/eegio_20261104_112304.json) | tcn_se_both | batchnorm | host · legacy | 20 | 256／32 | 因果 | — | — | — | — | 16.45 | 1.88 | 0.092 |
| [20261104 133131](../eegio_tests/eegio_20261104_133131.json) | tcn_both_s67 | batchnorm | host · legacy | 20 | 250／4 | 因果 | — | — | — | — | 15.98 | 1.49 | 0.273 |
| [20261104 145834](../eegio_tests/eegio_20261104_145834.json) | tcn_se_both_250hz | ChannelNorm | host · legacy | 20 | 250／4 | 因果 | — | — | — | — | 15.63 | 1.70 | 0.184 |
| [20261104 145834](../eegio_tests/eegio_20261104_145834.json) | tcn_se_both_250hz_s67 | ChannelNorm | host · legacy | 20 | 250／4 | 因果 | — | — | — | — | 17.04 | 1.57 | 0.204 |
| [20261104 145834](../eegio_tests/eegio_20261104_145834.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · legacy | 20 | 250／4 | 因果 | — | — | — | — | 16.56 | 1.46 | 0.125 |
| [20261104 162544](../eegio_tests/eegio_20261104_162544.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 3.99 | 20.22 | 87.87 | 287→39 | 16.56 | 1.46 | 0.125 |
| [20261104 162544](../eegio_tests/eegio_20261104_162544.json) | tcn_se_both_250hz | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 3.53 | 18.96 | 87.75 | 287→29 | 15.63 | 1.70 | 0.184 |
| [20261104 162544](../eegio_tests/eegio_20261104_162544.json) | tcn_se_both_250hz_s67 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 3.45 | 20.05 | 91.49 | 287→31 | 17.04 | 1.57 | 0.204 |
| [20261105 125602](../eegio_tests/eegio_20261105_125602.json) | tcn_both_s67 | batchnorm | host · 常规 | 20 | 250／4 | 因果 | 4.96 | 20.31 | 73.07 | 287→85 | 15.98 | 1.49 | 0.271 |
| [20261105 154312](../eegio_tests/eegio_20261105_154312.json) | tcn_se_both_250hz | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 3.53 | 18.96 | 87.75 | 287→29 | 15.63 | 1.70 | 0.184 |
| [20261105 160709](../eegio_tests/eegio_20261105_160709.json) | tcn_se_eog_250hz_s54 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 5.44 | 18.28 | 66.67 | 287→115 | 11.01 | 2.45 | 0.412 |
| [20261105 164435](../eegio_tests/eegio_20261105_164435.json) | tcn_channelnorm_both_250hz | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 4.34 | 20.59 | 91.49 | 287→36 | 16.54 | 1.39 | 0.196 |
| [20261106 123431](../eegio_tests/eegio_20261106_123431.json) | tcn_channelnorm_both_250hz)_s67 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 4.69 | 20.45 | 90.10 | 287→29 | 16.57 | 1.59 | 0.211 |
| [20261106 163724](../eegio_tests/eegio_20261106_163724_465464_75aac6ab.json) | tcn_channelnorm_both_250hz)_s67 | ChannelNorm | pi5 · paced | 20 | 250／4 | 因果 | 4.69 | 20.45 | 90.10 | 287→29 | 16.57 | 1.59 | 0.211 |
| [20261106 170450](../eegio_tests/eegio_20261106_170450_390445_02778b9b.json) | tcn_channelnorm_both_250hz)_s67 | ChannelNorm | pi5 · fast | 20 | 250／4 | 因果 | 4.69 | 20.45 | 90.10 | 287→29 | 16.57 | 1.59 | 0.211 |
| [20261106 172553](../eegio_tests/eegio_20261106_172553_922703_5d8b11e9.json) | tcn_se_both_250hz_s67 | ChannelNorm | pi5 · fast | 20 | 250／4 | 因果 | 3.45 | 20.05 | 91.49 | 287→31 | 17.04 | 1.57 | 0.204 |
| [20261110 144859](../eegio_tests/eegio_20261110_144859_895086_33347979.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 5.10 | 19.63 | 80.59 | 287→63 | 16.12 | 1.33 | 0.343 |
| [20261110 150502](../eegio_tests/eegio_20261110_150502_942052_09025375.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 常规 | 20 | 250／4 | 零相位 | 6.53 | 13.36 | 34.78 | 341→266 | 14.60 | 1.93 | 0.633 |
| [20261110 152100](../eegio_tests/eegio_20261110_152100_912205_79330b76.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · 历史 ±4 | 20 | 250／1 | 因果 | 7.74 | 19.20 | 0.00 | 287→274 | 13.72 | 1.76 | 0.586 |
| [20261110 152100](../eegio_tests/eegio_20261110_152100_912205_79330b76.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 历史 ±4 | 20 | 250／1 | 因果 | 7.02 | 19.18 | 14.25 | 287→241 | 14.17 | 1.62 | 0.438 |
| [20261110 152925](../eegio_tests/eegio_20261110_152925_265432_3b981a66.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 历史 ±4 | 20 | 250／1 | 因果 | 7.02 | 19.18 | 14.25 | 287→241 | 14.17 | 1.62 | 0.438 |
| [20261111 134139](../eegio_tests/eegio_20261111_134139_803408_e32232a6.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 因果 | 5.27 | 19.63 | 0.00 | 287→263 | 16.52 | 1.96 | 0.239 |
| [20261111 134216](../eegio_tests/eegio_20261111_134216_182753_b0ff10ef.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 因果 | 6.66 | 20.16 | 58.33 | 287→149 | 14.56 | 1.79 | 0.485 |
| [20261111 134607](../eegio_tests/eegio_20261111_134607_825792_2b75bebb.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 3.99 | 20.22 | 87.87 | 287→39 | 16.56 | 1.46 | 0.125 |
| [20261111 134746](../eegio_tests/eegio_20261111_134746_007229_d6dab177.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 因果 | 6.66 | 20.16 | 58.33 | 287→149 | 14.56 | 1.79 | 0.485 |
| [20261111 134746](../eegio_tests/eegio_20261111_134746_007229_d6dab177.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 因果 | 5.27 | 19.63 | 0.00 | 287→263 | 16.52 | 1.96 | 0.239 |
| [20261111 134809](../eegio_tests/eegio_20261111_134809_098499_62ef366d.json) | tcn_se_both_250hz_s54 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 零相位 | 9.82 | 13.28 | 60.00 | 341→161 | 13.93 | 1.96 | 0.691 |
| [20261111 134809](../eegio_tests/eegio_20261111_134809_098499_62ef366d.json) | tcn_se_bi_both_250hz_s54 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 零相位 | 4.62 | 12.61 | 88.00 | 341→56 | 18.01 | 2.43 | 0.040 |
| [20261111 142749](../eegio_tests/eegio_20261111_142749_326332_5feeddc5.json) | tcn_se_both_250hz | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 因果 | 7.10 | 18.84 | 69.05 | 287→125 | 13.81 | 2.17 | 0.650 |
| [20261111 142749](../eegio_tests/eegio_20261111_142749_326332_5feeddc5.json) | tcn_se_bi_both_250hz_s42 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 因果 | 5.14 | 19.20 | 16.67 | 287→233 | 16.99 | 2.15 | 0.172 |
| [20261111 142814](../eegio_tests/eegio_20261111_142814_211471_e2372a18.json) | tcn_se_both_250hz | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 零相位 | 10.04 | 12.85 | 66.67 | 341→173 | 13.29 | 2.42 | 0.725 |
| [20261111 142814](../eegio_tests/eegio_20261111_142814_211471_e2372a18.json) | tcn_se_bi_both_250hz_s42 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 零相位 | 4.72 | 12.50 | 88.74 | 341→48 | 18.12 | 2.53 | 0.039 |
| [20261111 150320](../eegio_tests/eegio_20261111_150320_904452_9cfc3327.json) | tcn_both_s67 | batchnorm | host · 常规 | 20 | 250／4 | 因果 | 4.96 | 20.31 | 73.07 | 287→85 | 15.98 | 1.49 | 0.271 |
| [20261111 150320](../eegio_tests/eegio_20261111_150320_904452_9cfc3327.json) | tcn_se_both_250hz_s67 | ChannelNorm | host · 常规 | 20 | 250／4 | 因果 | 3.45 | 20.05 | 91.49 | 287→31 | 17.04 | 1.57 | 0.204 |
| [20261111 150444](../eegio_tests/eegio_20261111_150444_884258_1d35a4bb.json) | tcn_se_bi_both_250hz_s67 | ChannelNorm | host · 离线 500/125+125 | 20 | 250／— | 零相位 | 3.60 | 12.94 | 81.14 | 341→65 | 19.31 | 2.24 | 0.053 |

### EEG-IO 树莓派回放记录（2026-11-06）

p50/p99 为各被试处理百分位再取中位数。fast 回放未记录的 deadline misses 以“—”标记。

| 记录 | 模型 | 节奏 | 后端 | Hop p50 ms | Hop p99 ms | 超时 hop | 稳态 hop | 总回放 s | ONNX/PyTorch 最大绝对差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [20261106_163724](../eegio_tests/eegio_20261106_163724_465464_75aac6ab.json) | tcn_channelnorm_both_250hz)_s67 | paced | onnx_windowed | 6.19 | 6.58 | 0 | 122688 | 2023.3 | 0.00000042 |
| [20261106_170450](../eegio_tests/eegio_20261106_170450_390445_02778b9b.json) | tcn_channelnorm_both_250hz)_s67 | fast | onnx_windowed | — | — | — | — | 137.4 | 0.00000042 |
| [20261106_172553](../eegio_tests/eegio_20261106_172553_922703_5d8b11e9.json) | tcn_se_both_250hz_s67 | fast | onnx_windowed | 6.21 | 6.52 | — | 122688 | 775.9 | 0.00000030 |

## C. 多通道压力测试

每个完成的实验测了 4 个通道档 × 3 次，20 s／次，使用同一录波、独立模型状态和训练权重。完整测量记录还要求外接功率计数据。

| 实验 | 模型 | 内部归一化 | 设备 | Hz | Hop | 线程 | 核心 | 输入 | 状态 | 已测／计划 | 完整测量记录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [channels_20261111_154801_bf8bd561](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/experiment.json) | tcn_se_both_250hz_s54_step | ChannelNorm | host | 250.0 | 4 | 2 | — | S00 / Fp1 | complete | 12/12 | 0 |
| [channels_20261111_155222_11b3b9c6](../deploy/artifacts/channel_sweeps/channels_20261111_155222_11b3b9c6/experiment.json) | tcn_se_both_250hz_s54_step | ChannelNorm | pi | 250.0 | 4 | 2 | 2-3 | S00 / Fp1 | failed | 0/12 | 0 |
| [channels_20261111_155612_7faa80bb](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/experiment.json) | tcn_se_both_250hz_s54_step | ChannelNorm | pi | 250.0 | 4 | 2 | 2-3 | S00 / Fp1 | complete | 12/12 | 0 |
| [channels_20261111_160141_03657c5d](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/experiment.json) | tcn_se_both_250hz_s54_step | ChannelNorm | pi | 250.0 | 1 | 2 | 2-3 | S00 / Fp1 | complete | 12/12 | 0 |
| [channels_20261111_162159_5574c8e1](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/experiment.json) | tcn_mla_both_step | 未记录 | host | 256.0 | 4 | 2 | — | S00 / Fp1 | complete | 12/12 | 0 |
| [channels_20261111_162615_2a26bf6c](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/experiment.json) | tcn_mla_both_step | 未记录 | pi | 256.0 | 4 | 2 | 2-3 | S00 / Fp1 | complete | 12/12 | 0 |

### channels_20261111_154801_bf8bd561

| 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3/3 | 2.02 ± 0.03 | 2.91 ± 0.14 | 17.85 ± 0.20 | 12.57 ± 0.34 | 134.14 ± 0.22 | 0.00 ± 0.00 | 0 |
| 2 | 3/3 | 3.71 ± 0.03 | 4.32 ± 0.06 | 19.16 ± 0.10 | 20.71 ± 0.06 | 134.20 ± 0.08 | 0.00 ± 0.00 | 0 |
| 8 | 3/3 | 5.44 ± 0.24 | 7.00 ± 0.51 | 20.94 ± 0.45 | 34.45 ± 2.54 | 134.37 ± 0.07 | 0.00 ± 0.00 | 0 |
| 16 | 3/3 | 7.29 ± 0.15 | 8.18 ± 0.20 | 21.81 ± 0.16 | 45.68 ± 0.86 | 135.42 ± 0.11 | 0.00 ± 0.00 | 0 |

| 单次记录 | 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop | 稳态 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [c1_r1](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c1_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c1_r1.csv) | 1 | 1 | 2.01 | 2.92 | 17.82 | 12.52 | 134.39 | 0.00 | 0 | 1219 |
| [c1_r2](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c1_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c1_r2.csv) | 1 | 2 | 2.06 | 3.05 | 18.06 | 12.93 | 134.02 | 0.00 | 0 | 1219 |
| [c1_r3](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c1_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c1_r3.csv) | 1 | 3 | 2.00 | 2.77 | 17.67 | 12.26 | 134.02 | 0.00 | 0 | 1219 |
| [c2_r1](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c2_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c2_r1.csv) | 2 | 1 | 3.68 | 4.37 | 19.08 | 20.65 | 134.11 | 0.00 | 0 | 1219 |
| [c2_r2](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c2_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c2_r2.csv) | 2 | 2 | 3.72 | 4.25 | 19.26 | 20.76 | 134.27 | 0.00 | 0 | 1219 |
| [c2_r3](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c2_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c2_r3.csv) | 2 | 3 | 3.73 | 4.34 | 19.13 | 20.73 | 134.23 | 0.00 | 0 | 1219 |
| [c8_r1](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c8_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c8_r1.csv) | 8 | 1 | 5.53 | 7.08 | 21.01 | 35.87 | 134.39 | 0.00 | 0 | 1219 |
| [c8_r2](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c8_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c8_r2.csv) | 8 | 2 | 5.62 | 7.47 | 21.35 | 35.97 | 134.30 | 0.00 | 0 | 1219 |
| [c8_r3](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c8_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c8_r3.csv) | 8 | 3 | 5.16 | 6.45 | 20.46 | 31.51 | 134.42 | 0.00 | 0 | 1219 |
| [c16_r1](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c16_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c16_r1.csv) | 16 | 1 | 7.37 | 8.41 | 21.99 | 46.25 | 135.55 | 0.00 | 0 | 1219 |
| [c16_r2](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c16_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c16_r2.csv) | 16 | 2 | 7.38 | 8.12 | 21.75 | 46.11 | 135.36 | 0.00 | 0 | 1219 |
| [c16_r3](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c16_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_154801_bf8bd561/c16_r3.csv) | 16 | 3 | 7.12 | 8.01 | 21.69 | 44.69 | 135.36 | 0.00 | 0 | 1219 |

### channels_20261111_155222_11b3b9c6

远端脚本已启动，在第一组实测前被 Pi 5 锁定检查拒绝。[启动及错误日志](../runs/gui_channels_20261111_155221.log)

### channels_20261111_155612_7faa80bb

| 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3/3 | 1.57 ± 0.03 | 1.80 ± 0.03 | 13.86 ± 0.03 | 9.85 ± 0.16 | 132.90 ± 0.38 | 0.00 ± 0.00 | 0 |
| 2 | 3/3 | 3.09 ± 0.09 | 3.49 ± 0.19 | 15.55 ± 0.19 | 19.43 ± 0.57 | 133.09 ± 0.53 | 0.00 ± 0.00 | 0 |
| 8 | 3/3 | 12.30 ± 0.15 | 12.60 ± 0.12 | 24.66 ± 0.12 | 76.93 ± 0.94 | 134.05 ± 0.07 | 0.00 ± 0.00 | 0 |
| 16 | 3/3 | 24.37 ± 0.21 | 24.78 ± 0.22 | 10491.70 ± 265.68 | 152.39 ± 1.34 | 135.19 ± 0.49 | 100.00 ± 0.00 | 3657 |

| 单次记录 | 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop | 稳态 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [c1_r1](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c1_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c1_r1.csv) | 1 | 1 | 1.54 | 1.81 | 13.87 | 9.69 | 133.28 | 0.00 | 0 | 1219 |
| [c1_r2](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c1_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c1_r2.csv) | 1 | 2 | 1.59 | 1.82 | 13.87 | 10.00 | 132.89 | 0.00 | 0 | 1219 |
| [c1_r3](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c1_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c1_r3.csv) | 1 | 3 | 1.57 | 1.77 | 13.83 | 9.85 | 132.52 | 0.00 | 0 | 1219 |
| [c2_r1](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c2_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c2_r1.csv) | 2 | 1 | 3.12 | 3.40 | 15.46 | 19.58 | 132.50 | 0.00 | 0 | 1219 |
| [c2_r2](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c2_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c2_r2.csv) | 2 | 2 | 3.17 | 3.70 | 15.76 | 19.91 | 133.27 | 0.00 | 0 | 1219 |
| [c2_r3](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c2_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c2_r3.csv) | 2 | 3 | 2.99 | 3.36 | 15.42 | 18.80 | 133.52 | 0.00 | 0 | 1219 |
| [c8_r1](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c8_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c8_r1.csv) | 8 | 1 | 12.18 | 12.50 | 24.56 | 76.20 | 134.08 | 0.00 | 0 | 1219 |
| [c8_r2](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c8_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c8_r2.csv) | 8 | 2 | 12.47 | 12.74 | 24.80 | 77.99 | 133.97 | 0.00 | 0 | 1219 |
| [c8_r3](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c8_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c8_r3.csv) | 8 | 3 | 12.23 | 12.57 | 24.62 | 76.59 | 134.09 | 0.00 | 0 | 1219 |
| [c16_r1](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c16_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c16_r1.csv) | 16 | 1 | 24.59 | 25.03 | 10771.89 | 153.81 | 134.92 | 100.00 | 1219 | 1219 |
| [c16_r2](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c16_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c16_r2.csv) | 16 | 2 | 24.17 | 24.63 | 10243.41 | 151.14 | 135.75 | 100.00 | 1219 | 1219 |
| [c16_r3](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c16_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_155612_7faa80bb/c16_r3.csv) | 16 | 3 | 24.35 | 24.68 | 10459.80 | 152.23 | 134.89 | 100.00 | 1219 | 1219 |

### channels_20261111_160141_03657c5d

| 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3/3 | 0.53 ± 0.01 | 0.64 ± 0.04 | 0.69 ± 0.04 | 13.32 ± 0.25 | 135.53 ± 0.47 | 0.00 ± 0.00 | 0 |
| 2 | 3/3 | 1.07 ± 0.03 | 1.19 ± 0.06 | 1.25 ± 0.06 | 26.78 ± 0.87 | 135.48 ± 0.27 | 0.00 ± 0.00 | 0 |
| 8 | 3/3 | 4.26 ± 0.08 | 4.40 ± 0.10 | 1629.40 ± 376.09 | 106.53 ± 1.91 | 136.29 ± 0.50 | 100.00 ± 0.00 | 14628 |
| 16 | 3/3 | 8.52 ± 0.12 | 8.75 ± 0.16 | 22749.29 ± 560.66 | 213.14 ± 2.83 | 137.23 ± 0.28 | 100.00 ± 0.00 | 14628 |

| 单次记录 | 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop | 稳态 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [c1_r1](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c1_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c1_r1.csv) | 1 | 1 | 0.52 | 0.59 | 0.65 | 13.13 | 134.98 | 0.00 | 0 | 4876 |
| [c1_r2](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c1_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c1_r2.csv) | 1 | 2 | 0.52 | 0.66 | 0.72 | 13.23 | 135.86 | 0.00 | 0 | 4876 |
| [c1_r3](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c1_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c1_r3.csv) | 1 | 3 | 0.54 | 0.65 | 0.71 | 13.59 | 135.73 | 0.00 | 0 | 4876 |
| [c2_r1](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c2_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c2_r1.csv) | 2 | 1 | 1.06 | 1.15 | 1.21 | 26.57 | 135.78 | 0.00 | 0 | 4876 |
| [c2_r2](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c2_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c2_r2.csv) | 2 | 2 | 1.04 | 1.15 | 1.21 | 26.04 | 135.39 | 0.00 | 0 | 4876 |
| [c2_r3](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c2_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c2_r3.csv) | 2 | 3 | 1.10 | 1.26 | 1.32 | 27.74 | 135.27 | 0.00 | 0 | 4876 |
| [c8_r1](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c8_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c8_r1.csv) | 8 | 1 | 4.30 | 4.44 | 1817.84 | 107.50 | 136.50 | 100.00 | 4876 | 4876 |
| [c8_r2](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c8_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c8_r2.csv) | 8 | 2 | 4.17 | 4.29 | 1196.34 | 104.33 | 135.72 | 100.00 | 4876 | 4876 |
| [c8_r3](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c8_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c8_r3.csv) | 8 | 3 | 4.31 | 4.47 | 1874.02 | 107.77 | 136.64 | 100.00 | 4876 | 4876 |
| [c16_r1](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c16_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c16_r1.csv) | 16 | 1 | 8.40 | 8.58 | 22169.71 | 210.21 | 137.39 | 100.00 | 4876 | 4876 |
| [c16_r2](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c16_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c16_r2.csv) | 16 | 2 | 8.51 | 8.78 | 22789.27 | 213.37 | 137.39 | 100.00 | 4876 | 4876 |
| [c16_r3](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c16_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_160141_03657c5d/c16_r3.csv) | 16 | 3 | 8.64 | 8.89 | 23288.89 | 215.85 | 136.91 | 100.00 | 4876 | 4876 |

### channels_20261111_162159_5574c8e1

| 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3/3 | 3.53 ± 0.12 | 4.86 ± 0.15 | 19.12 ± 0.20 | 21.32 ± 0.41 | 135.00 ± 0.81 | 0.03 ± 0.05 | 1 |
| 2 | 3/3 | 4.71 ± 0.36 | 7.57 ± 0.16 | 21.43 ± 0.18 | 30.66 ± 1.05 | 135.74 ± 0.69 | 0.00 ± 0.00 | 0 |
| 8 | 3/3 | 6.67 ± 0.38 | 7.79 ± 0.10 | 21.16 ± 0.11 | 42.60 ± 1.12 | 134.65 ± 5.85 | 0.00 ± 0.00 | 0 |
| 16 | 3/3 | 12.02 ± 0.33 | 12.82 ± 0.15 | 25.24 ± 0.11 | 76.98 ± 1.54 | 142.00 ± 0.96 | 0.00 ± 0.00 | 0 |

| 单次记录 | 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop | 稳态 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [c1_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c1_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c1_r1.csv) | 1 | 1 | 3.65 | 4.78 | 18.96 | 21.16 | 135.69 | 0.00 | 0 | 1152 |
| [c1_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c1_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c1_r2.csv) | 1 | 2 | 3.55 | 5.04 | 19.33 | 21.79 | 134.11 | 0.09 | 1 | 1152 |
| [c1_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c1_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c1_r3.csv) | 1 | 3 | 3.41 | 4.78 | 19.06 | 21.01 | 135.20 | 0.00 | 0 | 1152 |
| [c2_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c2_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c2_r1.csv) | 2 | 1 | 4.30 | 7.71 | 21.50 | 29.45 | 136.03 | 0.00 | 0 | 1152 |
| [c2_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c2_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c2_r2.csv) | 2 | 2 | 4.97 | 7.60 | 21.58 | 31.25 | 134.95 | 0.00 | 0 | 1152 |
| [c2_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c2_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c2_r3.csv) | 2 | 3 | 4.86 | 7.39 | 21.23 | 31.27 | 136.23 | 0.00 | 0 | 1152 |
| [c8_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c8_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c8_r1.csv) | 8 | 1 | 6.24 | 7.68 | 21.04 | 41.33 | 137.34 | 0.00 | 0 | 1152 |
| [c8_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c8_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c8_r2.csv) | 8 | 2 | 6.94 | 7.86 | 21.25 | 43.47 | 127.94 | 0.00 | 0 | 1152 |
| [c8_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c8_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c8_r3.csv) | 8 | 3 | 6.83 | 7.83 | 21.19 | 42.99 | 138.67 | 0.00 | 0 | 1152 |
| [c16_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c16_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c16_r1.csv) | 16 | 1 | 11.96 | 12.74 | 25.18 | 76.72 | 142.30 | 0.00 | 0 | 1152 |
| [c16_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c16_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c16_r2.csv) | 16 | 2 | 11.72 | 12.71 | 25.17 | 75.60 | 140.92 | 0.00 | 0 | 1152 |
| [c16_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c16_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162159_5574c8e1/c16_r3.csv) | 16 | 3 | 12.37 | 12.99 | 25.36 | 78.64 | 142.78 | 0.00 | 0 | 1152 |

### channels_20261111_162615_2a26bf6c

| 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3/3 | 3.45 ± 0.01 | 3.91 ± 0.15 | 15.69 ± 0.15 | 22.16 ± 0.08 | 135.40 ± 0.36 | 0.00 ± 0.00 | 0 |
| 2 | 3/3 | 6.84 ± 0.12 | 7.50 ± 0.13 | 19.28 ± 0.13 | 43.89 ± 0.76 | 135.51 ± 0.23 | 0.00 ± 0.00 | 0 |
| 8 | 3/3 | 27.08 ± 0.34 | 27.65 ± 0.36 | 14659.14 ± 420.86 | 173.45 ± 2.15 | 138.71 ± 0.30 | 100.00 ± 0.00 | 3456 |
| 16 | 3/3 | 55.62 ± 0.22 | 56.47 ± 0.19 | 50851.53 ± 278.54 | 356.13 ± 1.43 | 142.35 ± 0.01 | 100.00 ± 0.00 | 3456 |

| 单次记录 | 通道 | 重复 | p50 ms | p99 ms | 端到端 p99 ms | 占空比 % | 峰值 RAM MiB | 超期率 % | 超期 hop | 稳态 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [c1_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c1_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c1_r1.csv) | 1 | 1 | 3.46 | 3.90 | 15.68 | 22.23 | 135.59 | 0.00 | 0 | 1152 |
| [c1_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c1_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c1_r2.csv) | 1 | 2 | 3.43 | 4.07 | 15.85 | 22.08 | 135.62 | 0.00 | 0 | 1152 |
| [c1_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c1_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c1_r3.csv) | 1 | 3 | 3.45 | 3.77 | 15.55 | 22.17 | 134.98 | 0.00 | 0 | 1152 |
| [c2_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c2_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c2_r1.csv) | 2 | 1 | 6.71 | 7.42 | 19.20 | 43.08 | 135.27 | 0.00 | 0 | 1152 |
| [c2_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c2_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c2_r2.csv) | 2 | 2 | 6.94 | 7.65 | 19.43 | 44.59 | 135.72 | 0.00 | 0 | 1152 |
| [c2_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c2_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c2_r3.csv) | 2 | 3 | 6.85 | 7.43 | 19.21 | 43.98 | 135.53 | 0.00 | 0 | 1152 |
| [c8_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c8_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c8_r1.csv) | 8 | 1 | 27.32 | 27.90 | 14944.80 | 174.88 | 138.38 | 100.00 | 1152 | 1152 |
| [c8_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c8_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c8_r2.csv) | 8 | 2 | 27.24 | 27.81 | 14856.77 | 174.48 | 138.95 | 100.00 | 1152 | 1152 |
| [c8_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c8_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c8_r3.csv) | 8 | 3 | 26.69 | 27.24 | 14175.83 | 170.98 | 138.81 | 100.00 | 1152 | 1152 |
| [c16_r1](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c16_r1.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c16_r1.csv) | 16 | 1 | 55.56 | 56.42 | 50774.83 | 355.78 | 142.36 | 100.00 | 1152 | 1152 |
| [c16_r2](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c16_r2.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c16_r2.csv) | 16 | 2 | 55.44 | 56.31 | 50619.37 | 354.90 | 142.34 | 100.00 | 1152 | 1152 |
| [c16_r3](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c16_r3.json) · [时间序列](../deploy/artifacts/channel_sweeps/channels_20261111_162615_2a26bf6c/c16_r3.csv) | 16 | 3 | 55.86 | 56.68 | 51160.38 | 357.70 | 142.34 | 100.00 | 1152 | 1152 |

## D. 历史部署基准

locked、unlocked 按环境字段 verdict 原样引用。没有设备记录时可从 host 文件前缀识别主机。常规表的 p50/p99 为 total_ms，duty 表采用单独测得的 compute_*。

| 延迟记录 | 设备 | 环境 | 路径 | Hz | Hop | n hop | p50 ms | p99 ms | 平均占空比 % | 超期 hop | 状态字节 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [host_tcn_both_hop32](../deploy/artifacts/pi_latency/host_tcn_both_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_both_hop32.csv) | 主机 | unlocked | — | 256.0 | 32 | 1584 | 0.708 | 0.758 | 0.57 | 0 | — |
| [host_tcn_both_s42_hop32](../deploy/artifacts/pi_latency/host_tcn_both_s42_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_both_s42_hop32.csv) | 主机 | unlocked | windowed | 256.0 | 32 | 1584 | 0.708 | 0.787 | 0.57 | 0 | — |
| [host_tcn_eog_hop32](../deploy/artifacts/pi_latency/host_tcn_eog_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_eog_hop32.csv) | 主机 | unlocked | — | 256.0 | 32 | 1584 | 0.713 | 0.755 | 0.57 | 0 | — |
| [host_tcn_mla_both_s67_hop32](../deploy/artifacts/pi_latency/host_tcn_mla_both_s67_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_mla_both_s67_hop32.csv) | 主机 | unlocked | — | 256.0 | 32 | 1584 | 2.957 | 3.117 | 2.37 | 0 | — |
| [host_tcn_mla_eog_hop32](../deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32.csv) | 主机 | — | — | 256.0 | 32 | 1584 | 2.821 | 3.061 | 2.27 | 0 | — |
| [host_tcn_se_both_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_se_both_hop32.csv) | 主机 | unlocked | windowed | 256.0 | 32 | 1584 | 1.204 | 1.297 | 0.97 | 0 | — |
| [host_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s54_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_se_both_s54_hop32.csv) | 主机 | unlocked | windowed | 256.0 | 32 | 1584 | 1.206 | 1.301 | 0.97 | 0 | — |
| [host_tcn_se_both_s67_h64_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s67_h64_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_se_both_s67_h64_hop32.csv) | 主机 | unlocked | windowed | 256.0 | 32 | 1584 | 1.206 | 1.308 | 0.97 | 0 | — |
| [incremental_tcn_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_both_hop1.json) · [时间序列](../deploy/artifacts/pi_latency/incremental_tcn_both_hop1.csv) | Pi 5 | locked | incremental | 256.0 | 1 | 51200 | 0.247 | 0.267 | 6.36 | 0 | 31240 |
| [incremental_tcn_both_s42_hop1](../deploy/artifacts/pi_latency/incremental_tcn_both_s42_hop1.json) · [时间序列](../deploy/artifacts/pi_latency/incremental_tcn_both_s42_hop1.csv) | Pi 5 | locked | incremental | 256.0 | 1 | 51200 | 0.245 | 0.285 | 6.32 | 0 | 31240 |
| [incremental_tcn_mla_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_mla_both_hop1.json) · [时间序列](../deploy/artifacts/pi_latency/incremental_tcn_mla_both_hop1.csv) | Pi 5 | locked | incremental | 256.0 | 1 | 51200 | 0.767 | 0.850 | 19.76 | 0 | 197128 |
| [incremental_tcn_mla_eog_hop1](../deploy/artifacts/pi_latency/incremental_tcn_mla_eog_hop1.json) | 未记录 | — | incremental | 256.0 | 1 | 600 | 0.511 | 0.548 | 13.07 | — | 197900 |
| [incremental_tcn_se_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_hop1.json) · [时间序列](../deploy/artifacts/pi_latency/incremental_tcn_se_both_hop1.csv) | Pi 5 | locked | incremental | 256.0 | 1 | 51200 | 0.319 | 0.351 | 8.23 | 0 | 32540 |
| [incremental_tcn_se_both_s54_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s54_hop1.json) · [时间序列](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s54_hop1.csv) | Pi 5 | locked | incremental | 256.0 | 1 | 51200 | 0.321 | 0.380 | 8.28 | 0 | 32540 |
| [incremental_tcn_se_both_s67_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s67_hop1.json) · [时间序列](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s67_hop1.csv) | Pi 5 | locked | incremental | 256.0 | 1 | 51200 | 0.276 | 0.318 | 7.12 | 0 | 24412 |
| [latency_tcn_both_hop32](../deploy/artifacts/pi_latency/latency_tcn_both_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop32.csv) | Pi 5 | unlocked | windowed | 256.0 | 32 | 1584 | 3.230 | 3.358 | 2.58 | 0 | — |
| [latency_tcn_both_hop8](../deploy/artifacts/pi_latency/latency_tcn_both_hop8.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop8.csv) | Pi 5 | locked | — | 256.0 | 8 | 6336 | 3.227 | 3.516 | 10.32 | 0 | — |
| [latency_tcn_eog_hop32](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32.csv) | Pi 5 | locked | — | 256.0 | 32 | 1584 | 3.133 | 3.285 | 2.51 | 0 | — |
| [latency_tcn_mla_both_hop32](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32.csv) | Pi 5 | locked | — | 256.0 | 32 | 1584 | 17.512 | 17.963 | 14.02 | 0 | — |
| [latency_tcn_mla_both_s67_hop32](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32.csv) | Pi 5 | locked | — | 256.0 | 32 | 1584 | 14.980 | 15.333 | 11.99 | 0 | — |
| [latency_tcn_mla_eog_hop32](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32.csv) | 未记录 | — | — | 256.0 | 32 | 1584 | 15.228 | 16.689 | 12.21 | 0 | — |
| [latency_tcn_mla_eog_hop32_realtime](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_realtime.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_realtime.csv) | 未记录 | — | — | 256.0 | 32 | 224 | 16.881 | 17.750 | — | 0 | — |
| [latency_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/latency_tcn_se_both_s54_hop32.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_se_both_s54_hop32.csv) | Pi 5 | locked | windowed | 256.0 | 32 | 1584 | 4.498 | 4.626 | 3.60 | 0 | — |

### 按节奏运行的 duty-cycle 对照

| 原始记录 | 设备 | 环境 | Hop | n hop | 预算 ms | 计算 p50 ms | 计算 p99 ms | 平均占空比 % | 端到端 p50 ms | 端到端最大 ms | 超期 hop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [host_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/host_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_eog_hop32_duty.csv) | 主机 | unlocked | 32 | 16 | 125.000 | 4.278 | 5.542 | 3.48 | 9.216 | 10.670 | 0 |
| [host_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/host_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_eog_hop32_duty.csv) | 主机 | unlocked | 8 | 64 | 31.250 | 1.169 | 1.795 | 3.79 | 6.190 | 6.477 | 0 |
| [host_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32_duty.csv) | 主机 | — | 32 | 8 | 125.000 | 10.714 | 13.328 | 8.38 | 15.090 | 18.229 | 0 |
| [host_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32_duty.csv) | 主机 | — | 8 | 32 | 31.250 | 3.701 | 6.225 | 12.39 | 8.695 | 10.049 | 0 |
| [latency_tcn_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.csv) | Pi 5 | unlocked | 64 | 16 | 250.000 | 4.973 | 5.512 | 1.92 | 5.067 | 5.687 | 0 |
| [latency_tcn_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.csv) | Pi 5 | unlocked | 32 | 32 | 125.000 | 4.475 | 4.735 | 3.58 | 4.567 | 4.830 | 0 |
| [latency_tcn_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.csv) | Pi 5 | unlocked | 16 | 64 | 62.500 | 3.348 | 4.265 | 5.39 | 3.416 | 5.417 | 0 |
| [latency_tcn_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.csv) | Pi 5 | unlocked | 8 | 128 | 31.250 | 3.337 | 5.250 | 10.76 | 3.405 | 5.377 | 0 |
| [latency_tcn_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_both_hop32_duty.csv) | Pi 5 | unlocked | 4 | 256 | 15.625 | 3.311 | 3.445 | 21.08 | 3.379 | 3.915 | 0 |
| [latency_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.csv) | Pi 5 | locked | 64 | 72 | 250.000 | 3.332 | 3.553 | 1.34 | 3.411 | 3.653 | 0 |
| [latency_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.csv) | Pi 5 | locked | 32 | 144 | 125.000 | 3.305 | 3.599 | 2.66 | 3.379 | 3.760 | 0 |
| [latency_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.csv) | Pi 5 | locked | 16 | 288 | 62.500 | 3.256 | 3.542 | 5.22 | 3.325 | 3.761 | 0 |
| [latency_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.csv) | Pi 5 | locked | 8 | 576 | 31.250 | 3.232 | 3.697 | 10.58 | 3.300 | 16.159 | 0 |
| [latency_tcn_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_eog_hop32_duty.csv) | Pi 5 | locked | 4 | 1152 | 15.625 | 3.204 | 3.470 | 20.55 | 3.271 | 7.410 | 0 |
| [latency_tcn_mla_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.csv) | Pi 5 | locked | 64 | 72 | 250.000 | 17.471 | 17.959 | 7.00 | 17.547 | 18.321 | 0 |
| [latency_tcn_mla_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.csv) | Pi 5 | locked | 32 | 144 | 125.000 | 17.438 | 17.825 | 13.96 | 17.517 | 17.986 | 0 |
| [latency_tcn_mla_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.csv) | Pi 5 | locked | 16 | 288 | 62.500 | 17.431 | 17.721 | 27.91 | 17.508 | 18.920 | 0 |
| [latency_tcn_mla_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.csv) | Pi 5 | locked | 8 | 576 | 31.250 | 17.406 | 17.838 | 55.78 | 17.486 | 19.693 | 0 |
| [latency_tcn_mla_both_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_hop32_duty.csv) | Pi 5 | locked | 4 | 1152 | 15.625 | 17.415 | 17.895 | 111.54 | 1145.976 | 2306.870 | 1152 |
| [latency_tcn_mla_both_s67_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32_duty.csv) | Pi 5 | locked | 32 | 48 | 125.000 | 14.969 | 16.015 | 12.03 | 15.047 | 16.495 | 0 |
| [latency_tcn_mla_both_s67_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32_duty.csv) | Pi 5 | locked | 8 | 192 | 31.250 | 14.896 | 15.167 | 47.72 | 14.975 | 17.196 | 0 |
| [latency_tcn_mla_both_s67_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_both_s67_hop32_duty.csv) | Pi 5 | locked | 4 | 384 | 15.625 | 14.879 | 15.303 | 95.32 | 14.963 | 16.510 | 3 |
| [latency_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.csv) | 未记录 | — | 64 | 72 | 250.000 | 16.141 | 20.894 | 6.82 | 16.225 | 21.025 | 0 |
| [latency_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.csv) | 未记录 | — | 32 | 144 | 125.000 | 17.962 | 18.674 | 13.99 | 18.050 | 19.966 | 0 |
| [latency_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.csv) | 未记录 | — | 16 | 288 | 62.500 | 16.057 | 16.535 | 25.74 | 16.137 | 20.621 | 0 |
| [latency_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.csv) | 未记录 | — | 8 | 576 | 31.250 | 16.039 | 17.584 | 51.54 | 16.117 | 18.487 | 0 |
| [latency_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.csv) | 未记录 | — | 4 | 1152 | 15.625 | 16.056 | 17.654 | 102.98 | 368.352 | 759.388 | 1152 |
| [latency_tcn_mla_eog_hop32_duty](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.json) · [时间序列](../deploy/artifacts/pi_latency/latency_tcn_mla_eog_hop32_duty.csv) | 未记录 | — | 2 | 2304 | 7.812 | 16.061 | 17.730 | 206.03 | 9726.681 | 19514.836 | 2304 |

### 部署基准附带的半合成 test 集分数

| 原始记录 | 测试夹具 | n epoch | 划分 seed | 推理模式 | RRMSE_t | RRMSE_f | CC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [host_tcn_both_s42_hop32](../deploy/artifacts/pi_latency/host_tcn_both_s42_hop32.json) | BOTH_test100.npz | 100 | 42 | noisy input (no model) | 2.0008 | 5.9320 | 0.4713 |
| [host_tcn_both_s42_hop32](../deploy/artifacts/pi_latency/host_tcn_both_s42_hop32.json) | BOTH_test100.npz | 100 | 42 | epoch mode | 0.7277 | 0.6461 | 0.6918 |
| [host_tcn_both_s42_hop32](../deploy/artifacts/pi_latency/host_tcn_both_s42_hop32.json) | BOTH_test100.npz | 100 | 42 | stream mode (hop 32) | 0.7423 | 0.6640 | 0.6793 |
| [host_tcn_se_both_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_hop32.json) | EOG_test100.npz | 100 | 42 | noisy input (no model) | 1.4076 | 4.9135 | 0.5950 |
| [host_tcn_se_both_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_hop32.json) | EOG_test100.npz | 100 | 42 | epoch mode | 0.5226 | 0.5292 | 0.8529 |
| [host_tcn_se_both_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_hop32.json) | EOG_test100.npz | 100 | 42 | stream mode (hop 32) | 0.5893 | 0.6415 | 0.8208 |
| [host_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s54_hop32.json) | BOTH_test100_s54.npz | 100 | 54 | noisy input (no model) | 2.0013 | 5.8275 | 0.4767 |
| [host_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s54_hop32.json) | BOTH_test100_s54.npz | 100 | 54 | epoch mode | 0.7235 | 0.6530 | 0.6934 |
| [host_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s54_hop32.json) | BOTH_test100_s54.npz | 100 | 54 | stream mode (hop 32) | 0.7592 | 0.7265 | 0.6708 |
| [host_tcn_se_both_s67_h64_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s67_h64_hop32.json) | BOTH_test100_s67.npz | 100 | 67 | noisy input (no model) | 1.9635 | 5.9761 | 0.4952 |
| [host_tcn_se_both_s67_h64_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s67_h64_hop32.json) | BOTH_test100_s67.npz | 100 | 67 | epoch mode | 0.7007 | 0.6506 | 0.7159 |
| [host_tcn_se_both_s67_h64_hop32](../deploy/artifacts/pi_latency/host_tcn_se_both_s67_h64_hop32.json) | BOTH_test100_s67.npz | 100 | 67 | stream mode (hop 32) | 0.7505 | 0.8213 | 0.6832 |
| [incremental_tcn_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_both_hop1.json) | BOTH_test100_s67.npz | 100 | 67 | noisy input (no model) | 1.9635 | 5.9761 | 0.4952 |
| [incremental_tcn_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_both_hop1.json) | BOTH_test100_s67.npz | 100 | 67 | epoch mode (stepped) | 0.7125 | 0.6656 | 0.7031 |
| [incremental_tcn_both_s42_hop1](../deploy/artifacts/pi_latency/incremental_tcn_both_s42_hop1.json) | BOTH_test100.npz | 100 | 42 | noisy input (no model) | 2.0008 | 5.9320 | 0.4713 |
| [incremental_tcn_both_s42_hop1](../deploy/artifacts/pi_latency/incremental_tcn_both_s42_hop1.json) | BOTH_test100.npz | 100 | 42 | epoch mode (stepped) | 0.7277 | 0.6461 | 0.6918 |
| [incremental_tcn_mla_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_mla_both_hop1.json) | BOTH_test100_s67.npz | 100 | 67 | noisy input (no model) | 1.9635 | 5.9761 | 0.4952 |
| [incremental_tcn_mla_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_mla_both_hop1.json) | BOTH_test100_s67.npz | 100 | 67 | epoch mode | 0.7028 | 0.6479 | 0.7139 |
| [incremental_tcn_se_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_hop1.json) | BOTH_test100.npz | 100 | 42 | noisy input (no model) | 2.0008 | 5.9320 | 0.4713 |
| [incremental_tcn_se_both_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_hop1.json) | BOTH_test100.npz | 100 | 42 | epoch mode (stepped) | 0.7135 | 0.6352 | 0.7076 |
| [incremental_tcn_se_both_s54_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s54_hop1.json) | BOTH_test100_s54.npz | 100 | 54 | noisy input (no model) | 2.0013 | 5.8275 | 0.4767 |
| [incremental_tcn_se_both_s54_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s54_hop1.json) | BOTH_test100_s54.npz | 100 | 54 | epoch mode (stepped) | 0.7235 | 0.6530 | 0.6934 |
| [incremental_tcn_se_both_s67_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s67_hop1.json) | BOTH_test100_s67.npz | 100 | 67 | noisy input (no model) | 1.9635 | 5.9761 | 0.4952 |
| [incremental_tcn_se_both_s67_hop1](../deploy/artifacts/pi_latency/incremental_tcn_se_both_s67_hop1.json) | BOTH_test100_s67.npz | 100 | 67 | epoch mode (stepped) | 0.7025 | 0.6548 | 0.7144 |
| [latency_tcn_both_hop32](../deploy/artifacts/pi_latency/latency_tcn_both_hop32.json) | BOTH_test100_s67.npz | 100 | 67 | noisy input (no model) | 1.9635 | 5.9761 | 0.4952 |
| [latency_tcn_both_hop32](../deploy/artifacts/pi_latency/latency_tcn_both_hop32.json) | BOTH_test100_s67.npz | 100 | 67 | epoch mode | 0.7125 | 0.6656 | 0.7031 |
| [latency_tcn_both_hop32](../deploy/artifacts/pi_latency/latency_tcn_both_hop32.json) | BOTH_test100_s67.npz | 100 | 67 | stream mode (hop 32) | 0.7359 | 0.6974 | 0.6844 |
| [latency_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/latency_tcn_se_both_s54_hop32.json) | BOTH_test100_s54.npz | 100 | 54 | noisy input (no model) | 2.0013 | 5.8275 | 0.4767 |
| [latency_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/latency_tcn_se_both_s54_hop32.json) | BOTH_test100_s54.npz | 100 | 54 | epoch mode | 0.7235 | 0.6530 | 0.6934 |
| [latency_tcn_se_both_s54_hop32](../deploy/artifacts/pi_latency/latency_tcn_se_both_s54_hop32.json) | BOTH_test100_s54.npz | 100 | 54 | stream mode (hop 32) | 0.7592 | 0.7265 | 0.6708 |

### 导出和主机运行的诊断

[主机增量/窗口路径对照](../deploy/artifacts/incremental_host.json)含 per-sample 延迟、缓存、窗口冗余和环境。

| 导出侧车 | ONNX | 变体 | 归一化 | checkpoint | epoch | 量化 |
| --- | --- | --- | --- | --- | --- | --- |
| [tcn_both](../deploy/artifacts/tcn_both.json) | [ONNX](../deploy/artifacts/tcn_both.onnx) | tcn | 未记录 | runs/tcn_both/best.pt | 155 | — |
| [tcn_both_int8](../deploy/artifacts/tcn_both_int8.json) | [ONNX](../deploy/artifacts/tcn_both_int8.onnx) | — | 未记录 | — | — | static (scales are graph constants) |
| [tcn_both_s42](../deploy/artifacts/tcn_both_s42.json) | [ONNX](../deploy/artifacts/tcn_both_s42.onnx) | tcn | 未记录 | runs/tcn_both_s42/best.pt | 174 | — |
| [tcn_both_s42_step](../deploy/artifacts/tcn_both_s42_step.json) | [ONNX](../deploy/artifacts/tcn_both_s42_step.onnx) | tcn | 未记录 | runs/tcn_both_s42/best.pt | 174 | — |
| [tcn_both_step](../deploy/artifacts/tcn_both_step.json) | [ONNX](../deploy/artifacts/tcn_both_step.onnx) | tcn | 未记录 | runs/tcn_both/best.pt | 155 | — |
| [tcn_eog](../deploy/artifacts/tcn_eog.json) | [ONNX](../deploy/artifacts/tcn_eog.onnx) | tcn | 未记录 | runs/tcn_eog/best.pt | 132 | — |
| [tcn_mla_both](../deploy/artifacts/tcn_mla_both.json) | [ONNX](../deploy/artifacts/tcn_mla_both.onnx) | tcn_mla | 未记录 | runs/tcn_mla_both/best.pt | 184 | — |
| [tcn_mla_both_int8conv](../deploy/artifacts/tcn_mla_both_int8conv.json) | [ONNX](../deploy/artifacts/tcn_mla_both_int8conv.onnx) | — | 未记录 | — | — | static (scales are graph constants) |
| [tcn_mla_both_s67](../deploy/artifacts/tcn_mla_both_s67.json) | [ONNX](../deploy/artifacts/tcn_mla_both_s67.onnx) | tcn_mla | 未记录 | runs/tcn_mla_both_s67/best.pt | 184 | — |
| [tcn_mla_both_s67_step](../deploy/artifacts/tcn_mla_both_s67_step.json) | [ONNX](../deploy/artifacts/tcn_mla_both_s67_step.onnx) | tcn_mla | 未记录 | runs/tcn_mla_both_s67/best.pt | 184 | — |
| [tcn_mla_both_step](../deploy/artifacts/tcn_mla_both_step.json) | [ONNX](../deploy/artifacts/tcn_mla_both_step.onnx) | tcn_mla | 未记录 | runs/tcn_mla_both/best.pt | 184 | — |
| [tcn_mla_eog](../deploy/artifacts/tcn_mla_eog.json) | [ONNX](../deploy/artifacts/tcn_mla_eog.onnx) | tcn_mla | 未记录 | runs/tcn_mla_eog/best.pt | 139 | — |
| [tcn_se_both](../deploy/artifacts/tcn_se_both.json) | [ONNX](../deploy/artifacts/tcn_se_both.onnx) | tcn_se | 未记录 | runs/tcn_se_both/best.pt | 174 | — |
| [tcn_se_both_250hz_s54_step](../deploy/artifacts/tcn_se_both_250hz_s54_step.json) | [ONNX](../deploy/artifacts/tcn_se_both_250hz_s54_step.onnx) | tcn_se | ChannelNorm | runs/tcn_se_both_250hz_s54/best.pt | 182 | — |
| [tcn_se_both_s54](../deploy/artifacts/tcn_se_both_s54.json) | [ONNX](../deploy/artifacts/tcn_se_both_s54.onnx) | tcn_se | 未记录 | runs/tcn_se_both_s54/best.pt | 177 | — |
| [tcn_se_both_s54_step](../deploy/artifacts/tcn_se_both_s54_step.json) | [ONNX](../deploy/artifacts/tcn_se_both_s54_step.onnx) | tcn_se | 未记录 | runs/tcn_se_both_s54/best.pt | 177 | — |
| [tcn_se_both_s67](../deploy/artifacts/tcn_se_both_s67.json) | [ONNX](../deploy/artifacts/tcn_se_both_s67.onnx) | tcn_se | 未记录 | runs/tcn_se_both_s67/best.pt | 193 | — |
| [tcn_se_both_s67_h64](../deploy/artifacts/tcn_se_both_s67_h64.json) | [ONNX](../deploy/artifacts/tcn_se_both_s67_h64.onnx) | tcn_se | 未记录 | runs/tcn_se_both_s67_h64/best.pt | 184 | — |
| [tcn_se_both_s67_step](../deploy/artifacts/tcn_se_both_s67_step.json) | [ONNX](../deploy/artifacts/tcn_se_both_s67_step.onnx) | tcn_se | 未记录 | runs/tcn_se_both_s67/best.pt | 193 | — |
| [tcn_se_both_step](../deploy/artifacts/tcn_se_both_step.json) | [ONNX](../deploy/artifacts/tcn_se_both_step.onnx) | tcn_se | 未记录 | runs/tcn_se_both/best.pt | 174 | — |

部署测试夹具。[BOTH_test100.npz](../deploy/testdata/BOTH_test100.npz) [BOTH_test100_s54.npz](../deploy/testdata/BOTH_test100_s54.npz) [BOTH_test100_s67.npz](../deploy/testdata/BOTH_test100_s67.npz) [EOG_test100.npz](../deploy/testdata/EOG_test100.npz)

## E. 伪影切换、诊断和训练计划

| 伪影切换来源 | 模型 | Hz | 配对 epoch | 切换前 leakage RMS | 首次非零偏移（点） | overshoot | 原曲线恢复 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [figures/causality](../figures/causality/switch_response.json) | MLA-causal | 256 | 2550 | 0.00000 | — | 1.526 | 54.69 |
| [figures/causality](../figures/causality/switch_response.json) | MLA-bidir | 256 | 2550 | 0.01499 | -127 | 1.635 | 66.41 |
| [figures/causality](../figures/causality/switch_response.json) | TCN-noncausal | 256 | 2550 | 0.00571 | -62 | 1.607 | 66.41 |
| [figures/gui_eval_20260926_123603](../figures/gui_eval_20260926_123603/switch_response.json) | TCN BOTH | 256 | 2550 | 0.00000 | — | 1.594 | 66.41 |
| [figures/gui_eval_20260926_124029](../figures/gui_eval_20260926_124029/switch_response.json) | TCN MLA BOTH | 256 | 2550 | 0.00000 | — | 1.526 | 54.69 |
| [figures](../figures/switch_response.json) | TCN-only | 256 | 2550 | 0.00000 | — | 1.594 | 66.41 |
| [figures](../figures/switch_response.json) | TCN+MHA | 256 | 2550 | 0.00000 | — | 1.535 | 58.59 |
| [figures](../figures/switch_response.json) | TCN+MLA | 256 | 2550 | 0.00000 | — | 1.526 | 54.69 |

**eegio_20261105**

[EEG-IO诊断报告.md](../../diagnostics/eegio_20261105/EEG-IO%E8%AF%8A%E6%96%AD%E6%8A%A5%E5%91%8A.md) · [audit.json](../../diagnostics/eegio_20261105/audit.json) · [audit_overview.png](../../diagnostics/eegio_20261105/audit_overview.png) · [background_guard_sensitivity.json](../../diagnostics/eegio_20261105/background_guard_sensitivity.json) · [clean_probe.npz](../../diagnostics/eegio_20261105/clean_probe.npz) · [diagnose_eegio.py](../../diagnostics/eegio_20261105/diagnose_eegio.py) · [plot_audit.py](../../diagnostics/eegio_20261105/plot_audit.py) · [provenance.json](../../diagnostics/eegio_20261105/provenance.json) · [traces.npz](../../diagnostics/eegio_20261105/traces.npz)

**eegio_20261105_eog**

[EOG与BOTH训练比较.md](../../diagnostics/eegio_20261105_eog/EOG%E4%B8%8EBOTH%E8%AE%AD%E7%BB%83%E6%AF%94%E8%BE%83.md) · [probe_eog.py](../../diagnostics/eegio_20261105_eog/probe_eog.py) · [probe_results.json](../../diagnostics/eegio_20261105_eog/probe_results.json)

**eegio_20261105_se**

[TCN-SE结果分析.md](../../diagnostics/eegio_20261105_se/TCN-SE%E7%BB%93%E6%9E%9C%E5%88%86%E6%9E%90.md) · [blink_amplitude_components.json](../../diagnostics/eegio_20261105_se/blink_amplitude_components.json) · [common_clean_probe.npz](../../diagnostics/eegio_20261105_se/common_clean_probe.npz) · [compare_se.py](../../diagnostics/eegio_20261105_se/compare_se.py) · [comparison.json](../../diagnostics/eegio_20261105_se/comparison.json) · [plot_comparison.py](../../diagnostics/eegio_20261105_se/plot_comparison.py) · [provenance.json](../../diagnostics/eegio_20261105_se/provenance.json) · [tcn_se_comparison.png](../../diagnostics/eegio_20261105_se/tcn_se_comparison.png) · [tcn_se_comparison.svg](../../diagnostics/eegio_20261105_se/tcn_se_comparison.svg) · [threshold_sensitivity.json](../../diagnostics/eegio_20261105_se/threshold_sensitivity.json) · [traces.npz](../../diagnostics/eegio_20261105_se/traces.npz)

| 计划 | Hz | TCN 归一化 | SE 窗口 | 共享有效区间 | 任务数 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [paper_methodology_20261103_250](../runs/paper_methodology_20261103_250/plan.json) | 250 | ChannelNorm | 16 | [219, 398) | 27 | planned_not_run |
| [paper_methodology_20261103_256](../runs/paper_methodology_20261103_256/plan.json) | 256 | ChannelNorm | 16 | [199, 410) | 27 | planned_not_run |
| [paper_no_tcn_norm_250](../runs/paper_no_tcn_norm_250/plan.json) | 250 | none | 16 | [219, 398) | 27 | planned_not_run |
| [paper_no_tcn_norm_256](../runs/paper_no_tcn_norm_256/plan.json) | 256 | none | 16 | [199, 410) | 27 | planned_not_run |

旧论文草稿和方法记录。[论文 TeX](../../Causal_Attention_with_Low_Rank_KV_Compressionfor_Online_EEG_Artifact_Removal%20%281%29/main.tex) · [归档研究笔记](../docs/archive/research-notes-before-roadmap.md) · [当前论文协议](../docs/methodology.md)

## F. 启动、失败与过程日志

按 GUI 功能归档，同一实验可能对应多个过程日志。计划、日志和生成配置不另计为完成实验。

### GUI 训练（103 份）

- [gui_train_20260917_160445.log](../runs/gui_train_20260917_160445.log)
- [gui_train_20260918_125018.log](../runs/gui_train_20260918_125018.log)
- [gui_train_20260922_142532.log](../runs/gui_train_20260922_142532.log)
- [gui_train_20260922_151036.log](../runs/gui_train_20260922_151036.log)
- [gui_train_20260922_155112.log](../runs/gui_train_20260922_155112.log)
- [gui_train_20260922_232540.log](../runs/gui_train_20260922_232540.log)
- [gui_train_20260924_115759.log](../runs/gui_train_20260924_115759.log)
- [gui_train_20260924_130035.log](../runs/gui_train_20260924_130035.log)
- [gui_train_20260924_131314.log](../runs/gui_train_20260924_131314.log)
- [gui_train_20260924_141220.log](../runs/gui_train_20260924_141220.log)
- [gui_train_20260924_154117.log](../runs/gui_train_20260924_154117.log)
- [gui_train_20260924_184349.log](../runs/gui_train_20260924_184349.log)
- [gui_train_20260924_201928.log](../runs/gui_train_20260924_201928.log)
- [gui_train_20260924_220321.log](../runs/gui_train_20260924_220321.log)
- [gui_train_20260924_233221.log](../runs/gui_train_20260924_233221.log)
- [gui_train_20260925_112517.log](../runs/gui_train_20260925_112517.log)
- [gui_train_20260925_112836.log](../runs/gui_train_20260925_112836.log)
- [gui_train_20260925_114539.log](../runs/gui_train_20260925_114539.log)
- [gui_train_20260925_134753.log](../runs/gui_train_20260925_134753.log)
- [gui_train_20260925_141424.log](../runs/gui_train_20260925_141424.log)
- [gui_train_20260925_151220.log](../runs/gui_train_20260925_151220.log)
- [gui_train_20260925_160950.log](../runs/gui_train_20260925_160950.log)
- [gui_train_20260925_161031.log](../runs/gui_train_20260925_161031.log)
- [gui_train_20260925_161841.log](../runs/gui_train_20260925_161841.log)
- [gui_train_20260925_181348.log](../runs/gui_train_20260925_181348.log)
- [gui_train_20260925_190349.log](../runs/gui_train_20260925_190349.log)
- [gui_train_20260925_190407.log](../runs/gui_train_20260925_190407.log)
- [gui_train_20260925_191637.log](../runs/gui_train_20260925_191637.log)
- [gui_train_20260925_193856.log](../runs/gui_train_20260925_193856.log)
- [gui_train_20260925_195046.log](../runs/gui_train_20260925_195046.log)
- [gui_train_20260925_204441.log](../runs/gui_train_20260925_204441.log)
- [gui_train_20260925_222557.log](../runs/gui_train_20260925_222557.log)
- [gui_train_20260925_231454.log](../runs/gui_train_20260925_231454.log)
- [gui_train_20260925_232539.log](../runs/gui_train_20260925_232539.log)
- [gui_train_20260929_200144.log](../runs/gui_train_20260929_200144.log)
- [gui_train_20261013_140542.log](../runs/gui_train_20261013_140542.log)
- [gui_train_20261013_142918.log](../runs/gui_train_20261013_142918.log)
- [gui_train_20261013_143942.log](../runs/gui_train_20261013_143942.log)
- [gui_train_20261013_160324.log](../runs/gui_train_20261013_160324.log)
- [gui_train_20261013_161515.log](../runs/gui_train_20261013_161515.log)
- [gui_train_20261014_112728.log](../runs/gui_train_20261014_112728.log)
- [gui_train_20261014_130241.log](../runs/gui_train_20261014_130241.log)
- [gui_train_20261021_013519.log](../runs/gui_train_20261021_013519.log)
- [gui_train_20261021_115013.log](../runs/gui_train_20261021_115013.log)
- [gui_train_20261021_120135.log](../runs/gui_train_20261021_120135.log)
- [gui_train_20261021_120159.log](../runs/gui_train_20261021_120159.log)
- [gui_train_20261021_131314.log](../runs/gui_train_20261021_131314.log)
- [gui_train_20261021_132729.log](../runs/gui_train_20261021_132729.log)
- [gui_train_20261021_134156.log](../runs/gui_train_20261021_134156.log)
- [gui_train_20261021_135304.log](../runs/gui_train_20261021_135304.log)
- [gui_train_20261021_140740.log](../runs/gui_train_20261021_140740.log)
- [gui_train_20261021_151300.log](../runs/gui_train_20261021_151300.log)
- [gui_train_20261021_151322.log](../runs/gui_train_20261021_151322.log)
- [gui_train_20261021_152435.log](../runs/gui_train_20261021_152435.log)
- [gui_train_20261021_154040.log](../runs/gui_train_20261021_154040.log)
- [gui_train_20261022_143035.log](../runs/gui_train_20261022_143035.log)
- [gui_train_20261022_144514.log](../runs/gui_train_20261022_144514.log)
- [gui_train_20261022_151144.log](../runs/gui_train_20261022_151144.log)
- [gui_train_20261022_162948.log](../runs/gui_train_20261022_162948.log)
- [gui_train_20261022_163000.log](../runs/gui_train_20261022_163000.log)
- [gui_train_20261022_163023.log](../runs/gui_train_20261022_163023.log)
- [gui_train_20261022_173705.log](../runs/gui_train_20261022_173705.log)
- [gui_train_20261022_173715.log](../runs/gui_train_20261022_173715.log)
- [gui_train_20261022_173724.log](../runs/gui_train_20261022_173724.log)
- [gui_train_20261022_185652.log](../runs/gui_train_20261022_185652.log)
- [gui_train_20261022_185705.log](../runs/gui_train_20261022_185705.log)
- [gui_train_20261022_185711.log](../runs/gui_train_20261022_185711.log)
- [gui_train_20261022_203107.log](../runs/gui_train_20261022_203107.log)
- [gui_train_20261022_205759.log](../runs/gui_train_20261022_205759.log)
- [gui_train_20261022_210432.log](../runs/gui_train_20261022_210432.log)
- [gui_train_20261022_211639.log](../runs/gui_train_20261022_211639.log)
- [gui_train_20261022_211713.log](../runs/gui_train_20261022_211713.log)
- [gui_train_20261022_212344.log](../runs/gui_train_20261022_212344.log)
- [gui_train_20261023_133557.log](../runs/gui_train_20261023_133557.log)
- [gui_train_20261023_133557_2.log](../runs/gui_train_20261023_133557_2.log)
- [gui_train_20261023_133557_3.log](../runs/gui_train_20261023_133557_3.log)
- [gui_train_20261023_141521.log](../runs/gui_train_20261023_141521.log)
- [gui_train_20261023_141521_2.log](../runs/gui_train_20261023_141521_2.log)
- [gui_train_20261023_141521_3.log](../runs/gui_train_20261023_141521_3.log)
- [gui_train_20261023_145408.log](../runs/gui_train_20261023_145408.log)
- [gui_train_20261023_145408_2.log](../runs/gui_train_20261023_145408_2.log)
- [gui_train_20261023_145408_3.log](../runs/gui_train_20261023_145408_3.log)
- [gui_train_20261029_153623.log](../runs/gui_train_20261029_153623.log)
- [gui_train_20261029_230234.log](../runs/gui_train_20261029_230234.log)
- [gui_train_20261029_231441.log](../runs/gui_train_20261029_231441.log)
- [gui_train_20261029_235956.log](../runs/gui_train_20261029_235956.log)
- [gui_train_20261030_131036.log](../runs/gui_train_20261030_131036.log)
- [gui_train_20261030_131048.log](../runs/gui_train_20261030_131048.log)
- [gui_train_20261030_145136.log](../runs/gui_train_20261030_145136.log)
- [gui_train_20261103_143417_aeb1a5a5ef8e4fd3b3c8c3641ef4e638.log](../runs/gui_train_20261103_143417_aeb1a5a5ef8e4fd3b3c8c3641ef4e638.log) · [保存的启动配置](../runs/gui_train_20261103_143417_aeb1a5a5ef8e4fd3b3c8c3641ef4e638.config.json)
- [gui_train_20261103_143532_47236bd3b1a54c65b8a21d91ff0ff175.log](../runs/gui_train_20261103_143532_47236bd3b1a54c65b8a21d91ff0ff175.log) · [保存的启动配置](../runs/gui_train_20261103_143532_47236bd3b1a54c65b8a21d91ff0ff175.config.json)
- [gui_train_20261103_143547_6e6cf790345f4a0485ebe953cc8d86f3.log](../runs/gui_train_20261103_143547_6e6cf790345f4a0485ebe953cc8d86f3.log) · [保存的启动配置](../runs/gui_train_20261103_143547_6e6cf790345f4a0485ebe953cc8d86f3.config.json)
- [gui_train_20261103_143600_d83a443016984fdb9cf829ac8861dcd1.log](../runs/gui_train_20261103_143600_d83a443016984fdb9cf829ac8861dcd1.log) · [保存的启动配置](../runs/gui_train_20261103_143600_d83a443016984fdb9cf829ac8861dcd1.config.json)
- [gui_train_20261103_160202_bf2bf935d9724adf864d11dec95f8a38.log](../runs/gui_train_20261103_160202_bf2bf935d9724adf864d11dec95f8a38.log) · [保存的启动配置](../runs/gui_train_20261103_160202_bf2bf935d9724adf864d11dec95f8a38.config.json)
- [gui_train_20261104_134442_f6c64b44e31a4f17a9f7ac4c63a4d6a1.log](../runs/gui_train_20261104_134442_f6c64b44e31a4f17a9f7ac4c63a4d6a1.log) · [保存的启动配置](../runs/gui_train_20261104_134442_f6c64b44e31a4f17a9f7ac4c63a4d6a1.config.json)
- [gui_train_20261104_134451_42473cb2808a4778a8083be9f0bddb50.log](../runs/gui_train_20261104_134451_42473cb2808a4778a8083be9f0bddb50.log) · [保存的启动配置](../runs/gui_train_20261104_134451_42473cb2808a4778a8083be9f0bddb50.config.json)
- [gui_train_20261104_134457_1cce09ce97364935b792e34a88a2da47.log](../runs/gui_train_20261104_134457_1cce09ce97364935b792e34a88a2da47.log) · [保存的启动配置](../runs/gui_train_20261104_134457_1cce09ce97364935b792e34a88a2da47.config.json)
- [gui_train_20261105_154434_b2ba7b6af91147a0a9c87b5ed0becab6.log](../runs/gui_train_20261105_154434_b2ba7b6af91147a0a9c87b5ed0becab6.log) · [保存的启动配置](../runs/gui_train_20261105_154434_b2ba7b6af91147a0a9c87b5ed0becab6.config.json)
- [gui_train_20261105_162845_01f39e6092154f0ea99feec0f034e419.log](../runs/gui_train_20261105_162845_01f39e6092154f0ea99feec0f034e419.log) · [保存的启动配置](../runs/gui_train_20261105_162845_01f39e6092154f0ea99feec0f034e419.config.json)
- [gui_train_20261105_164826_c6fc23f3846e4e00a15c4b5f091ffd97.log](../runs/gui_train_20261105_164826_c6fc23f3846e4e00a15c4b5f091ffd97.log) · [保存的启动配置](../runs/gui_train_20261105_164826_c6fc23f3846e4e00a15c4b5f091ffd97.config.json)
- [gui_train_20261110_141641_d3b44bcd95dc405fba5fdbb5f9aa695a.log](../runs/gui_train_20261110_141641_d3b44bcd95dc405fba5fdbb5f9aa695a.log) · [保存的启动配置](../runs/gui_train_20261110_141641_d3b44bcd95dc405fba5fdbb5f9aa695a.config.json)
- [gui_train_20261110_141649_944ee60123d7463e9f011beacf976486.log](../runs/gui_train_20261110_141649_944ee60123d7463e9f011beacf976486.log) · [保存的启动配置](../runs/gui_train_20261110_141649_944ee60123d7463e9f011beacf976486.config.json)
- [gui_train_20261111_143220_657c5349437a4d8ead240575a347b354.log](../runs/gui_train_20261111_143220_657c5349437a4d8ead240575a347b354.log) · [保存的启动配置](../runs/gui_train_20261111_143220_657c5349437a4d8ead240575a347b354.config.json)

### 命令行训练（7 份）

- [cli_train_tcn_se_both_s67_h64.log](../runs/cli_train_tcn_se_both_s67_h64.log)
- [cli_train_tcn_se_la16_both_s54.log](../runs/cli_train_tcn_se_la16_both_s54.log)
- [cli_train_tcn_se_la16_both_s67.log](../runs/cli_train_tcn_se_la16_both_s67.log)
- [cli_train_tcn_se_la1_both_s54.log](../runs/cli_train_tcn_se_la1_both_s54.log)
- [cli_train_tcn_se_la1_both_s67.log](../runs/cli_train_tcn_se_la1_both_s67.log)
- [cli_train_tcn_se_la4_both_s54.log](../runs/cli_train_tcn_se_la4_both_s54.log)
- [cli_train_tcn_se_la4_both_s67.log](../runs/cli_train_tcn_se_la4_both_s67.log)

### 评估（2 份）

- [gui_eval_20260926_123603.log](../runs/gui_eval_20260926_123603.log)
- [gui_eval_20260926_124029.log](../runs/gui_eval_20260926_124029.log)

### 感受野（5 份）

- [gui_rfsweep_20261023_123229.log](../runs/gui_rfsweep_20261023_123229.log)
- [gui_rfsweep_20261027_111512.log](../runs/gui_rfsweep_20261027_111512.log)
- [gui_rfsweep_20261027_122340.log](../runs/gui_rfsweep_20261027_122340.log)
- [gui_rfsweep_20261027_150606.log](../runs/gui_rfsweep_20261027_150606.log)
- [gui_rfsweep_20261030_144141.log](../runs/gui_rfsweep_20261030_144141.log)

### Pi EEG-IO（5 份）

- [gui_eegio_pi_20261106_131944_552386.log](../runs/gui_eegio_pi_20261106_131944_552386.log)
- [gui_eegio_pi_20261106_132032_392195.log](../runs/gui_eegio_pi_20261106_132032_392195.log)
- [gui_eegio_pi_20261106_133637_392461.log](../runs/gui_eegio_pi_20261106_133637_392461.log)
- [gui_eegio_pi_20261106_160239_310763.log](../runs/gui_eegio_pi_20261106_160239_310763.log)
- [gui_eegio_pi_20261106_171216_055082.log](../runs/gui_eegio_pi_20261106_171216_055082.log)

### 部署（46 份）

- [gui_deploy_20261014_142816.log](../runs/gui_deploy_20261014_142816.log)
- [gui_deploy_20261014_143604.log](../runs/gui_deploy_20261014_143604.log)
- [gui_deploy_20261014_143654.log](../runs/gui_deploy_20261014_143654.log)
- [gui_deploy_20261014_150709.log](../runs/gui_deploy_20261014_150709.log)
- [gui_deploy_20261014_150843.log](../runs/gui_deploy_20261014_150843.log)
- [gui_deploy_20261014_170611.log](../runs/gui_deploy_20261014_170611.log)
- [gui_deploy_20261016_111800.log](../runs/gui_deploy_20261016_111800.log)
- [gui_deploy_20261016_113645.log](../runs/gui_deploy_20261016_113645.log)
- [gui_deploy_20261016_144943.log](../runs/gui_deploy_20261016_144943.log)
- [gui_deploy_20261016_154424.log](../runs/gui_deploy_20261016_154424.log)
- [gui_deploy_20261016_154453.log](../runs/gui_deploy_20261016_154453.log)
- [gui_deploy_20261016_154526.log](../runs/gui_deploy_20261016_154526.log)
- [gui_deploy_20261016_154714.log](../runs/gui_deploy_20261016_154714.log)
- [gui_deploy_20261016_155158.log](../runs/gui_deploy_20261016_155158.log)
- [gui_deploy_20261019_114557.log](../runs/gui_deploy_20261019_114557.log)
- [gui_deploy_20261019_114606.log](../runs/gui_deploy_20261019_114606.log)
- [gui_deploy_20261019_114635.log](../runs/gui_deploy_20261019_114635.log)
- [gui_deploy_20261019_114648.log](../runs/gui_deploy_20261019_114648.log)
- [gui_deploy_20261019_123202.log](../runs/gui_deploy_20261019_123202.log)
- [gui_deploy_20261019_123238.log](../runs/gui_deploy_20261019_123238.log)
- [gui_deploy_20261019_123403.log](../runs/gui_deploy_20261019_123403.log)
- [gui_deploy_20261019_161616.log](../runs/gui_deploy_20261019_161616.log)
- [gui_deploy_20261019_180642.log](../runs/gui_deploy_20261019_180642.log)
- [gui_deploy_20261019_180905.log](../runs/gui_deploy_20261019_180905.log)
- [gui_deploy_20261021_144859.log](../runs/gui_deploy_20261021_144859.log)
- [gui_deploy_20261021_144936.log](../runs/gui_deploy_20261021_144936.log)
- [gui_deploy_20261021_145038.log](../runs/gui_deploy_20261021_145038.log)
- [gui_deploy_20261021_145803.log](../runs/gui_deploy_20261021_145803.log)
- [gui_deploy_20261021_145836.log](../runs/gui_deploy_20261021_145836.log)
- [gui_deploy_20261021_145938.log](../runs/gui_deploy_20261021_145938.log)
- [gui_deploy_20261021_150059.log](../runs/gui_deploy_20261021_150059.log)
- [gui_deploy_20261021_151015.log](../runs/gui_deploy_20261021_151015.log)
- [gui_deploy_20261021_151045.log](../runs/gui_deploy_20261021_151045.log)
- [gui_deploy_20261021_155156.log](../runs/gui_deploy_20261021_155156.log)
- [gui_deploy_20261021_155239.log](../runs/gui_deploy_20261021_155239.log)
- [gui_deploy_20261021_155252.log](../runs/gui_deploy_20261021_155252.log)
- [gui_deploy_20261021_155609.log](../runs/gui_deploy_20261021_155609.log)
- [gui_deploy_20261021_155945.log](../runs/gui_deploy_20261021_155945.log)
- [gui_deploy_20261021_160316.log](../runs/gui_deploy_20261021_160316.log)
- [gui_deploy_20261021_161516.log](../runs/gui_deploy_20261021_161516.log)
- [gui_deploy_20261021_161543.log](../runs/gui_deploy_20261021_161543.log)
- [gui_deploy_20261021_161737.log](../runs/gui_deploy_20261021_161737.log)
- [gui_deploy_20261021_172609.log](../runs/gui_deploy_20261021_172609.log)
- [gui_deploy_20261021_172613.log](../runs/gui_deploy_20261021_172613.log)
- [gui_deploy_20261021_172729.log](../runs/gui_deploy_20261021_172729.log)
- [gui_deploy_20261103_164658.log](../runs/gui_deploy_20261103_164658.log)

### 多通道（6 份）

- [gui_channels_20261111_154759.log](../runs/gui_channels_20261111_154759.log)
- [gui_channels_20261111_155221.log](../runs/gui_channels_20261111_155221.log)
- [gui_channels_20261111_155611.log](../runs/gui_channels_20261111_155611.log)
- [gui_channels_20261111_160139.log](../runs/gui_channels_20261111_160139.log)
- [gui_channels_20261111_162158.log](../runs/gui_channels_20261111_162158.log)
- [gui_channels_20261111_162614.log](../runs/gui_channels_20261111_162614.log)
