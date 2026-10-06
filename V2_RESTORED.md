# v2 代码与实验结果

本目录已恢复为 `ir_kernel_structural_v2_full10_01` 实际训练使用的源码。
快照包含的 634 个文件逐一按 SHA-256 核对；原有第三方补充文件及 Git 历史保留。

源码来源：
`/home/lf/code/Our_Project-New-2-1/output/ir_kernel_structural_v2_full10_01/code_snapshot/source`

完整十场景实验结果已复制到 `output/ir_kernel_structural_v2_full10_01/`，
包括模型、checkpoint、渲染图、六项指标、比较报告、源码快照和 CUDA 运行库。
这是既有实验的原样副本，历史元数据中的绝对路径仍指向原运行位置。
平均指标见其中的 `comparison.md`。

## 使用 v2 配置

先查看十场景执行计划；此命令不训练、不使用 GPU：

```bash
./run_v2.sh --dry-run
```

需要重新训练时执行：

```bash
./run_v2.sh
```

启动脚本使用 `thermalgaussian` 环境、归档的 v2 CUDA 扩展和本目录中的基线，
在 GPU 0 依次完成十个场景，每场从数据集初始化训练 30,000 步。
IR kernel 在 18,000 步启用，SSIM 损失权重为 0.01；此版本没有 v3 的 log-MSE 项。
新结果默认写入 `output/ir_kernel_structural_v2_rerun_<时间戳>/`。
可以通过 `--root output/<新目录名>` 指定输出目录，原有结果目录不能复用。

恢复前被替换的 5 个文件及 Git 差异保存在 `output/_backup_before_v2_*/`。
完整恢复记录见 `output/v2_restore_manifest.json`。
