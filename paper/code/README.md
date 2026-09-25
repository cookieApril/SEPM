# 论文结果图表代码

结果图保存在 `paper/image/`，论文表直接位于 `paper/main.tex`，不再引用旧的 `paper/tables/` 或 `paper/sections/`。

方法总览图使用 `image/Fig1-GEMS-framework.pdf`。该文件由现有 Draw.io PDF 规范化为 PDF 1.5，以便论文编译器稳定嵌入；图中内容保持不变。

## 绘图

```powershell
python paper/code/draw-Fig2-E2-procedural-memory-model-sensitivity.py
python paper/code/draw-Fig3-effectiveness-evidence.py
```

需要 matplotlib。生成 Figure 2 的程序记忆模型敏感性图，以及 Figure 3 的主结果与配对结果图；两张图均提供 PDF、SVG 和 PNG。Figure 2 承接正文中唯一的综合方法图。

## E3 案例证据

```powershell
python paper/code/extract-E3-case-study.py
```

生成 `code/data/E3-case-study.json`，保存正文附录所用配对案例的原始结果路径、动作轨迹、评价器和事件计数。

## 更新数据表

```powershell
python paper/code/refresh-result-tables.py
```

只刷新 `main.tex` 中 `% BEGIN AUTO TABLE ...` 与 `% END AUTO TABLE ...` 标记间的六张表：三张主文实验表、三张附录结果表。正文、方法及前半篇不被重写。case 数表和配置默认值表是解释性表格，直接在正文维护。

`result_data.py` 从 E1 quality、E2 master、E3 master 构建展示值。刷新脚本额外重算 E2 全部 864 个原始 case 分数，并验证 12 个模型使用相同 case 集合。数据保存在 `code/data/E2-SOP-model-scores.csv` 和 `code/data/result-audit.json`。不运行 benchmark、不调用模型、不修改实验结果。

## 编译后半篇预览

```powershell
python paper/code/build-results-preview.py --engine tectonic
```

生成 `paper/Experiments-and-Appendix.pdf`，沿用论文样式，实验从 Section 5 / Figure 2 开始。首次 Tectonic 使用可能下载 TeX 包。构建临时文件放在项目 `tmp/paper-revision/`。

## 证据说明

图表描述现有记录，不将 endpoint 配置自动解释为已执行的 SOP 模型干预。E2 全部 864 条记录报告零检索，且没有 SOP 调用计数。E3 master 的 `divergence_recovery` / `sop_reuse_success` 历史字段实际聚合事件数 / 检索数，正文和附录已用真实语义命名。

视觉参考沿用用户提供的 G-Memory：Table 1 的 MAS 分块比较、Figure 4(a,b) 的敏感性分面和 Figure 4(c) 的组件开关表。所有实验数值来自当前项目，未复制参考论文结果。E2 图使用原生 rating 与百分比分开的坐标轴，并以点形配合颜色区分模型配置组。
