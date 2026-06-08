# Skill: Molecular Docking Analysis

## Description

This skill performs molecular docking experiments using AutoDock Vina to predict the binding affinity (kcal/mol) between
compounds and target proteins.

## Usage Instructions

### 1. Running the Experiment

Always run the docking pipeline first to ensure results are up-to-date.

- **Default mode**: `python skills/molecular-docking/scripts/docking.py`
- **Custom mode**: `python skills/molecular-docking/scripts/docking.py --csv "path/to/your.csv"`

**Output Directory**: Results are saved to `skills/molecular-docking/results/`

### 2. Analyzing Results (CRITICAL FOR LARGE DATA)

When processing many molecules, follow this priority sequence:

1. **Priority 1 (Quick Scan)**: Look for the `--- AGENT_SUMMARY_START ---` block in the terminal output. It contains
   CSV-formatted rows of `Compound,Target,Affinity`. Use this data directly for the final report.
2. **Priority 2 (Detailed Check)**: If the summary is missing or you need full details (PDBID, Coordinates, Timestamps),
   find the line `JSON_OUTPUT_PATH: ...`.
3. **Command**: Use `type` (Windows) or `cat` (Linux) to read the JSON file path found above.

---

## Analysis Logic (Mandatory)

Apply these thresholds for every sample:

| Affinity Score (kcal/mol) | Interpretation            | Result Label            |
|:--------------------------|:--------------------------|:------------------------|
| **≤ -7.0**                | High binding potential    | **Strong Interaction**  |
| **> -7.0**                | Low or negligible binding | **Weak/No Interaction** |

---

## Output Format

Generate the final response in Chinese using this structure:

"
**分子对接实验报告**

1. **详细分析**:
    - **{Compound} & {Target}**: 结合能为 **{Affinity} kcal/mol**。
        - 结论：{上述逻辑判断的结论}。
    - ... (循环列出所有样本，如样本量超过 20 个，请优先列出结合能最强的 Top 10)

2. **总体统计**:
    - 本次实验共测试了 **{N}** 对样本。
    - 其中 **{M}** 对样本显示出显著的结合潜力（结合能 ≤ -7.0 kcal/mol）。
    - **总体趋势**: {总结一句话。例如：该批次化合物对靶点展现出普遍较强的亲和力 / 亲和力表现分化明显 / 整体亲和力较弱。}
      "