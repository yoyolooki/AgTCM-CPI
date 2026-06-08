# TCM-Skill-Agent

基于 Anthropic Claude Agent 的中药复方分子机制分析平台，主要围绕**当归芍药散**进行多维度药理分析。

## 项目结构

```
custom_skills/
├── .env                      # 统一配置文件 (API keys, R path等)
├── run_skill.py              # 主入口：智能技能选择与执行引擎
└── skills/                   # 核心技能目录
    ├── data/                 # 共享输入数据
    │   └── top_predicted_pairs_new.csv
    ├── molecular-docking/     # 分子对接
    │   ├── SKILL.md
    │   ├── scripts/
    │   │   └── docking.py
    │   ├── cache/            # 缓存 PDB/SDF 等中间文件
    │   │   ├── receptors/
    │   │   └── ligands/
    │   └── results/          # 对接结果
    ├── enrichment-analysis/   # KEGG富集分析
    │   ├── SKILL.md
    │   ├── scripts/
    │   │   ├── enrichment.py
    │   │   └── run_enrichment.R
    │   └── results/          # 富集分析结果
    └── literature_search/     # 文献验证
        ├── SKILL.md
        ├── scripts/
        │   └── searching.py
        └── results/          # 文献检索结果
```

## 核心技能

| 技能 | 功能 | 输入 | 输出 |
|:---|:---|:---|:---|
| **molecular-docking** | 分子对接实验 | CSV (Compound, Target, PDBID, CID) | JSON + 终端摘要 |
| **enrichment-analysis** | KEGG通路富集 | CSV (Target列) | JSON + Markdown表格 + 气泡图 |
| **literature_search** | PubMed文献验证 | CSV (Compound, Target, CID) | JSON + CSV |

## 使用方法

### 1. 配置环境

编辑 `custom_skills/.env` 文件，配置必要的路径：

```bash
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
ANTHROPIC_MODEL=glm-4.7
R_EXECUTABLE=E:/Development Language/R/R-4.5.1/bin/Rscript.exe
PUBMED_EMAIL=your_email@example.com
```

### 2. 运行技能

#### 方式一：通过主引擎运行（推荐）

```bash
cd custom_skills
python run_skill.py "对数据进行分子对接分析"
```

#### 方式二：直接运行单个技能

```bash
# 分子对接
python skills/molecular-docking/scripts/docking.py

# 富集分析
python skills/enrichment-analysis/scripts/enrichment.py

# 文献检索
python skills/literature_search/scripts/searching.py
```

**注意**: 请使用 Python 3.7+ 运行，本项目使用 f-string 等现代 Python 特性。

## 环境依赖

- Python 3.7+
- R (clusterProfiler)
- AutoDock Vina
- OpenBabel / Meeko
- RDKit

详细依赖请参考各技能目录下的 SKILL.md 文件。
