# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import time
from Bio import Entrez
import pubchempy as pcp
import sys
import io
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 强制将标准输出和标准错误流设置为 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# === 1. 配置项 ===
# 从环境变量读取 PubMed 邮箱，如果没有则使用默认值
Entrez.email = os.getenv("PUBMED_EMAIL", "885258359@qq.com")
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "..", "results")
os.makedirs(BASE_DIR, exist_ok=True)

JSON_PATH = os.path.join(BASE_DIR, "literature_verification.json")
CSV_OUT_PATH = os.path.join(BASE_DIR, "literature_verification.csv")

# 验证性关键词（用于提高搜索结果的相关性）
VERIFY_KEYWORDS = "(interaction OR binding OR inhibition OR agonism OR targeting OR effect)"


def search_pubmed_for_pair(compound, target):
    """
    三阶梯度检索法：确保在保证相关性的前提下，最大化搜索成功率
    """
    # 1. 预处理：如果化合物名称太长（化学全称），尝试截取或提示
    # 对于像 2-(4-ethenyl... 这种，PubMed 很难处理，建议后续引入 PubChem 别名转换
    search_comp = compound
    if len(compound) > 40:
        # 如果是超长名称，尝试取前段或简化，这里我们先保持原样但记录日志
        pass

    # 定义三个等级的检索式
    queries = [
        # 等级 1: 精准检索 (成分+靶点+验证词)
        f'("{search_comp}"[TiAb] AND "{target}"[TiAb]) AND (interaction OR binding OR inhibition OR protein)',
        # 等级 2: 标准检索 (成分+靶点)
        f'("{search_comp}"[TiAb] AND "{target}"[TiAb])',
        # 等级 3: 宽泛检索 (直接关键词组合)
        f'{search_comp} {target}'
    ]

    for i, query in enumerate(queries):
        try:
            # 执行搜索
            handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
            record = Entrez.read(handle)
            handle.close()

            id_list = record["IdList"]

            if id_list:
                # 如果找到了结果，获取详情并返回
                print(f"    在第 {i + 1} 阶检索中找到 {len(id_list)} 篇文献")
                results = []
                handle = Entrez.esummary(db="pubmed", id=",".join(id_list))
                summaries = Entrez.read(handle)
                handle.close()

                for summary in summaries:
                    results.append({
                        "PMID": str(summary["Id"]),
                        "Title": summary["Title"],
                        "Link": f"https://pubmed.ncbi.nlm.nih.gov/{summary['Id']}/",
                        "Search_Level": i + 1  # 记录是在哪一阶搜到的
                    })
                return results

        except Exception as e:
            print(f"    第 {i + 1} 阶检索异常: {e}")
            continue

    return []  # 三阶都搜不到才返回空


def get_compound_synonym(cid):
    """
    通过 PubChem CID 获取最适合检索的俗名
    """
    if not cid or str(cid) == 'nan':
        return None

    try:
        # 获取同义词列表
        synonyms = pcp.get_synonyms(cid, 'cid')
        if not synonyms:
            return None

        # 获取该 CID 的所有同义词
        name_list = synonyms[0].get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])

        if not name_list:
            return None

        # 筛选逻辑：优先选择简短、纯英文、非 IUPAC 格式的名称
        # 1. 过滤掉过长的名称
        # 2. 过滤掉包含过多数字和括号的名称 (典型的 IUPAC)
        candidates = []
        for name in name_list[:15]:  # 只看前15个，通常最常用的在前面
            # 简单的俗名判断：长度适中，不含太多 '-' 和 '('
            if len(name) < 20 and name.count('(') == 0:
                candidates.append(name)

        # 如果有合适的候选，选最短的那个（通常是俗名）
        if candidates:
            return sorted(candidates, key=len)[0]

        # 如果没有合适的，返回第一个（通常是 PubChem 默认名称）
        return name_list[0]

    except Exception as e:
        print(f"    获取 CID {cid} 的别名失败: {e}")
        return None


def main():
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../../molecular-docking/docking_results/docking_results_summary.csv")

    if not os.path.exists(input_csv):
        print(f" 找不到输入文件: {input_csv}")
        return

    # 1. 加载历史记录 (实现断点续传)
    full_results = {}
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                full_results = json.load(f)
            print(f" 已加载 {len(full_results)} 条历史文献记录。")
        except Exception as e:
            print(f"加载旧 JSON 失败: {e}")

    # 2. 读取任务清单
    df = pd.read_csv(input_csv)
    df = df[df['Affinity(kcal/mol)'] < -6]  # 模拟agent自主判断，取结合能<-6的样本为显著样本，取显著样本的成分-靶点对进行检索
    print(f" 开始进行文献验证，共 {len(df)} 组样本...")

    for index, row in df.iterrows():
        # 兼容性列读取
        target = str(row['Target']).strip()
        raw_comp = str(row['Compound']).strip()
        cid = row.get('CID')

        task_id = f"{raw_comp}_{target}"

        # 3. 查重逻辑：如果已经搜到过文献，则跳过
        if task_id in full_results and full_results[task_id].get("Found_Count", 0) > 0:
            print(f" [{index + 1}/{len(df)}] {task_id} 已有文献，跳过。")
            continue

        if (index + 1) % 5 == 0 or (index + 1) == len(df):
            print(f" 进度: [{index + 1}/{len(df)}] 已处理...", end='\r')

        # 4. 获取化合物俗名 (Synonym) 提高搜索成功率
        search_comp = raw_comp
        if cid and str(cid) != 'nan':
            synonym = get_compound_synonym(cid)
            if synonym and len(synonym) < len(raw_comp):
                search_comp = synonym
                print(f"    识别到俗名: {search_comp}")

        # 5. 执行三阶梯度检索
        papers = search_pubmed_for_pair(search_comp, target)

        # 6. 存储并实时保存 JSON
        full_results[task_id] = {
            "Compound_Original": raw_comp,
            "Compound_Search": search_comp,
            "Target": target,
            "CID": str(cid),
            "Found_Count": len(papers),
            "Papers": papers,
            "Update_Time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False)

        # 遵守 API 频率限制
        time.sleep(0.5)

    # 任务完成后，只打印统计信息，不要打印文件路径（防止 Agent 去 type 路径）
    print(
        f"\n 检索完成。总计: {len(df)} 对，新发现文献: {len([v for v in full_results.values() if v['Found_Count'] > 0])} 对。")
    print("结果已存入指定数据库文件。")

    # 7. 任务全部完成后，导出扁平化 CSV 方便人类查看
    try:
        csv_rows = []
        for tid, info in full_results.items():
            if not info["Papers"]:
                csv_rows.append({
                    "Compound": info["Compound_Original"],
                    "Target": info["Target"],
                    "Evidence_Found": "No",
                    "PMID": "N/A",
                    "Paper_Title": "N/A",
                    "Search_Keyword": info["Compound_Search"]
                })
            else:
                for p in info["Papers"]:
                    csv_rows.append({
                        "Compound": info["Compound_Original"],
                        "Target": info["Target"],
                        "Evidence_Found": "Yes",
                        "PMID": p["PMID"],
                        "Paper_Title": p["Title"],
                        "Search_Keyword": info["Compound_Search"],
                        "Search_Level": p.get("Search_Level", "N/A")
                    })

        pd.DataFrame(csv_rows).to_csv(CSV_OUT_PATH, index=False, encoding='utf-8-sig')
        print("\n" + "=" * 40)
        print(f" 文献验证 Skill 运行完毕！")
        print(f" 详细 JSON (供 Agent 使用): {JSON_PATH}")
        print(f" 汇总 CSV (供人类使用): {CSV_OUT_PATH}")
        print("=" * 40)
    except PermissionError:
        print(f" 导出 CSV 失败：请关闭已打开的 {CSV_OUT_PATH} 文件。")


if __name__ == "__main__":
    main()
