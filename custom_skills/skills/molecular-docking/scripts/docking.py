# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBList
from Bio.PDB import PDBParser, PDBIO, Select
import re
import pubchempy as pcp
import ssl
import time
import subprocess
from openbabel import pybel
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
import warnings
import json
import argparse
import sys
import io
import requests
from tqdm import tqdm

# 强制将标准输出和标准错误流设置为 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 忽略特定的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 屏蔽所有 OpenBabel 警告
pybel.ob.obErrorLog.SetOutputLevel(0)

# === 1. 路径配置 ===
# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取技能根目录 (molecular-docking)
skill_root_dir = os.path.dirname(script_dir)
# 定义缓存目录 (存放 PDB/SDF 等中间文件)
BASE_DIR = os.path.join(skill_root_dir, "cache")
RECEPTOR_DIR = os.path.join(BASE_DIR, "receptors")  # 存放靶点 PDB/PDBQT
LIGAND_DIR = os.path.join(BASE_DIR, "ligands")  # 存放成分 SDF

# 确保文件夹存在
for d in [RECEPTOR_DIR, LIGAND_DIR]:
    os.makedirs(d, exist_ok=True)


# === 2. 准备蛋白质PDB数据 ===
class ProteinSelect(Select):
    """自定义选择器：只保留标准氨基酸，剔除配体、水分子和杂原子"""

    def accept_residue(self, residue):
        # get_id()[0] 为 ' ' 表示标准氨基酸
        return residue.get_id()[0] == ' '


def download_clean_target_pdb(pdb_id, save_path):
    """
    下载 PDB 并自动提纯（仅保留蛋白骨架）
    pdb_id: 例如 "5RE4"
    save_path: 最终提纯后的保存路径，例如 "./receptors/target_A.pdb"
    """
    # 1. 准备目录
    target_dir = os.path.dirname(save_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"正在下载并提纯靶点 PDB ID: {pdb_id} ...")
    pdbl = PDBList()

    try:
        # 2. 下载原始 PDB 文件 (通常为 .ent 格式)
        raw_file_path = pdbl.retrieve_pdb_file(pdb_id, pdir=target_dir, file_format='pdb', overwrite=True)

        if not os.path.exists(raw_file_path):
            print(f"下载失败：无法获取 {pdb_id}")
            return False

        # 3. 提纯蛋白 (执行 Clean 逻辑)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', raw_file_path)
        io = PDBIO()
        io.set_structure(structure)

        # 将提纯后的结构直接保存到最终目标路径 save_path
        io.save(save_path, ProteinSelect())

        # 4. 清理痕迹：删除下载的原始未提纯文件 (.ent 文件)
        if os.path.exists(raw_file_path):
            os.remove(raw_file_path)

        print(f"[Success] Cleaning finished!The final protein structure file is saved in: {save_path}\n")

        return True

    except Exception as e:
        print(f"处理过程中出现异常: {e}")
        return False


# === 3. 准备成分SDF数据 ===

# 全局禁用 SSL 证书检查，解决 [SSL: UNEXPECTED_EOF_WHILE_READING] 报错
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符。"""
    if name is None: return "compound"
    name = name.strip()
    name = re.sub(r'[\\/:"*?<>|]+', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name or "compound"


def download_compound_sdf(cid, save_path, retries=3):
    """
    直接从 PubChem 下载 2D SDF 文件，带有 SSL 容错和重试机制。
    """
    target_dir = os.path.dirname(save_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    print(f"正在尝试从 PubChem 下载 CID: {cid} (2D 模式)...")

    for i in range(retries):
        try:
            # 逻辑修改：直接固定下载 record_type='2d'
            pcp.download('SDF', save_path, identifier=str(cid), overwrite=True, record_type='2d')

            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f" 第 {i + 1} 次尝试成功 文件下载到：{save_path}\n")
                return True
            else:
                print(f"⚠ 第 {i + 1} 次尝试文件为空，重试中...")

        except Exception as e:
            print(f"️ 第 {i + 1} 次尝试失败，错误原因: {e}")
            if i < retries - 1:
                time.sleep(2)  # 等待 2 秒后重试
            else:
                print(f" 经过 {retries} 次尝试后仍然下载失败。")
                return False
    return False


# === 4. 分子对接 ===


def run_molecular_docking(target_name, target_pdb_path, compound_name, compound_sdf_path, results_dir):
    """
    执行单个靶点与成分的分子对接，修复了 tuple 报错和 TML 电荷溢出问题
    """
    print(f"\n 正在启动对接: {target_name} <-> {compound_name}")

    vina_exe = "vina"

    task_id = f"{target_name}_{compound_name}"
    # 使用相对路径 ../docking_results 确保指向正确目录
    prot_pdbqt = os.path.join(results_dir, f"{target_name}_temp.pdbqt")
    lig_pdbqt = os.path.join(results_dir, f"{compound_name}_temp.pdbqt")
    output_pdbqt = os.path.join(results_dir, f"{task_id}_out.pdbqt")

    # 确保输出目录存在
    os.makedirs(results_dir, exist_ok=True)

    try:
        # === 1. 蛋白质预处理 (修复 write tuple 报错) ===
        if not os.path.exists(target_pdb_path):
            print(f" 找不到蛋白质文件: {target_pdb_path}")
            return None

        mol = next(pybel.readfile("pdb", target_pdb_path))
        mol.addh()

        # --- [修复 1] 兼容性处理：无论 write 返回字符串还是元组，都强转为字符串 ---
        pdbqt_raw = mol.write("pdbqt")
        pdbqt_str = pdbqt_raw[0] if isinstance(pdbqt_raw, tuple) else pdbqt_raw
        pdbqt_content = pdbqt_str.split('\n')
        # ---------------------------------------------------------------------

        with open(prot_pdbqt, 'w') as f:
            for line in pdbqt_content:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    f.write(line + '\n')
            f.write("TER\n")

        # === 2. 成分预处理 (修复 TML 电荷 NaN 问题) ===
        if not os.path.exists(compound_sdf_path):
            print(f" 找不到成分文件: {compound_sdf_path}")
            return None

        rd_suppl = Chem.SDMolSupplier(compound_sdf_path)
        rd_mol = rd_suppl[0]
        if rd_mol is None:
            print(" RDKit 无法读取 SDF 文件")
            return None

        # ========================================================
        # 【在此处插入：广谱修复方案】
        # 作用：在生成3D坐标前，把非法原子(如Pb)改为常见原子(如N/C)
        # ========================================================
        valid_element_symbols = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se']
        for atom in rd_mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in valid_element_symbols or atom.GetAtomicNum() > 53:
                # 针对 TML 这类季铵盐的逻辑：如果这个怪原子连接了 4 个及以上原子，大概率是氮 (N)
                if len(atom.GetNeighbors()) >= 4:
                    atom.SetAtomicNum(7)  # 设为氮
                    atom.SetFormalCharge(1)  # 季铵盐正电荷
                else:
                    atom.SetAtomicNum(6)  # 否则统一设为碳
                    atom.SetFormalCharge(0)
        # ========================================================

        rd_mol = Chem.AddHs(rd_mol)
        from rdkit.Chem import AllChem

        # 生成初始 3D 构象
        params = AllChem.ETKDG()
        params.useRandomCoords = True
        AllChem.EmbedMolecule(rd_mol, params)

        # --- [修复 2] 防御性优化策略 ---
        # 尝试优化，如果力场不支持（如 TML），则捕获异常并跳过优化
        try:
            if AllChem.MMFFGetMoleculeForceField(rd_mol, AllChem.MMFFGetMoleculeProperties(rd_mol)):
                AllChem.MMFFOptimizeMolecule(rd_mol)
            else:
                AllChem.UFFOptimizeMolecule(rd_mol)
        except Exception:
            # 如果优化报错，静默失败，使用初始构象
            pass

        # [关键一步] 强制计算 Gasteiger 电荷
        # 这会覆盖掉可能因为优化失败而产生的 NaN 电荷，确保 Meeko 不会崩溃
        AllChem.ComputeGasteigerCharges(rd_mol)
        # --------------------------------

        preprocs = MoleculePreparation()
        mol_setups = preprocs.prepare(rd_mol)

        if not mol_setups:
            print(" Meeko 转换失败：未生成有效 setup")
            return None

        # 获取小分子的 PDBQT 内容
        lig_raw = PDBQTWriterLegacy.write_string(mol_setups[0])
        # 关键修复：如果是元组，取第一个元素（字符串）
        lig_pdbqt_str = lig_raw[0] if isinstance(lig_raw, tuple) else lig_raw

        with open(lig_pdbqt, "w") as f:
            f.write(lig_pdbqt_str)

        # === 3. 动态计算盒子参数 ===
        coords = np.array([atom.coords for atom in mol.atoms])
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        center = (min_coords + max_coords) / 2
        size = np.clip((max_coords - min_coords) + 10.0, 20.0, 100.0)

        print(f" 任务盒子参数 [{task_id}]:")
        print(f"   - 中心 Center: {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}")
        print(f"   - 尺寸 Size  : {size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}")

        # === 4. 调用 Vina ===
        cmd = [
            vina_exe,
            "--receptor", str(prot_pdbqt),
            "--ligand", str(lig_pdbqt),
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{size[0]:.1f}",
            "--size_y", f"{size[1]:.1f}",
            "--size_z", f"{size[2]:.1f}",
            "--out", str(output_pdbqt),
            "--exhaustiveness", "32"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # === 5. 解析结果 ===
        affinity = None
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "   1  " in line:
                    affinity = float(line.split()[1])
                    break
        else:
            print(f" Vina 运行报错 [{task_id}]: {result.stderr}")

        # 清理临时文件
        for tmp_file in [prot_pdbqt, lig_pdbqt]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        if affinity is not None:
            print(f" {task_id} 对接完成 | 结合能: {affinity} kcal/mol")
        else:
            print(f"️ {task_id} 对接未产生有效结果")

        return {
            "affinity": affinity,
            "center_x": round(center[0], 3), "center_y": round(center[1], 3), "center_z": round(center[2], 3),
            "size_x": round(size[0], 1), "size_y": round(size[1], 1), "size_z": round(size[2], 1)
        }

    except Exception as e:
        print(f" {task_id} 运行异常: {e}")
        import traceback
        traceback.print_exc()  # 打印详细报错方便最后检查
        return None


def fetch_cid(name):
    """根据成分名查询 CID"""
    if pd.isna(name) or str(name).strip() == "": return "Not Found"
    try:
        results = pcp.get_compounds(name, 'name')
        return results[0].cid if results else "Not Found"
    except Exception:
        return "Error"


def fetch_best_pdb(target_name):
    """查询分辨率最优的 PDB ID"""
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": target_name}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 5}}
    }
    try:
        response = requests.post(search_url, json=query, timeout=10)
        if response.status_code != 200: return "Not Found"
        ids = [item["identifier"] for item in response.json().get("result_set", [])]
        if not ids: return "Not Found"

        best_id, min_res = ids[0], 99.0
        for pdb_id in ids:
            res_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            r = requests.get(res_url, timeout=5).json()
            res = r.get("rcsb_entry_info", {}).get("resolution_combined", [99.0])[0]
            if res < min_res:
                min_res, best_id = res, pdb_id
        return best_id
    except:
        return "Not Found"


def generate_agent_summary(results_dict):
    """
    生成一个极简的摘要字符串，只包含核心字段，便于 Agent 处理大数据量。
    """
    summary_lines = ["--- AGENT_SUMMARY_START ---"]
    # 添加表头
    summary_lines.append("Compound,Target,Affinity(kcal/mol)")

    for key, data in results_dict.items():
        line = f"{data['Compound']},{data['Target']},{data['Affinity(kcal/mol)']}"
        summary_lines.append(line)

    summary_lines.append("--- AGENT_SUMMARY_END ---")
    return "\n".join(summary_lines)


# === 5. 主流程逻辑 ===

def main():
    import argparse  # 引入参数解析库

    # === 1. 路径与参数配置 ===

    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录 (molecular-docking)
    skill_root_dir = os.path.dirname(script_dir)

    # 定义结果输出目录
    results_dir = os.path.join(skill_root_dir, "docking_results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "docking_results_summary.json")
    csv_out_path = os.path.join(results_dir, "docking_results_summary.csv")

    # 定义默认的 CSV 输入路径
    default_csv_path = os.path.abspath(os.path.join(skill_root_dir, "../data/top_predicted_pairs_new.csv"))

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run molecular docking pipeline.")
    parser.add_argument('--csv', type=str, default=default_csv_path,
                        help='Path to the input CSV file containing target and compound pairs.')
    args = parser.parse_args()

    csv_path = args.csv

    # 检查输入文件是否存在
    if not os.path.exists(csv_path):
        print(f"找不到输入文件: {csv_path}")
        print(f"   (尝试读取的路径是: {csv_path})")
        return

    print(f"[Input] Reading CSV: {csv_path}")
    print(f"结果将输出到: {results_dir}")

    # === 2. 初始化结果字典 (断点续传) ===
    full_results_dict = {}

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                full_results_dict = json.load(f)
            print(f"[Info] Found existing records, {len(full_results_dict)} historical tasks have been loaded.")
        except Exception as e:
            print(f"读取旧 JSON 失败（可能是文件损坏），将重新开始: {e}")

    # === 3. 读取任务并循环 ===
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取 CSV 失败: {e}");
        return

        # --- [新增逻辑] 自动补全 CID 和 PDBID ---
    print("正在检查并补全 CID 和 PDBID...")

    # 如果列不存在则创建
    if 'CID' not in df.columns: df['CID'] = None
    if 'PDBID' not in df.columns: df['PDBID'] = None

    needs_save = False
    for index, row in tqdm(df.iterrows(), total=len(df), desc="数据补全进度"):
        # 补全 CID (基于 Compound 列)
        if pd.isna(row['CID']) or str(row['CID']).lower() in ['nan', 'not found', '']:
            df.at[index, 'CID'] = fetch_cid(row['Compound'])
            needs_save = True

        # 补全 PDBID (基于 Target 列)
        if pd.isna(row['PDBID']) or str(row['PDBID']).lower() in ['nan', 'not found', '']:
            df.at[index, 'PDBID'] = fetch_best_pdb(row['Target'])
            needs_save = True

    if needs_save:
        # 另存为新文件，并把当前 df 指向这个新文件
        new_csv_path = csv_path.replace(".csv", "_with_IDs.csv")
        df.to_csv(new_csv_path, index=False, encoding='utf-8-sig')
        print(f"已生成补全后的 CSV 文件: {new_csv_path}")
    # ----------------------------------------------

    print(f"共有 {len(df)} 个任务待处理...")

    for index, row in df.iterrows():
        target_name = str(row['Target'])
        compound_name = str(row['Compound'])
        pdb_id = str(row['PDBID'])
        cid = str(row['CID'])

        # === 查重判断 ===
        task_id = f"{target_name}_{compound_name}"

        # === 1. 新增：前置过滤逻辑 ===
        # 如果 PDBID 为空或为 'Not Found'，直接在下载前就跳过
        if pd.isna(row['PDBID']) or pdb_id.lower() in ['Not Found', 'nan', 'none', '']:
            print(f"任务 [{index + 1}/{len(df)}]: {task_id} 因无 PDBID 直接跳过。")
            continue

        if task_id in full_results_dict:
            if full_results_dict[task_id].get("Affinity(kcal/mol)") is not None:
                print(f"任务 [{index + 1}/{len(df)}]: {task_id} 已完成，自动跳过。")
                continue

        print(f"\n任务 [{index + 1}/{len(df)}]: 处理 {target_name} 与 {compound_name}")

        # --- 数据准备 ---
        target_pdb_path = os.path.join(RECEPTOR_DIR, f"{target_name}.pdb")
        if not os.path.exists(target_pdb_path):
            success = download_clean_target_pdb(pdb_id, target_pdb_path)
            if not success:
                print(f"  ❌ 靶点 {target_name} 下载失败，跳过对接。")
                continue
        else:
            print(f"  靶点数据已存在。")

        compound_sdf_path = os.path.join(LIGAND_DIR, f"{compound_name}.sdf")
        if not os.path.exists(compound_sdf_path):
            download_compound_sdf(cid, compound_sdf_path)
        else:
            print(f"成分数据已存在。")

        # --- 分子对接核心逻辑 ---
        try:
            docking_res = run_molecular_docking(target_name, target_pdb_path, compound_name, compound_sdf_path,
                                                results_dir)

            if docking_res is None:
                print(f"任务中止: {task_id} 对接环境构建失败")
                continue

            entry = {
                "Compound": compound_name,
                "Target": target_name,
                "PDBID": pdb_id,
                "CID": cid,
                "Affinity(kcal/mol)": docking_res["affinity"],
                "Box": {
                    "Center": [docking_res["center_x"], docking_res["center_y"], docking_res["center_z"]],
                    "Size": [docking_res["size_x"], docking_res["size_y"], docking_res["size_z"]]
                },
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            full_results_dict[task_id] = entry
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(full_results_dict, f, indent=4, ensure_ascii=False)

            if docking_res["affinity"] is None:
                print(f"警告: {task_id} 未产生能量分数，请检查小分子结构。")

        except Exception as e:
            print(f"对接异常: {task_id}, 错误详情: {e}")

    # === 4. 导出汇总 CSV 与 精简摘要 ===
    try:
        csv_list = []
        for val in full_results_dict.values():
            item = val.copy()
            if "Box" in item:
                box = item.pop("Box")
                item["Center_X"], item["Center_Y"], item["Center_Z"] = box["Center"]
                item["Size_X"], item["Size_Y"], item["Size_Z"] = box["Size"]
            csv_list.append(item)

        summary_df = pd.DataFrame(csv_list)
        summary_df.to_csv(csv_out_path, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 40)
        print(f"DONE: All tasks completed.")
        print(f"JSON 结果: {json_path}")
        print(f"CSV 结果: {csv_out_path}")
        print("=" * 40)

        # === [核心新增] 生成供 Agent 分析的精简块 ===
        if full_results_dict:
            # 调用之前添加的 generate_agent_summary 函数
            agent_summary = generate_agent_summary(full_results_dict)
            print("\nREDUCED SUMMARY FOR AI ANALYSIS:")
            print(agent_summary)
            print("=" * 40)

        # 供 Agent 读取的标记
        print(f"\nEXECUTION_COMPLETE")
        print(f"JSON_OUTPUT_PATH: {os.path.abspath(json_path)}")

    except PermissionError:
        print(f"\n错误：无法保存 CSV，请关闭文件后重试。")
        print(f"JSON_OUTPUT_PATH: {os.path.abspath(json_path)}")
    except Exception as e:
        print(f"\n保存结果时发生意外错误: {e}")


if __name__ == "__main__":
    main()
