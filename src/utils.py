# src/utils.py
import os
import subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from src.config import Config
from typing import Optional

# 导入 SA_Score 模块
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from sascorer import calculateScore

def run_docking(mol: Chem.Mol, config: Config) -> float:
    try:
        # 生成 3D 坐标
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # 保存配体文件
        os.makedirs(config.temp_dir, exist_ok=True)
        ligand_path = os.path.join(config.temp_dir, "ligand.pdbqt")
        Chem.MolToPDBFile(mol, ligand_path)
        
        # 执行对接
        cmd = [
            config.vina_executable,
            "--receptor", config.receptor_pdbqt,
            "--ligand", ligand_path,
            "--center_x", str(config.docking_center[0]),
            "--center_y", str(config.docking_center[1]),
            "--center_z", str(config.docking_center[2]),
            "--size_x", str(config.box_size[0]),
            "--size_y", str(config.box_size[1]),
            "--size_z", str(config.box_size[2])
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return parse_vina_result(result.stdout)
    except Exception as e:
        print(f"Docking failed: {str(e)}")
        return -7.0

def parse_vina_result(output: str) -> float:
    """解析 Vina 输出获取亲和力评分"""
    for line in output.split('\n'):
        if "Affinity" in line:
            try:
                return float(line.split()[1]) / 10.0
            except (IndexError, ValueError):
                continue
    return -7.0

def calculate_reward(mol: Chem.Mol, config: Config) -> float:
    """计算分子奖励值"""
    try:
        if mol is None or mol.GetNumAtoms() == 0:
            return -1.0
        
        # 对接评分
        vina_score = run_docking(mol, config)
        
        # 分子量
        mw = Descriptors.MolWt(mol)
        mw_penalty = 0.0 if config.target_mw[0] <= mw <= config.target_mw[1] else -1.0
        
        # LogP
        logp = Descriptors.MolLogP(mol)
        logp_reward = 1.0 - abs((logp - sum(config.target_logp)) / 2.0)
        
        # SA Score
        sa_score = 1.0 - (calculateScore(mol) / 10.0)
        
        # QED
        qed = QED.qed(mol)
        
        # 奖励计算组件
        components = [
            config.weights['vina'] * vina_score,
            config.weights['mw'] * mw_penalty,
            config.weights['logp'] * logp_reward,
            config.weights['sa'] * sa_score,
            config.weights['qed'] * qed
        ]
        
        # 检查组件是否有效
        if any(not np.isfinite(x) for x in components):
            return -1.0
        
        # 返回奖励值，限制在[-1, 1]范围内
        return float(np.clip(sum(components), -1.0, 1.0))
    except Exception as e:
        print(f"Reward calculation error: {str(e)}")
        return -1.0

def load_scaffolds(config: Config) -> list:
    """加载用户自定义的生长片段"""
    try:
        with open(config.scaffold_smiles_file, 'r') as f:
            return [line.strip().split()[0] for line in f 
                    if line.strip() and not line.startswith("#")]
    except Exception as e:
        raise RuntimeError(f"Failed to load scaffolds: {str(e)}")