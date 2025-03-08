# src/environment.py
import numpy as np
import gym
from gym import spaces
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from typing import Optional, Tuple, Dict
from src.config import Config
from src.utils import load_scaffolds, run_docking, calculate_reward, calculateScore

class MolecularRLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.current_mol = None
        self.smiles_length = 100
        self._init_spaces()
        self._init_starting_molecule()

    def _init_spaces(self):
        """初始化动作和观察空间"""
        self.action_space = spaces.Discrete(10)
        self.char_set = self._get_char_set()
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(len(self.char_set), self.smiles_length),
            dtype=np.float32
        )

    def _get_char_set(self) -> list:
        """定义 SMILES 字符集"""
        base_chars = list("BCNOPSFBrCl[]()=#-@H\\/1234567890")
        return sorted(set(base_chars))

    def _init_starting_molecule(self):
        """初始化起始分子"""
        if self.config.generation_mode == "de_novo":
            self.current_mol = Chem.MolFromSmiles("C")
        elif self.config.generation_mode == "scaffold":
            self._load_custom_scaffold()
        else:
            raise ValueError(f"无效的生成模式: {self.config.generation_mode}")

    def _load_custom_scaffold(self):
        """加载用户自定义的生长片段"""
        scaffolds = load_scaffolds(self.config)
        if not scaffolds:
            raise ValueError("配置文件中未找到有效生长片段")
        
        selected = np.random.choice(scaffolds)
        self.current_mol = Chem.MolFromSmiles(selected)
        if self.current_mol is None:
            raise ValueError(f"无效的 SMILES 片段: {selected}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作并返回结果"""
        new_mol = self._modify_molecule(action)
        self.current_mol = new_mol if new_mol else self.current_mol
        
        is_valid = self._validate_molecule()
        reward = calculate_reward(self.current_mol, self.config) if is_valid else -1.0
        reward = np.clip(reward, -1.0, 1.0)  # 限制奖励范围
        
        obs = self._smiles_to_obs()
        done = not is_valid
        
        info = {
            'smiles': Chem.MolToSmiles(self.current_mol) if is_valid else "",
            'valid': is_valid,
            'vina_score': run_docking(self.current_mol, self.config) if is_valid else -1,
            'qed': QED.qed(self.current_mol) if is_valid else 0,
            'sa_score': 1.0 - (calculateScore(self.current_mol) / 10.0) if is_valid else 0
        }
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """重置环境"""
        self._init_starting_molecule()
        return self._smiles_to_obs()

    def _smiles_to_obs(self) -> np.ndarray:
        """将 SMILES 字符串转换为观察值"""
        smiles = Chem.MolToSmiles(self.current_mol) if self.current_mol else ""
        smiles = smiles.ljust(self.smiles_length)[:self.smiles_length]
        
        obs = np.zeros((len(self.char_set), self.smiles_length), dtype=np.float32)
        for i, char in enumerate(smiles):
            if char in self.char_set:
                obs[self.char_set.index(char), i] = 1.0
        return obs

    def _modify_molecule(self, action: int) -> Optional[Chem.Mol]:
        """根据动作修改分子"""
        try:
            if action < 0 or action >= self.action_space.n:
                return None
            
            new_mol = Chem.RWMol(self.current_mol)
            # 分子修改逻辑保持不变
            # ...
            return Chem.Mol(new_mol)
        except Exception as e:
            print(f"分子修改失败: {str(e)}")
            return None

    def _validate_molecule(self) -> bool:
        """验证分子有效性"""
        if self.current_mol is None:
            return False
        try:
            Chem.SanitizeMol(self.current_mol)
            return True
        except:
            return False