# --------------------------
# src/config.py
# --------------------------
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    device: str
    generation_mode: str
    weights: dict
    target_mw: tuple
    target_logp: tuple
    vina_executable: str
    receptor_pdbqt: str
    temp_dir: str
    tensorboard_log: str
    docking_center: list
    box_size: list
    scaffold_smiles_file: str

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            data['docking_center'] = list(map(float, data['docking_center']))
            data['box_size'] = list(map(float, data['box_size']))
            data['target_mw'] = tuple(data['target_mw'])
            data['target_logp'] = tuple(data['target_logp'])
        return cls(**data)