# src/training.py
import torch
import csv
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from torch.utils.tensorboard import SummaryWriter
from src.config import Config
from src.environment import MolecularRLEnv

class MolGenCallback(BaseCallback):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.csv_path = f"molecules_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        self._init_csv()
        self.writer = SummaryWriter(config.tensorboard_log)

    def _init_csv(self):
        """初始化 CSV 文件，写入表头"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'smiles', 'vina', 'qed', 'sa', 'reward'])

    def _on_step(self) -> bool:
        """每100步记录一次分子信息和训练指标"""
        if self.n_calls % 100 == 0:
            infos = self.locals.get('infos', [])
            for info in infos:
                if info.get('valid', False):
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.num_timesteps,
                            info.get('smiles', ''),
                            f"{info.get('vina_score', -1):.2f}",
                            f"{info.get('qed', 0):.2f}",
                            f"{info.get('sa_score', 0):.2f}",
                            f"{info.get('reward', 0):.2f}"
                        ])
                    
                    # 记录到 TensorBoard
                    self.writer.add_scalar('VinaScore', info.get('vina_score', -1), self.num_timesteps)
                    self.writer.add_scalar('QED', info.get('qed', 0), self.num_timesteps)
                    self.writer.add_scalar('SA_Score', info.get('sa_score', 0), self.num_timesteps)
        
        # 记录训练指标
        if 'loss' in self.locals:
            self.logger.record('train/loss', self.locals['loss'])
        if 'entropy' in self.locals:
            self.logger.record('train/entropy', self.locals['entropy'])
        return True

class TrainingSystem:
    def __init__(self, config: Config):
        self.config = config
        self.env = DummyVecEnv([lambda: MolecularRLEnv(config)])
        self.model = None

    def initialize_model(self):
        """初始化 PPO 模型"""
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[512, 512]
        )
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.config.device,
            tensorboard_log=self.config.tensorboard_log
        )

    def train(self, total_steps=100000):
        """训练模型"""
        callbacks = [
            CheckpointCallback(save_freq=5000, save_path="./checkpoints/"),
            MolGenCallback(self.config)
        ]
        try:
            self.model.learn(
                total_timesteps=total_steps,
                callback=callbacks,
                reset_num_timesteps=False
            )
            self.model.save("final_model")
            print("Training completed successfully")
        except Exception as e:
            print(f"Training failed: {str(e)}")