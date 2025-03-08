# 安装依赖
pip install -r requirements.txt

# 运行训练
python src/main.py --config configs/config.yaml
TensorBoard查看:

bash
复制
tensorboard --logdir logs/

# 从项目根目录运行（注意路径）
PYTHONPATH=. python src/main.py --config config.yaml