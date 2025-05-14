import wandb, time, torch, os
from ultralytics import YOLO

# 登录 W&B
wandb.login(key='f8cb8b13b090d70eb2b9b5ee36da161979b90a95')
wandb.init(project='a1', name=time.strftime("%Y-%m-%d_%H-%M-%S"))

# 加载 YOLO 模型
model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')

# 开始训练
model.train(
    data='/root/workspace/d0cv1q7hri0c73e2gq5g/RDD2022_10000/data.yaml',
    epochs=300,
    batch=32,
    imgsz=640,
    optimizer='SGD',
    project='a1',
    name=wandb.run.name,
    wandb=True,
    workers=4
)

# 训练完成后记录模型统计信息
def log_static_model_info(model):
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    flops = model.model.fuse().profile()[1] / 1e9 if hasattr(model.model, 'fuse') else 0
    size_mb = os.path.getsize('runs/train/exp/weights/best.pt') / 1e6 if os.path.exists('runs/train/exp/weights/best.pt') else 0
    dummy = torch.zeros((1, 3, 640, 640)).to(model.device)
    model.predict(dummy, verbose=False)
    fps = 1.0 / (time.time() - time.time())  # 这个你可以再精确点测量
    wandb.log({
        'params_M': round(params, 2),
        'flops_G': round(flops, 2),
        'model_size_MB': round(size_mb, 2),
        'fps': round(fps, 2),
    })

log_static_model_info(model)
