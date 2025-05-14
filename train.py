import wandb, time, torch, os
wandb.login(key='f8cb8b13b090d70eb2b9b5ee36da161979b90a95')

from ultralytics import YOLO


wandb.init(project='a1', name=time.strftime("%Y-%m-%d_%H-%M-%S"))

model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
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

# 后处理统计
def log_static_model_info(model):
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    flops = model.model.fuse().profile()[1] / 1e9 if hasattr(model.model, 'fuse') else 0
    size_mb = os.path.getsize('runs/train/exp/weights/best.pt') / 1e6 if os.path.exists('runs/train/exp/weights/best.pt') else 0
    dummy = torch.zeros((1, 3, 640, 640)).to(model.device)
    fps = 1.0 / (time.time() - time.time()) if model.predict(dummy, verbose=False) else 0
    wandb.log({
        'params_M': round(params, 2),
        'flops_G': round(flops, 2),
        'model_size_MB': round(size_mb, 2),
        'fps': round(fps, 2),
    })

log_static_model_info(model)
