import os
import time
import torch
import wandb
import warnings
from ultralytics import YOLO

# ✅ 创建运行名
run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

# ✅ 登录并初始化 wandb
wandb.login(key="f8cb8b13b090d70eb2b9b5ee36da161979b90a95")
wandb.init(project="a1", name=run_name)

warnings.filterwarnings('ignore')

# ✅ 每个 epoch 结束时记录指标
def on_fit_epoch_end(trainer):
    metrics = trainer.metrics
    wandb.log({
        'epoch': trainer.epoch,
        'precision': metrics.get('metrics/precision(B)', 0),
        'recall': metrics.get('metrics/recall(B)', 0),
        'map50': metrics.get('metrics/mAP50(B)', 0),
        'map75': metrics.get('metrics/mAP75(B)', 0),
        'map50-95': metrics.get('metrics/mAP50-95(B)', 0),
    })

# ✅ 训练结束后记录静态模型指标
def log_static_model_info(model):
    try:
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        try:
            flops = model.model.fuse().profile()[1] / 1e9
        except:
            flops = 0
        weight_path = f'runs/train/{run_name}/weights/best.pt'
        size_mb = os.path.getsize(weight_path) / 1e6 if os.path.exists(weight_path) else 0
        dummy = torch.zeros((1, 3, 640, 640)).to(model.device)
        start = time.time()
        model.predict(dummy, verbose=False)
        end = time.time()
        fps = 1.0 / (end - start) if end > start else 0

        wandb.log({
            "params_M": round(params, 2),
            "flops_G": round(flops, 2),
            "model_size_MB": round(size_mb, 2),
            "fps": round(fps, 2)
        })
        print("✅ 模型静态信息已记录到 WandB")
    except Exception as e:
        print("⚠️ 记录模型信息失败：", e)

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    model.train(
        data='/root/workspace/d0cv1q7hri0c73e2gq5g/RDD2022_10000/data.yaml',
        cache=False,
        imgsz=640,
        epochs=300,
        batch=32,
        close_mosaic=0,
        workers=4,
        optimizer='SGD',
        project='runs/train',
        name=run_name,
        callbacks={"on_fit_epoch_end": on_fit
