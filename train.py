import warnings, os, time
import wandb
import torch
from ultralytics import YOLO

wandb.init(
    project='a1',
    name=time.strftime("%Y-%m-%d_%H-%M-%S"),
)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案


warnings.filterwarnings('ignore')

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!


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


# ✅ 训练完成后记录静态信息：参数量、FLOPs、模型大小、FPS
def log_static_model_info(model):
    # 参数量
    params = sum(p.numel() for p in model.model.parameters()) / 1e6  # M

    # FLOPs（估算推理复杂度）
    try:
        flops = model.model.fuse().profile()[1] / 1e9  # G
    except:
        flops = 0

    # 模型文件大小（MB）
    model_path = 'runs/train/exp/weights/best.pt'
    size_mb = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0

    # FPS（推理速度估算）
    dummy = torch.zeros((1, 3, 640, 640)).to(model.device)
    t0 = time.time()
    model.predict(dummy, verbose=False)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)

    wandb.log({
        'params_M': round(params, 2),
        'flops_G': round(flops, 2),
        'model_size_MB': round(size_mb, 2),
        'fps': round(fps, 2),
    })


if __name__ == '__main__':
    # 加载模型结构（你可以换成 'yolo11n.pt' 以加载预训练权重）
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')

    # 开始训练
    model.train(
        data='/root/workspace/d0cv1q7hri0c73e2gq5g/RDD2022_10000/data.yaml',
        cache=False,
        imgsz=640,
        epochs=300,
        batch=32,
        close_mosaic=0,
        workers=4,
        optimizer='SGD',
        project='a1',
        name=wandb.run.name,
        visualize=True,
        wandb=True,
        callbacks={'on_fit_epoch_end': on_fit_epoch_end},
    )

    # 训练完成后记录静态模型信息
    log_static_model_info(model)
