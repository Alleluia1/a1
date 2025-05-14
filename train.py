import warnings, os, time
import wandb
from ultralytics import YOLO

# ✅ 登录 wandb（你只需配置一次）
wandb.login(key="f8cb8b13b090d70eb2b9b5ee36da161979b90a95")

# ✅ 初始化一个新的 wandb 运行
wandb.init(
    project="a1",
    name=time.strftime("%Y-%m-%d_%H-%M-%S")  # 自动根据时间命名
)

# ✅ 可选：设置显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings('ignore')

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
        name=wandb.run.name,   # ✅ 用 wandb 名字同步命名
        wandb=True             # ✅ 开启自动记录
    )
