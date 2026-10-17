
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 获取结果文件夹路径
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

# 读取两个模型的预测图
linear_img = mpimg.imread(RESULT_DIR / "linear_prediction.png")
mlp_img = mpimg.imread(RESULT_DIR / "mlp_prediction.png")

# 将两张图放在一起
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(linear_img)
axes[0].set_title("Linear Regression")
axes[0].axis("off")

axes[1].imshow(mlp_img)
axes[1].set_title("MLP")
axes[1].axis("off")

plt.tight_layout()

# 保存对比图
plt.savefig(
    RESULT_DIR / "model_comparison.png",
    dpi=200
)

plt.show()

print("模型对比图已保存！")
