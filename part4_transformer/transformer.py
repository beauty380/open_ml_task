
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 基本设置
torch.manual_seed(42)

batch_size = 2
seq_len = 4
embed_dim = 8
num_heads = 2
num_layers = 2

# 结果保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# 构造输入
x = torch.randn(batch_size, seq_len, embed_dim)

# 原始 Transformer Encoder
print("========== 原始 Transformer ==========")

encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    batch_first=True
)

encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers
)

encoder.eval()

with torch.no_grad():
    output = encoder(x)

print("模型结构：")
print(encoder)

print("\nInput shape:", x.shape)
print("Output shape:", output.shape)
print("num_heads:", num_heads)
print("num_layers:", num_layers)

# 修改注意力头数量
print("\n========== 修改 num_heads ==========")

new_num_heads = 4

new_encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=new_num_heads,
    batch_first=True
)

new_encoder = nn.TransformerEncoder(
    new_encoder_layer,
    num_layers=num_layers
)

new_encoder.eval()

with torch.no_grad():
    new_output = new_encoder(x)

print("Input shape:", x.shape)
print("Output shape:", new_output.shape)
print("num_heads:", new_num_heads)
print("num_layers:", num_layers)

# 结果对比
print("\n========== 结果对比 ==========")

same_shape = output.shape == new_output.shape
same_values = torch.allclose(output, new_output)

print("原始输出 Shape:", output.shape)
print("修改后输出 Shape:", new_output.shape)
print("Shape 是否相同:", same_shape)
print("输出数值是否完全相同:", same_values)

# 保存实验结果图片
result_text = (
    "Transformer Encoder Experiment\n\n"
    "Original Model:\n"
    f"Input Shape:   {tuple(x.shape)}\n"
    f"Output Shape:  {tuple(output.shape)}\n"
    f"num_heads:     {num_heads}\n"
    f"num_layers:    {num_layers}\n\n"
    "Modified Model:\n"
    f"Input Shape:   {tuple(x.shape)}\n"
    f"Output Shape:  {tuple(new_output.shape)}\n"
    f"num_heads:     {new_num_heads}\n"
    f"num_layers:    {num_layers}\n\n"
    "Comparison:\n"
    f"Same Shape:    {same_shape}\n"
    f"Same Values:   {same_values}"
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")

ax.text(
    0.05,
    0.95,
    result_text,
    fontsize=13,
    fontfamily="monospace",
    verticalalignment="top",
    transform=ax.transAxes
)

plt.savefig(
    os.path.join(RESULT_DIR, "output.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n实验完成！")
print("结果图片已保存到 results/output.png")
