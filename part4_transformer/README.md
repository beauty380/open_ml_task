# Part 4：Transformer 学习与实践

## 一、任务内容

本次任务主要是学习 Transformer 的基本原理，并使用 PyTorch 完成一个简单的 Transformer Encoder 实验。

任务分为两部分：

1. **理论学习**：阅读《Attention Is All You Need》，了解 Attention、多头注意力、位置编码以及 Encoder 和 Decoder 等基本知识，并完成理论问答。
2. **代码实践**：使用 PyTorch 搭建 Transformer Encoder，观察输入和输出的变化，并尝试修改模型参数。

## 二、代码实现

代码文件：`transformer.py`

主要完成了以下内容：

1. 创建一个形状为 `(2, 4, 8)` 的随机输入张量。
2. 使用 PyTorch 自带的组件搭建 Transformer Encoder。
3. 将输入数据传入模型，查看输出结果。
4. 将注意力头数 `num_heads` 从 2 修改为 4，再次运行模型。
5. 对比修改前后的输出形状和数值。
6. 将实验结果保存为图片。

## 三、实验结果

| 对比内容 | 修改前 | 修改后 |
|---|---|---|
| 输入形状 | (2, 4, 8) | (2, 4, 8) |
| 输出形状 | (2, 4, 8) | (2, 4, 8) |
| 注意力头数 | 2 | 4 |
| Encoder 层数 | 2 | 2 |

通过实验发现：

- 修改注意力头数后，模型的输出形状没有变化。
- 两个模型的输出数值通常不同。
- 注意力头数会影响模型内部处理信息的方式，但不会直接改变输出的整体维度。

## 四、运行方式

安装所需依赖：

```bash
pip install torch matplotlib
```

运行代码：

```bash
python transformer.py
```

运行后，终端会显示模型结构、输入输出形状及对比结果。

实验结果图片保存在 `results/output.png` 中。

## 五、项目结构

```text
part4_transformer/
├── transformer.py
├── README.md
└── results/
    └── output.png
```

## 六、学习总结

通过这次任务，我初步了解了 Transformer 的基本结构以及 Attention 的作用，也理解了为什么 Transformer 能够更好地处理序列数据。

在代码实践中，我学会了使用 PyTorch 搭建简单的 Transformer Encoder，并通过修改参数观察模型的变化。

这次实验让我对 Transformer 的结构有了更直观的认识，也为后续学习更复杂的模型打下了基础。
