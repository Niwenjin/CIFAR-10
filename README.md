# CIFAR-10

使用 Resnet 模型在 CIFAR-10 上训练高精度模型。

使用基础模型, Adam, batch_size=128, epochs=500, lr=1e-3 初步验证模型性能。

| 模型      | 精度   |
| --------- | ------ |
| resnet18  | 90.34% |
| resnet34  | 93.99% |
| resnet50  | 93.47% |
| resnet101 | 94.24% |
| resnet152 | 93.87% |

基于 resnet34，调整学习率 lr=2e-3 ，精度提升到 94.35%。

继续增加学习率到 lr=5e-3，效果不佳。增加 scheduler = CosineAnnealingLR ，学习速度加快，精度略微提升到 94.48%。
