# CIFAR-10

使用 Resnet 模型在 CIFAR-10 上训练高精度模型。

使用基础模型, Adam, batch_size=128, epochs=500, lr=1e-3 初步验证模型性能。

| 模型      | 参数量 | 训练耗时 | 训练精度 | 测试精度 |
| --------- | ------ | -------- | -------- | -------- |
| resnet18  | 11.17M | 3h 8m    | 99.61%   | 80.34%   |
| resnet34  | 21.28M | 8h 53m   | 99.98%   | 83.99%   |
| resnet50  | 23.52M | 18h 42m  | 99.95%   | 83.47%   |
| resnet101 | 42.51M | 28h 5m   | 99.99%   | 84.24%   |
| resnet152 | 58.15M | 38h 55m  | 99.81%   | 83.87%   |

基于 resnet34，增加数据增强 ，增加 transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()，精度提升到 94.35%。

继续增加学习率到 lr=5e-3，效果不佳。增加 scheduler = CosineAnnealingLR ，学习速度加快，精度略微提升到 94.48%。

模型在训练集上的准确率已经达到 1，尝试对数据集进行数据增强。增加 transforms.ColorJitter，精度略微提升到 94.95%。

使用 SGD 优化器，参数 optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)，拟合能力优秀，泛化效果不佳。

增加 cutmix 和 mixup，继续使用 SGD 优化器，精度达到 96.47%。
