import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


cifar_trainset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False
)
data = cifar_trainset.data / 255

mean = data.mean(axis=(0, 1, 2))
std = data.std(axis=(0, 1, 2))


# 定义数据转换
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0
        ),  # 随机调整亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


# 加载训练集和测试集
DATASET_PATH = "data"

trainset = torchvision.datasets.CIFAR10(
    root=DATASET_PATH, train=True, download=False, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root=DATASET_PATH, train=False, download=False, transform=transform_test
)

# 将训练集拆分为训练集和验证集
# train_size = int(0.8 * len(trainset))
# val_size = len(trainset) - train_size
# trainset, valset = random_split(trainset, [train_size, val_size])

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
# val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# CIFAR-10 类别
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

if __name__ == "__main__":
    print(f"Mean : {mean}   STD: {std}")
