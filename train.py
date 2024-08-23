from model.net import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.dataset import trainset, testset
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 定义device
device = torch.device(f"cuda:{3}" if torch.cuda.is_available() else "cpu")


# def train(mymodel, data_loader, mycriterion, myoptimizer):
#     mymodel.train()
#     running_loss = 0.0
#     for images, labels in tqdm(data_loader, desc="Training", leave=False):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = mycriterion(outputs, labels)

#         myoptimizer.zero_grad()
#         loss.backward()
#         myoptimizer.step()

#         running_loss += loss.item()

#     avg_training_loss = running_loss / len(train_loader)

#     return avg_training_loss


def train_evaluate(mymodel, train_loader):
    mymodel.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            train_loader, desc="Evaluating Training Set", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = mymodel(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    return train_accuracy


def test(mymodel, data_loader):
    mymodel.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = mymodel(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler=None,
    save_dir="runs/",
    epochs_num=100,
):
    writer = SummaryWriter(save_dir)

    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    best_accuracy = 0.0
    best_model = None

    for epoch in range(epochs_num):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_training_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Training", avg_training_loss, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{epochs_num}], Training Loss: {avg_training_loss:.4f}"
        )

        train_accuracy = train_evaluate(model, train_loader)
        test_accuracy = test(model, test_loader)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)
        writer.add_scalar("Accuracy/Test", test_accuracy, epoch + 1)

        print(
            f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}\n"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model

        if scheduler is not None:
            scheduler.step()

    writer.close()

    return model, best_model


if __name__ == "__main__":
    model = resnet34().to(device)

    # 统计模型参数量
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable_num}")

    # 加载数据
    batch_size = 128  # batch size
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 超参数
    criterion = nn.CrossEntropyLoss()
    num_epochs = 500  # epochs_num
    learning_rate = 5e-3  # lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    save_dir = "runs/train2/"

    # 训练模型
    model, best_model = train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        epochs_num=num_epochs,
    )

    # 保存最佳模型
    best_model_path = os.path.join(save_dir, "best.pth")
    torch.save(best_model.state_dict(), best_model_path)

    last_model_path = os.path.join(save_dir, "last.pth")
    torch.save(model.state_dict(), last_model_path)
