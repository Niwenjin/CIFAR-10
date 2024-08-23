import matplotlib.pyplot as plt


def print_figure(training_curve_path, num_epochs, train_losses, test_losses, test_accuracies):
    # 绘制训练损失和验证准确率曲线
    plt.figure(figsize=(12, 4))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证准确率曲线
    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(training_curve_path)
