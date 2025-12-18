import os
import pandas as pd
import matplotlib.pyplot as plt

def append_log(epoch, train_acc, test_acc, avg_loss, log_path):
    header = not os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df = pd.DataFrame([{
        "epoch": epoch,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "avg_loss": avg_loss
    }])
    df.to_csv(log_path, mode='a', header=header, index=False)

def plot_classifier_log(log_path):
    log_path1=os.path.join(log_path, "aug_train_log.csv")
    log_path2=os.path.join(log_path, "raw_train_log.csv")

    if not (os.path.exists(log_path1) or os.path.exists(log_path2)):
        print(f"Log file not found: {log_path1} or {log_path2}")
        return

    df1 = pd.read_csv(log_path1)
    df2 = pd.read_csv(log_path2)
    os.makedirs(log_path, exist_ok=True)

    plt.figure()
    plt.plot(df1["epoch"], df1["avg_loss"], 'r-', label="aug")
    plt.plot(df2["epoch"], df2["avg_loss"], 'g-', label="raw")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    loss_path = os.path.join(log_path, "loss_curve.png")
    plt.savefig(loss_path)
    # plt.close()

    plt.figure()
    plt.plot(df1["epoch"], df1["train_acc"], 'r--', label="aug train")
    plt.plot(df2["epoch"], df2["train_acc"], 'g--', label="raw train")
    plt.plot(df1["epoch"], df1["test_acc"], 'r-', label="aug test")
    plt.plot(df2["epoch"], df2["test_acc"], 'g-', label="raw test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train & Test Accuracy")
    plt.legend()
    acc_path = os.path.join(log_path, "acc_curve.png")
    plt.savefig(acc_path)
    # plt.close()

    # print(f"Saved plot to {log_path}")
 