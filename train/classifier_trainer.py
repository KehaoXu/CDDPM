import os
import torch
from torch import nn, optim
from tqdm import tqdm

from utils.classifier_evaluator import evaluate_classifier
from utils.classifier_logger import append_log

def train_classifier(
    model,
    train_loader,
    num_epochs,
    train_loader_raw,
    test_loader,
    log_path,
    ):

    if len(train_loader) == 8:
        save_path = os.path.join(log_path,"raw_train_log.csv")
        # print("raw")
    elif len(train_loader) == 15:
        save_path = os.path.join(log_path,"aug_train_log.csv")
        # print("aug")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
           
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            avg_loss = epoch_loss / total_samples
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        train_metrics = evaluate_classifier(model, train_loader_raw, 'train')
        test_metrics = evaluate_classifier(model, test_loader, 'test')

        append_log(epoch+1, train_metrics.ACC, test_metrics.ACC, avg_loss, save_path)
        print(f"Train ACC: {train_metrics.ACC:.3f}, Test ACC: {test_metrics.ACC:.3f}")