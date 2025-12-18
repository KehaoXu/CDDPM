import os
import torch
import pandas as pd

from medmnist.evaluator import Evaluator, getAUC, getACC, Metrics

# evaluation
def evaluate_classifier(model, data_loader, split, save_folder=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)

    # data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        # print(y_score.shape[0])

        evaluator = MyEvaluator(labels = y_true, split = split)
        metrics = evaluator.evaluate(y_score, save_folder = save_folder)

        # print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
        return metrics

class MyEvaluator(Evaluator):
    def __init__(self, labels, split, task="binary-class"):
        self.split = split
        self.labels = labels
        self.info = {"task": task}

    def evaluate(self, y_score, save_folder=None):
        assert y_score.shape[0] == self.labels.shape[0]
        auc = getAUC(self.labels, y_score, self.info["task"])
        acc = getACC(self.labels, y_score, self.info["task"])
        metrics = Metrics(auc, acc)

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            path = os.path.join(
                save_folder, self.get_standard_evaluation_filename(metrics)
            )
            pd.DataFrame(y_score).to_csv(path, header=None)
        return metrics

    def get_standard_evaluation_filename(self, metrics):
        eval_txt = "_".join([f"[{k}]{v:.3f}" for k, v in zip(metrics._fields, metrics)])

        ret = f"{self.split}_{eval_txt}.csv"
        return ret