import torch
import pandas as pd
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from dataset.dataset import LeavesDataset
from dataset.transform import transform_train, transform_test
from models.model import Model
from trainer.train import train


def main():
    num_classes, batch_size = 176, 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Model(num_classes, model='ResNet')

    writer = SummaryWriter('logs')
    train_csv = pd.read_csv('E:\\PycharmProjects\\kaggle_project\\data\\classify-leaves\\train.csv')
    test_csv = pd.read_csv('E:\\PycharmProjects\\kaggle_project\\data\\classify-leaves\\test.csv')
    labels = list(train_csv['label'])
    labels_unique = list(set(list(labels)))
    labels_num = []
    for i in range(len(labels)):
        labels_num.append(labels_unique.index(labels[i]))
    train_csv['number'] = labels_num
    imgs_path = 'E:\\PycharmProjects\\TestProject\\data\\classify-leaves'
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    prediction_df = pd.DataFrame()
    for fold_n, (train_idx, val_idx) in enumerate(skf.split(train_csv, train_csv['number'])):
        print(f'fold {fold_n} training...')
        train_data, valid_data = train_csv.iloc[train_idx], train_csv.iloc[val_idx]
        train_set = LeavesDataset(train_data, imgs_path, transform=transform_train)
        valid_set = LeavesDataset(valid_data, imgs_path, transform=transform_test)
        best_model_state = train(net, train_set, valid_set, batch_size=batch_size, num_epochs=30, lr=0.0001, weight_decay=1e-5, device=device, writer=writer)

        torch.save(best_model_state, f'./checkpoints/model_weights_fold_{fold_n}.pth')
        net.load_state_dict(best_model_state)

        predictions = []
        test_set = LeavesDataset(test_csv, imgs_path, test=True, transform=transform_test)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        net.eval()
        with torch.no_grad():
            for x in tqdm(test_dataloader):
                x = x.to(device)
                y_hat = net(x)
                y_hat_softmax = nn.functional.softmax(y_hat, dim=1)
                predict = y_hat_softmax.argmax(dim=1).reshape(-1)
                predict = list(predict.cpu().detach().numpy())
                predictions.extend(predict)
        prediction_df[f'fold_{fold_n}'] = predictions

    all_predictions = list(prediction_df.mode(axis=1)[0].astype(int))
    predict_label = []
    for i in range(len(all_predictions)):
        predict_label.append(labels_unique[all_predictions[i]])

    submission = test_csv
    submission['label'] = pd.Series(predict_label)
    submission.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    main()
