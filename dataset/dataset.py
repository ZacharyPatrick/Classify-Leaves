import os
import cv2
from torch.utils.data import Dataset


class LeavesDataset(Dataset):
    """
    file_path: 训练集/测试集的csv文件路径
    imgs_path: 训练集和测试集的父级目录
    test: 数据集是否是测试集
    transform: image的预处理
    target_transform: label的预处理
    """

    def __init__(self, dataset_df, imgs_path, test=False, transform=None, target_transform=None):
        self.test = test
        self.dataset_df = dataset_df
        self.imgs_path = imgs_path
        if not test:
            self.labels = list(self.dataset_df['number'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        imgs_abs_path = os.path.join(self.imgs_path, self.dataset_df.iloc[idx, 0])
        img = cv2.imread(imgs_abs_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']

        if not self.test:
            if self.target_transform:
                label = self.labels[idx]
                label = self.target_transform(label)
                return img, label
            else:
                label = self.labels[idx]
                return img, label
        else:
            return img
