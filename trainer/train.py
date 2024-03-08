import copy
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from utils.metric import evaluate_loss, accuracy


def train(net, train_data, valid_data, batch_size, num_epochs, lr, weight_decay, device, writer):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)
    net = net.to(device)
    writer.add_graph(net, train_data[0][0].reshape(1, 3, 320, 320).to(device))
    best_epoch, best_score, best_model_state, early_stopping_round = 0, 0, None, 3
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc = 0.0, 0
        for x, y in tqdm(train_dataloader):
            x, y = x.to(device), y.to(device)
            l = loss(net(x), y)
            train_loss += l.item() * y.numel()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            y_hat_softmax = torch.nn.functional.softmax(net(x), dim=1)
            train_acc += torch.sum(y_hat_softmax.argmax(dim=1) == y)
        scheduler.step()
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        print("epoch: ", epoch, "loss: ", train_loss / len(train_data), "训练集准确度: ", (train_acc / len(train_data)), end=" ")
        writer.add_scalar('train loss', train_loss / len(train_data), epoch)
        writer.add_scalar('train accuracy', train_acc / len(train_data), epoch)
        if valid_dataloader is not None:
            net.eval()
            with torch.no_grad():
                valid_loss = evaluate_loss(net, valid_dataloader, loss, device)
                valid_accuracy = accuracy(net, valid_dataloader, device)
            print('验证集准确度: ', valid_accuracy)
            writer.add_scalar('valid loss', valid_loss, epoch)
            writer.add_scalar('valid accuracy', valid_accuracy, epoch)
        if valid_accuracy * len(valid_data) > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = valid_accuracy * len(valid_data)
            best_epoch = epoch
            print('best epoch save!')
        if epoch - best_epoch >= early_stopping_round:
            break
    return best_model_state
