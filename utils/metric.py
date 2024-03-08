import torch


def evaluate_loss(net, valid_dataloader, loss, device):
    l_sum, n = 0.0, 0
    for x, y in valid_dataloader:
        x, y = x.to(device), y.to(device)
        l = loss(net(x), y)
        l_sum += l.item() * y.numel()
        n += y.numel()
    return l_sum / n


def accuracy(net, valid_dataloader, device):
    sample_sum, n = 0.0, 0.0
    for x, y in valid_dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        y_hat_softmax = torch.nn.functional.softmax(y_hat, dim=1)
        class_indices = y_hat_softmax.argmax(dim=1)
        accuracy_mat = (class_indices == y).float()
        accuracy_batch = accuracy_mat.sum()
        sample_sum += accuracy_batch
        n += y.numel()
    return sample_sum / n
