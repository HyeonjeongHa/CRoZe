import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
    
    
def adjust_learning_rate(args, optimizer, epoch, learning_rate, epochs=None, dataset=None, lr_scheduler=None):
    
    if dataset == 'ImageNet16-120':
        if args.lr_scheduler == 'cosine':
            assert lr_scheduler is not None
            lr_scheduler.step()
        elif args.lr_scheduler == 'linear':
            if epochs-epoch > 5:
                lr = learning_rate * (epochs - 5 - epoch) / (epochs - 5)
            else:
                lr = learning_rate * (epochs - epoch) / ((epochs - 5) * 5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            NotImplementedError()
            
        if epoch < 5 and args.batch_size > 256:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate * (epoch + 1) / 5.0
    else:
        lr = learning_rate
        if epoch >= 99:
            lr = learning_rate * 0.1
        if epoch >= 149:
            lr = learning_rate * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr