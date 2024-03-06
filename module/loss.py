from torch.nn import CrossEntropyLoss
from torch.nn import Module


class MyLoss(Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_function = CrossEntropyLoss()

    def forward(self, y_pre, y_true):
        y_true = y_true.long()
        loss = self.loss_function(y_pre, y_true)

        return loss
