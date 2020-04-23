import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.functional import _Reduction 
from maskrcnn_benchmark import _C

# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    # class_range = torch.ones(num_classes, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    # print(f"t:{t}\nclass_range:{class_range}\nrs:{torch.sum(t == class_range)}")
    # print(logits)
    p = torch.sigmoid(logits)
    # print(p)
    term1 = (1 - p) ** gamma * torch.log(p)
    # print(term1)
    term2 = p ** gamma * torch.log(1 - p)
    return (-(t == 1).float() * term1 * alpha - ((t != 1) * (t >= 0)).float() * term2 * (1 - alpha))/(logits.shape[0]*num_classes)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # logits = torch.sigmoid(predict)
        device = logits.device
        # if logits.is_cuda:
            # loss_func = sigmoid_focal_loss_cuda
        # else:
        loss_func = sigmoid_focal_loss_cpu
        # reduction  = _Reduction.get 

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
