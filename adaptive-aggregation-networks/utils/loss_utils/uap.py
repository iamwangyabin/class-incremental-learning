import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


'''
Basic version of untargeted stochastic gradient descent UAP adapted from:
[AAAI 2020] Universal Adversarial Training
- https://ojs.aaai.org//index.php/AAAI/article/view/6017
Layer maximization attack from:
Universal Adversarial Perturbations to Understand Robustness of Texture vs. Shape-biased Training
- https://arxiv.org/abs/1911.10364
'''


def uap_sgd(model, loader, nb_epoch, eps, beta=12, step_decay=0.8, y_target=None, loss_fn=None, layer_name=None,
            uap_init=None, device=None):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    y_target    target class label for Targeted UAP variation
    loss_fn     custom loss function (default is CrossEntropyLoss)
    layer_name  target layer name for layer maximization attack
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})

    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    _, (_, x_val, y_val) = next(enumerate(loader))
    if uap_init is None:
        delta = torch.zeros(x_val.shape[1:], requires_grad=True).unsqueeze(0)  # initialize as zero vector
    else:
        delta = uap_init

    losses = []


    # loss function
    if layer_name is None:
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta]).to(device)

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss

    # layer maximization attack
    else:
        def get_norm(self, forward_input, forward_output):
            global main_value
            main_value = torch.norm(forward_output, p='fro')

        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_norm)

    delta.requires_grad_()

    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay

        for i, (_, x_val, y_val) in enumerate(loader):

            # for targeted UAP, switch output labels to y_target
            if y_target is not None: y_val = torch.ones(size=y_val.shape, dtype=y_val.dtype) * y_target

            perturbed = torch.clamp((x_val + delta).to(device), 0, 1)
            outputs = model(perturbed)['logits']


            # loss function value
            if layer_name is None:
                loss = clamped_loss(outputs, y_val.to(device))
            else:
                loss = main_value

            if y_target is not None: loss = -loss  # minimize loss for targeted UAP
            losses.append(torch.mean(loss))

            grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
            delta = Variable(delta.data + eps_step * torch.sign(grad), requires_grad=True)
            delta = Variable(torch.clamp(delta.data, -eps, eps), requires_grad=True)

            delta = torch.clamp(delta, -eps, eps)

    if layer_name is not None: handle.remove()  # release hook

    return delta.data, losses