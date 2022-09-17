import torch

from losses.BCELoss import BCELoss


def get_criterion():
    """
    Gets loss function
    :return: BCELoss function
    """
    criterion = BCELoss()
    return criterion


def get_optimizer(cfg, model):
    """
    Gets optimizer for parameters update
    :param cfg: config with all parameters needed for training
    :param model: Unet model
    :return: optimizer
    """
    opt = torch.optim.SGD(model.parameters(), cfg.lr, cfg.momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    # torch.optim.Adam([{'params': model.parameters(), 'lr': cfg.lr}])
    return opt


def make_training_step(batch, model, criterion, opt):
    """
    Makes single parameters updating step.
    :param batch: current batch
    :param model: Unet model
    :param criterion: criterion
    :param optimizer: optimizer
    :param iter_: current iteration
    :return: current loss value
    """
    (images, _), targets = batch
    images, targets = images.cuda(), targets.cuda()
    opt.zero_grad()
    out = model(images)
    loss = criterion(out, targets.float())
    loss.backward()
    opt.step()
    return loss.item()
