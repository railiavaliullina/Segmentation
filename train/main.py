import torch
import time
import numpy as np
from tensorboardX import SummaryWriter

from utils.log_utils import start_logging, end_logging, log_metrics, log_params
from utils.train_utils import get_criterion, get_optimizer, make_training_step
from utils.debug_utils import overfit_on_batch
from utils.data_utils import get_dataloader
from utils.eval_utils import evaluate
from models.unet_small import get_model
from configs.config import cfg


def train(model, criterion, opt, train_dl, valid_dl, writer):

    # restore model if necessary
    global_step, start_epoch = 0, 0
    if cfg.load_model:
        print(f'Trying to load checkpoint from epoch {cfg.epoch_to_load}...')
        checkpoint = torch.load(cfg.checkpoints_dir + f'checkpoint_{cfg.epoch_to_load}.pth')
        load_state_dict = checkpoint['model']
        model.load_state_dict(load_state_dict)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
        print(f'Successfully loaded checkpoint from epoch {cfg.epoch_to_load}.')

    # evaluate on train and test data before training
    if cfg.evaluate_before_training:
        model.eval()
        with torch.no_grad():
            evaluate(cfg, model, valid_dl, start_epoch - 1, 'valid', writer=writer)
            if cfg.evaluate_on_train_data:
                evaluate(cfg, model, train_dl, start_epoch - 1, 'train')
        model.train()

    # training loop
    nb_iters_per_epoch = len(train_dl.dataset) // cfg.batch_size
    for epoch in range(start_epoch, cfg.epochs):
        losses = []
        epoch_start_time = time.time()
        print(f'Epoch: {epoch}')
        for iter_, batch in enumerate(train_dl):
            loss = make_training_step(batch, model, criterion, opt)
            losses.append(loss)
            global_step += 1

            log_metrics(['train/loss'], [loss], global_step, cfg)

            if global_step % 100 == 0:
                mean_loss = np.mean(losses[:-20]) if len(losses) > 20 else np.mean(losses)
                print(f'step: {global_step}, loss: {mean_loss}')

        # log mean loss per epoch
        log_metrics(['train/mean_loss'], [np.mean(losses[:-nb_iters_per_epoch])], epoch, cfg)
        print(f'Epoch time: {round((time.time() - epoch_start_time) / 60, 3)} min')

        # save model
        if cfg.save_model and epoch % cfg.save_frequency == 0:
            print('Saving current model...')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'opt': opt.state_dict(),
            }
            torch.save(state, cfg.checkpoints_dir + f'checkpoint_{epoch}.pth')

        # evaluate on train and test data
        model.eval()
        with torch.no_grad():
            if cfg.evaluate_on_train_data:
                evaluate(cfg, model, train_dl, epoch, 'train')
            evaluate(cfg, model, valid_dl, epoch, 'valid')
        model.train()


if __name__ == '__main__':
    train_dl = get_dataloader('train')
    valid_dl = get_dataloader('valid')

    model = get_model(cfg)
    criterion = get_criterion()
    opt = get_optimizer(cfg, model)

    writer = SummaryWriter(cfg.tensorboard_dir) if cfg.use_tensorboard else None

    # check training procedure before training
    if cfg.overfit_on_batch:
        overfit_on_batch(model, opt, criterion, train_dl, writer=writer)

    # save experiment name and experiment params to mlflow
    start_logging(cfg)
    log_params(cfg)

    # train model
    start_time = time.time()
    train(model, criterion, opt, train_dl, valid_dl, writer)
    print(f'Total time: {round((time.time() - start_time) / 60, 3)} min')

    # end logging
    end_logging(cfg)
