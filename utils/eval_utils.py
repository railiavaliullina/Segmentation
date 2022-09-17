from torch.nn import functional as F
import torch
import numpy as np
import time

from utils.log_utils import log_metrics
from configs.config import cfg


def evaluate(cfg, model, dl, epoch, dataset_type, writer=None):
    """
    Evaluates on train/valid data
    :param cfg: config
    :param model: unet model
    :param dl: train/valid dataloader
    :param epoch: epoch for logging
    :param dataset_type: type of current data ('train' or 'valid')
    """
    print(f'Evaluating on {dataset_type} data...')
    eval_start_time = time.time()
    accuracies, mIoUs = [], []

    dl_len = len(dl)
    for i, batch in enumerate(dl):
        (images, image_unpadded), targets = batch
        images, targets = images.cuda(), targets.cuda()
        if i % 50 == 0:
            print(f'iter: {i}/{dl_len}')

        out = model(images)
        prediction = torch.argmax(F.softmax(out), dim=1)
        acc, mIoU = calculate_metrics(prediction.cpu().numpy(), targets.cpu().numpy())
        accuracies.append(acc)
        mIoUs.append(mIoU)

        if i < 3 and dataset_type is 'valid' and cfg.save_results_to_tensorboard and writer is not None:
            print(f'Saving results to Tensorboard...')
            writer.add_image(f'final_result_{i}/image', image_unpadded[0].permute(2, 0, 1))
            writer.add_image(f'final_result_{i}/target', targets[0].unsqueeze(0).cpu())
            writer.add_image(f'final_result_{i}/prediction', np.round(out[0][1].cpu().detach().unsqueeze(0).numpy()))

            # writer.add_image(f'intermediate result_{i}/image', image_unpadded[0].permute(2, 0, 1))
            # writer.add_image(f'intermediate result_{i}/target', targets[0].unsqueeze(0).cpu())
            # writer.add_image(f'intermediate result_{i}/prediction', out[0][1].cpu().detach().unsqueeze(0).numpy())
        if i == 3 and cfg.terminate_after_saving_results:
            quit()

    global_accuracy = np.mean(accuracies)
    mIoU_ = np.mean(mIoUs)
    log_metrics([f'{dataset_type}_eval/global_accuracy', f'{dataset_type}_eval/mIoU'],
                [global_accuracy, mIoU_], epoch, cfg)

    print(f'Global accuracy on {dataset_type} data: {global_accuracy}\n'
          f'mIoU on {dataset_type} data: {mIoU_}')
    print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')


def calculate_metrics(y_pred, y_true):
    global_accuracy_cur, metrics, final_metrics = [], [], []

    for pred_, gt_ in zip(y_pred, y_true):
        gt_ = gt_.astype('uint8')
        pred_ = (pred_ > 0.5).astype('uint8')
        h, w = gt_.shape
        global_acc_ = [np.sum(pred_ == gt_), float(h * w)]
        global_accuracy_cur.append(global_acc_)

        metrics_ = []
        for i in range(cfg.num_classes):
            metrics_.append([np.sum((pred_ == i) & (gt_ == i)),
                             np.sum((pred_ == i) & (gt_ != i)),
                             np.sum((pred_ != i) & (gt_ == i))])
        metrics.append(metrics_)
    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])

    for i in range(cfg.num_classes):
        final_metrics.append([np.sum([v[i][0] for v in metrics]),
                              np.sum([v[i][1] for v in metrics]),
                              np.sum([v[i][2] for v in metrics])])

    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in final_metrics]) / cfg.num_classes
    return global_acc * 100, mean_iou_acc * 100
