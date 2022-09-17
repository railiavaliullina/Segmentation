from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from utils.eval_utils import calculate_metrics
from configs.config import cfg


def overfit_on_batch(model, opt, criterion, dl, writer=None):
    dl = iter(dl)
    batch = next(dl)
    (image, image_unpadded), target = batch
    image, target = image.cuda(), target.cuda()

    im = image[0].cpu().numpy().transpose(1, 2, 0)
    plt.imshow(im)
    plt.show()
    plt.imshow(target[0].cpu())
    plt.show()

    best_mIoU, best_global_acc = [], []
    for i in range(cfg.overfit_on_batch_iters):
        opt.zero_grad()
        out = model(image)
        loss = criterion(out, target.float())
        prediction = torch.argmax(F.softmax(out), dim=1)
        acc, mIoU = calculate_metrics(prediction.cpu().numpy(), target.cpu().numpy())

        print(f'iter: {i}, loss: {loss}, acc: {acc}, mIoU: {mIoU}')
        if mIoU == 100:
            best_mIoU.append(mIoU)
        if acc == 100:
            best_global_acc.append(acc)
        if len(best_mIoU) >= 5 and len(best_global_acc) >= 5:
            plt.imshow(np.round(out[0][1].cpu().detach().numpy()))
            plt.show()

            if writer is not None:
                print(f'Saving overfit on batch results to Tensorboard...')
                writer.add_image('overfit_on_batch_result/image', image_unpadded[0].permute(2, 0, 1))
                writer.add_image('overfit_on_batch_result/target', target[0].unsqueeze(0).cpu())
                writer.add_image('overfit_on_batch_result/prediction', np.round(out[0][1].cpu().detach().unsqueeze(0).numpy()))
            break
        loss.backward()
        opt.step()
