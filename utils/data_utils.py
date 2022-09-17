from torch.utils.data import DataLoader

from data.mnist_cifar_dataset import MNISTCIFARSegmentation
from configs.config import cfg


def get_dataloader(dataset_type):
    print(f'Getting {dataset_type} dataloader...')
    dataset = MNISTCIFARSegmentation(cfg, dataset_type)
    dl = DataLoader(dataset, batch_size=cfg.batch_size)
    return dl
