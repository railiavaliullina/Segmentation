from easydict import EasyDict

cfg = EasyDict()
cfg.datasets = EasyDict()
cfg.datasets.train = EasyDict()
cfg.datasets.valid = EasyDict()

# data params
cfg.num_classes = 2
cfg.in_channels = 3
cfg.pad = (20, 20)
cfg.mean = [0.485, 0.456, 0.406]  # [0.471, 0.448, 0.408]
cfg.std = [0.229, 0.224, 0.225]  # [0.234, 0.239, 0.242]
cfg.weights_path = 'D:/datasets/homeworks/cv segmentation(synthetic_dataset)/class_weights.pickle'
cfg.datasets.train.root = 'D:/datasets/homeworks/cv segmentation(synthetic_dataset)/cifar-10-batches-py/'
cfg.datasets.train.annFile = 'D:/datasets/homeworks/cv segmentation(synthetic_dataset)/MNIST/raw/'
cfg.datasets.valid.root = 'D:/datasets/homeworks/cv segmentation(synthetic_dataset)/cifar-10-batches-py/'
cfg.datasets.valid.annFile = 'D:/datasets/homeworks/cv segmentation(synthetic_dataset)/MNIST/raw/'
cfg.images_path = 'D:/datasets/homeworks/cv segmentation(synthetic_dataset)/saved_images/'

# training params
cfg.lr = 0.001
cfg.momentum = 0.99
cfg.weight_decay = 1e-4
cfg.nesterov = True
cfg.batch_size = 1
cfg.epochs = 100

# evaluation params
cfg.evaluate_before_training = True
cfg.evaluate_on_train_data = True
cfg.save_results_to_tensorboard = False
cfg.terminate_after_saving_results = False

# overfit on batch params
cfg.overfit_on_batch = False
cfg.overfit_on_batch_iters = 10000

# mlflow params
cfg.log_metrics = True
cfg.experiment_name = 'segmentation (synthetic data)'

# tensorboard
cfg.use_tensorboard = True
cfg.tensorboard_dir = 'tensorboard/'

# model store/restore params
cfg.load_model = True
cfg.epoch_to_load = 3
cfg.checkpoints_dir = 'checkpoints/'
cfg.save_model = True
cfg.save_frequency = 1
