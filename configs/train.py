import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def sgd_config():
    _config = mlc.ConfigDict()

    _config.seed = 0

    _config.train_batch_size = 128
    _config.val_batch_size = 128
    _config.epochs = 100

    _config.optimizer = "SGD"
    _config.lr = 1e-2
    _config.optimizer_kwargs = {"momentum": 0.9}

    _config.target_model = 100.0
    _config.target_layer = 100.0

    return _config


def adam_config():
    _config = mlc.ConfigDict()

    _config.seed = 0

    _config.train_batch_size = 128
    _config.val_batch_size = 128
    _config.epochs = 100

    _config.optimizer = "Adam"
    _config.lr = 1e-4
    _config.optimizer_kwargs = {}

    _config.target_model = 100.0
    _config.target_layer = 100.0

    return _config
