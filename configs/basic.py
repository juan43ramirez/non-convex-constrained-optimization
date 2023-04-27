import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def make_train_config():
    _config = mlc.ConfigDict()

    _config.seed = MLC_PH(int)

    _config.train_batch_size = MLC_PH(int)
    _config.val_batch_size = MLC_PH(int)
    _config.epochs = MLC_PH(int)

    _config.optimizer = MLC_PH(str)
    _config.lr = MLC_PH(float)
    _config.optimizer_kwargs = MLC_PH(dict)

    _config.target_model = MLC_PH(float)
    _config.target_layer = MLC_PH(float)

    return _config


def make_model_config():
    _config = mlc.ConfigDict()

    _config.name = MLC_PH(str)
    _config.kwargs = MLC_PH(dict)

    return _config


def make_formulation_config():
    _config = mlc.ConfigDict()

    _config.name = MLC_PH(str)
    _config.kwargs = mlc.ConfigDict()
    _config.multiplier_init = MLC_PH(float)

    return _config


def build_basic_config():
    config = mlc.ConfigDict()

    # Populate top-level configs which are common to all experiments
    config.train = make_train_config()
    config.model = make_model_config()
    config.formulation = make_formulation_config()

    # Fixed defaults for logging across all experiments
    config.logging = mlc.ConfigDict()
    config.logging.log_level = "INFO"
    config.logging.wandb_mode = "online"

    return config
