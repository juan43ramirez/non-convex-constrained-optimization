import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def logreg_config():
    _config = mlc.ConfigDict()

    _config.name = "LogisticRegression"
    _config.kwargs = {}

    return _config


def mlp_config():
    _config = mlc.ConfigDict()

    _config.name = "MLP"
    _config.kwargs = {"hidden_dims": [128]}

    return _config
