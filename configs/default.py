"""
python main.py --config=configs/default.py:logreg-sgd-qp --config.logging.wandb_mode=disabled --config.train.epochs=1
"""

from configs.basic import build_basic_config
from configs.formulation import alm_config, exact_config, lag_config, qp_config
from configs.model import logreg_config, mlp_config
from configs.train import adam_config, sgd_config


def get_config(config_string):
    model, optimizer, formulation = config_string.split("-")

    config = build_basic_config()
    config.model = globals()[model + "_config"]()
    config.train = globals()[optimizer + "_config"]()
    config.formulation = globals()[formulation + "_config"]()

    return config
