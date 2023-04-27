import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def lag_config():
    _config = mlc.ConfigDict()

    _config.name = "LagrangianFormulation"

    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.dual_lr = 0.1

    _config.multiplier_init = 0.0

    return _config


def qp_config():
    _config = mlc.ConfigDict()

    _config.name = "QuadraticPenaltyFormulation"

    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.penalty_coefficient = 1.0
    _config.kwargs.penalty_gamma = 2.0

    _config.multiplier_init = 0.0

    return _config


def exact_config():
    _config = mlc.ConfigDict()

    _config.name = "ExactPenaltyFormulation"

    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.penalty_coefficient = 1.0
    _config.kwargs.penalty_gamma = 2.0

    _config.multiplier_init = 0.0

    return _config


def alm_config():
    _config = mlc.ConfigDict()

    _config.name = "AugmentedLagrangianFormulation"

    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.penalty_coefficient = 0.1
    _config.kwargs.penalty_gamma = 2.0

    _config.multiplier_init = 0.0

    return _config
