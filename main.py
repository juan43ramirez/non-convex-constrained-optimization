import logging

import torch
from absl import app
from absl.flags import FLAGS
from ml_collections.config_flags import config_flags as MLC_FLAGS

import wandb
from src import datasets, experiment_utils, formulations, models, multipliers, utils

# Initialize "FLAGS.config" object with a placeholder config
MLC_FLAGS.DEFINE_config_file("config", default="configs/default.py")

logging.basicConfig()
logger = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(_):

    config = FLAGS.config
    logger.setLevel(getattr(logging, config.logging.log_level))
    wandb.init(
        project="lagrangian", entity="juan_ramirez_research", config=config.to_dict(), mode=config.logging.wandb_mode
    )

    utils.seed_all(config.train.seed)

    train_loader, val_loader, num_classes, input_shape = datasets.mnist(
        config.train.train_batch_size, config.train.val_batch_size
    )

    model_class = models.__dict__[config.model.name]
    model = model_class(input_shape, num_classes, **config.model.kwargs).to(DEVICE)

    loss_func = torch.nn.CrossEntropyLoss()
    target_model, target_layer = config.train.target_model, config.train.target_layer
    optimizer_kwargs = config.train.optimizer_kwargs or {}
    optimizer = torch.optim.__dict__[config.train.optimizer](model.parameters(), lr=config.train.lr, **optimizer_kwargs)

    formulation_kwargs = config.formulation.kwargs or {}
    formulation = formulations.__dict__[config.formulation.name](**formulation_kwargs)

    num_layers = model.num_layers
    # Adding one for the model-wise constraint
    multiplier_init = torch.tensor(config.formulation.multiplier_init, device=DEVICE).repeat(num_layers + 1)
    multiplier = multipliers.Multiplier(init=multiplier_init, enforce_positive=True)

    step = 0

    logger.info("Validation loop at initialization")
    step, epoch_val_log = experiment_utils.validation_loop(
        model, val_loader, loss_func, target_model, target_layer, step
    )
    wandb.log({**epoch_val_log, "_epoch": 0}, step=step)

    for epoch in range(config.train.epochs):
        logger.info(f"Epoch {epoch}/{config.train.epochs}")

        logger.info("Training loop started")
        step, epoch_train_log = experiment_utils.training_loop(
            model, train_loader, loss_func, target_model, target_layer, optimizer, formulation, multiplier, step
        )

        logger.info("Qualifications loop started")
        # Feasibility, constraint qualification and first order optimality conditions
        (violations, is_feasible, complimentarity, grad_lag_norm, smallest_sv) = experiment_utils.qualifications(
            model, train_loader, loss_func, target_model, target_layer, multiplier
        )

        # Update multipliers
        formulation.update_state_(ineq_multipliers=multiplier, ineq_violations=violations)

        logger.info("Validation loop started")
        step, epoch_val_log = experiment_utils.validation_loop(
            model, val_loader, loss_func, target_model, target_layer, step
        )

        if epoch % 10 == 0:
            logger.info("\n")
            logger.info("Loss: {:.4f}".format(epoch_train_log["epoch/train/loss"]))
            logger.info("Train Accuracy: {:.4f}".format(epoch_train_log["epoch/train/accuracy"]))
            logger.info("Model-wise constraint: {:.4f}".format(violations[0]))

        epoch_log = {
            "epoch/train/is_feasible": is_feasible.float(),
            "epoch/train/complimentary_slackness": complimentarity,
            "epoch/train/grad_lagrangian_norm": grad_lag_norm,
            "epoch/train/constraint_jacobian_smallest_sv": smallest_sv,
        }
        wandb.log({**epoch_log, **epoch_train_log, **epoch_val_log, "_epoch": epoch}, step=step)

    # Save final model
    if config.logging.wandb_mode == "online":
        torch.save(model.state_dict(), wandb.run.dir + "model.pt")


if __name__ == "__main__":
    app.run(main)
