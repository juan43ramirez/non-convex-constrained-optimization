import torch

import wandb
from src.formulations import AugmentedLagrangianFormulation, KthOrderLagrangianFormulation, KthOrderPenaltyFormulation

from .utils import AverageMeter, accuracy_func, calculate_constraints, ensure_sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def training_loop(model, train_loader, loss_func, target_model, target_layer, optimizer, formulation, multiplier, step):
    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    lagrangian_meter = AverageMeter()

    for batch, labels in train_loader:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        # Measure loss and accuracy
        logits = model(batch)
        loss = loss_func(logits, labels)
        accuracy = accuracy_func(logits, labels)

        # Measure constraint violation
        constraint_violation = calculate_constraints(model, target_model, target_layer)

        # Calculate Lagrangian/Penalty
        lagrangian = formulation.compute_lagrangian(
            loss=loss, ineq_constraints=constraint_violation, ineq_multipliers=multiplier
        )

        optimizer.zero_grad()
        lagrangian.backward()

        # Update primal variables
        optimizer.step()

        if formulation.update_multipliers_on_step:
            formulation.update_multipliers_(ineq_multipliers=multiplier, ineq_constraints=constraint_violation)

        # Log stuff
        loss_meter.update(loss.detach(), n=batch.shape[0])
        accuracy_meter.update(accuracy.detach(), n=batch.shape[0])
        lagrangian_meter.update(lagrangian.detach(), n=batch.shape[0])

        to_log = {"constraint_" + str(i): val.detach() for i, val in enumerate(ensure_sequence(constraint_violation))}

        if isinstance(formulation, (KthOrderLagrangianFormulation, AugmentedLagrangianFormulation)):
            to_log.update({"multiplier_" + str(i): val.detach() for i, val in enumerate(ensure_sequence(multiplier()))})
        to_log["lagrangian"] = lagrangian.detach()
        to_log["loss"] = loss.detach()
        to_log["accuracy"] = accuracy
        to_log = {f"batch/train/{k}": v for k, v in to_log.items()}

        wandb.log(to_log, step=step)

        step += 1

    epoch_train_log = {"loss": loss_meter.avg, "accuracy": accuracy_meter.avg, "lagrangian": lagrangian_meter.avg}
    epoch_train_log.update({f"constraint_{i}": val for i, val in enumerate(ensure_sequence(constraint_violation))})

    if isinstance(formulation, (KthOrderLagrangianFormulation, AugmentedLagrangianFormulation)):
        epoch_train_log.update({f"multiplier_{i}": val for i, val in enumerate(ensure_sequence(multiplier()))})

    if not isinstance(formulation, KthOrderLagrangianFormulation):
        epoch_train_log["penalty"] = formulation.penalty_coefficient

    epoch_train_log = {f"epoch/train/{k}": v for k, v in epoch_train_log.items()}

    return step, epoch_train_log


def qualifications(model, train_loader, loss_func, target_model, target_layer, multiplier):
    """Measure various metrics to assess convergence to a local minimizer"""

    # TODO: this could be extended to stochastic constraints
    constraint_violation = calculate_constraints(model, target_model, target_layer)
    is_feasible = torch.all(constraint_violation <= 0)
    complimentarity = (constraint_violation * multiplier()).sum()

    constraint_jacobian = []
    for constraint in ensure_sequence(constraint_violation):
        for param in model.parameters():
            param.grad = None

        # Setting retain_graph=True is necessary as different constraints refer to the
        # same parameters.
        constraint.backward(retain_graph=True)
        constraint_jacobian.append(torch.cat([param.grad.flatten() for param in model.parameters()]))

    constraint_jacobian = torch.stack(constraint_jacobian, dim=1)
    smallest_sv = torch.linalg.svdvals(constraint_jacobian).min()

    model.eval()
    loss_grad_meter = AverageMeter()
    for batch, labels in train_loader:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        for param in model.parameters():
            param.grad = None

        loss = loss_func(model(batch), labels).mean()
        loss.backward()

        param_grads = torch.cat([param.grad.flatten() for param in model.parameters()])
        loss_grad_meter.update(param_grads, n=batch.shape[0])

    grad_loss = loss_grad_meter.avg

    with torch.no_grad():
        mult_val = multiplier() if multiplier.shape != torch.Size([]) else multiplier().unsqueeze(0)
        grad_lagrangian = grad_loss + constraint_jacobian @ mult_val
        grad_lagrangian_norm = grad_lagrangian.pow(2).sum()

    # TODO: could check if the Hessian of the Lagrangian is positive definite

    return constraint_violation, is_feasible, complimentarity, grad_lagrangian_norm, smallest_sv


@torch.inference_mode()
def validation_loop(model, val_loader, loss_func, target_model, target_layer, step):
    model.eval()
    loss_meter, accuracy_meter = AverageMeter(), AverageMeter()

    for batch, labels in val_loader:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        # Measure loss and accuracy
        logits = model(batch)
        loss = loss_func(logits, labels)
        accuracy = accuracy_func(logits, labels)

        loss_meter.update(loss.detach(), n=batch.shape[0])
        accuracy_meter.update(accuracy.detach(), n=batch.shape[0])

        # Measure constraint violation
        constraint_violation = calculate_constraints(model, target_model, target_layer)
        to_log = {
            "batch/val/constraint_" + str(i): val.detach()
            for i, val in enumerate(ensure_sequence(constraint_violation))
        }

        to_log["batch/val/loss"] = loss
        to_log["batch/val/accuracy"] = accuracy

        wandb.log(to_log, step=step)

        step += 1

    epoch_val_log = {"epoch/val/loss": loss_meter.avg, "epoch/val/accuracy": accuracy_meter.avg}

    return step, epoch_val_log
