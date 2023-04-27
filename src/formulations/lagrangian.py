from typing import Optional

import torch

from .formulation import Formulation


class KthOrderLagrangianFormulation(Formulation):
    """Kth-order Lagrangian formulation. f + sum_i lam_i * |g_i|^k.

    This formulation is equivalent to a Lagrangian for order=1."""

    def __init__(self, order: int, dual_lr: float, *args, **kwargs):
        assert order == 1
        # TODO(juan43ramirez): which order values are principled?
        self.order = order
        self.dual_lr = dual_lr

        super().__init__(*args, **kwargs)

    def compute_lagrangian(
        self,
        loss: torch.Tensor,
        ineq_constraints: Optional[tuple[torch.Tensor]] = None,
        eq_constraints: Optional[tuple[torch.Tensor]] = None,
        ineq_multipliers: Optional[tuple[torch.Tensor]] = None,
        eq_multipliers: Optional[tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        lagrangian = torch.clone(loss)

        if ineq_constraints is not None:
            assert ineq_multipliers.shape == ineq_constraints.shape
            sign = torch.sign(ineq_constraints)
            lagrangian += torch.sum(ineq_multipliers() * sign * ineq_constraints.abs().pow(self.order))

        if eq_constraints is not None:
            assert eq_multipliers.shape == eq_constraints.shape
            sign = torch.sign(eq_constraints)
            lagrangian += torch.sum(eq_multipliers() * sign * eq_constraints.abs().pow(self.order))

        return lagrangian

    @torch.no_grad()
    def update_state_(self, *args, **kwargs):
        """Multipliers are updated on every step based on the constraint violation."""
        pass

    @property
    def update_multipliers_on_step(self):
        """Multipliers are updated on every step based on the constraint violation."""
        return True

    def update_multipliers_(
        self,
        ineq_constraints: Optional[tuple[torch.Tensor]] = None,
        eq_constraints: Optional[tuple[torch.Tensor]] = None,
        ineq_multipliers: Optional[tuple[torch.Tensor]] = None,
        eq_multipliers: Optional[tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        if ineq_multipliers is None and eq_multipliers is None:
            raise ValueError("No multipliers to update")

        for multiplier, constraint in zip([ineq_multipliers, eq_multipliers], [ineq_constraints, eq_constraints]):
            if multiplier is not None:
                multiplier.weight.data += self.dual_lr * constraint
                multiplier.post_step_()


def LagrangianFormulation(*args, **kwargs):
    return KthOrderLagrangianFormulation(order=1, *args, **kwargs)
