from typing import Optional

import torch

from .formulation import Formulation


class AugmentedLagrangianFormulation(Formulation):
    def __init__(self, penalty_coefficient: float, penalty_gamma: float, *args, **kwargs):
        assert penalty_coefficient > 0
        assert penalty_gamma > 1

        self.penalty_coefficient = penalty_coefficient
        self.penalty_gamma = penalty_gamma

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
            lagrangian += torch.sum(ineq_multipliers() * ineq_constraints)
            lagrangian += self.penalty_coefficient * ineq_constraints.relu().pow(2).sum() / 2

        if eq_constraints is not None:
            assert eq_multipliers.shape == eq_constraints.shape
            lagrangian += torch.sum(eq_multipliers() * eq_constraints)
            lagrangian += self.penalty_coefficient * eq_constraints.pow(2).sum() / 2

        return lagrangian

    @torch.no_grad()
    def update_state_(self, *args, **kwargs):
        """Multipliers are updated on every step based on the constraint violation."""
        # TODO: could update only if the constraint is violated
        self.penalty_coefficient *= self.penalty_gamma

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
                # The "learning rate" is the penalty coefficient
                multiplier.weight.data += self.penalty_coefficient * constraint
                multiplier.post_step_()
