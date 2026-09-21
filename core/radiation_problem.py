from core.base_problem import PhysicsProblem


class RadiationProblem(PhysicsProblem):
    def __init__(self):
        super().__init__()

    def pde_residual(self, batch, pred):
        raise NotImplementedError(
            "RadiationProblem not implemented in AIPlasma v1."
        )

    def boundary_conditions(self, batch, pred):
        raise NotImplementedError(
            "RadiationProblem not implemented in AIPlasma v1."
        )