import math
import torch


class D2BetaController:
    def __init__(
        self,
        num_components: int,
        target_d2: float,
        beta_init: float = 1e-3,
        beta_min: float  = 1e-8,
        beta_max: float  = 1e1,
        eta: float = 0.05,
        ema_gamma: float = 0.1,
        deadzone: float  = 0.0,
    ) -> None:
        self.num_components = num_components
        self.target_d2 = target_d2
        self.beta_min  = beta_min
        self.beta_max  = beta_max
        self.eta = eta
        self.ema_gamma = ema_gamma
        self.deadzone  = deadzone

        self.log_beta = math.log(beta_init)
        self.d2_ema: float | None = None

    def get_beta(self) -> float:
        beta = math.exp(self.log_beta)
        return max(self.beta_min, min(self.beta_max, beta))

    def update(self, d2_epoch: float) -> tuple[float, float]:
        if self.d2_ema is None:
            self.d2_ema = d2_epoch
        else:
            self.d2_ema = (1.0 - self.ema_gamma) * self.d2_ema + self.ema_gamma * d2_epoch

        error = (self.target_d2 - self.d2_ema) / max(self.num_components - 1, 1)
        if abs(error) > self.deadzone:
            self.log_beta += self.eta * error
            self.log_beta = min(math.log(self.beta_max), max(math.log(self.beta_min), self.log_beta))

        return self.d2_ema, self.get_beta()
