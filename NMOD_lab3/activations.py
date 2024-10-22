from typing import Callable

import numpy as np


class Activation:
    def __init__(self,
                 activation_fn: Callable[[np.array], np.array],
                 derivative_fn: Callable[[np.array], np.array]):
        self.activation_fn = activation_fn
        self.derivative_fn = derivative_fn

    def __call__(self, x: np.array) -> np.array:
        return self.activation_fn(x)
