from . import rk_common
from .solvers import FixedGridODESolver


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y, solutions):
        return tuple(dt * f_ for f_ in func(t, y, solutions))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y, his_solution):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y, his_solution):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4
