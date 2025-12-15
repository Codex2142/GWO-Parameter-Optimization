import numpy as np
import random


class GrayWolfOptimizer:
    def __init__(self, n_wolves, max_iter, dim, lb, ub):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub

        self.positions = np.random.uniform(lb, ub, (n_wolves, dim))

        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)

        self.alpha_score = -np.inf
        self.beta_score = -np.inf
        self.delta_score = -np.inf

    def optimize(self, fitness_func):
        for t in range(self.max_iter):
            for i in range(self.n_wolves):
                score = fitness_func(self.positions[i])

                if score > self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = score, self.positions[i].copy()

                elif score > self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = score, self.positions[i].copy()

                elif score > self.delta_score:
                    self.delta_score, self.delta_pos = score, self.positions[i].copy()

            a = 2 - t * (2 / self.max_iter)

            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i][j] = (X1 + X2 + X3) / 3

                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.alpha_pos, self.alpha_score
