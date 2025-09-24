class EGRMWOA():
    def __init__(self, fitness, D=D, P=P, G=G, ub=ub, lb=lb, b=b, a_max=a_max, a_min=a_min, a2_max=a2_max, a2_min=a2_min, l_max=l_max,
                 l_min=-l_min, b_max=b_max, b_min=b_min, sigma=sigma, radius=radius, lambda_num=lambda_num):
        self.dim = D
        self.pop = P
        self.T = G

        self.fitness = fitness
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        self.sigma = sigma
        self.b_max = b_max
        self.b_min = b_min
        self.F_alpha = np.inf
        self.F_beta = np.inf
        self.F_delta = np.inf
        self.F_gamma = np.inf
        self.X_alpha = np.zeros([self.dim])
        self.X_beta = np.zeros([self.dim])
        self.X_delta = np.zeros([self.dim])
        self.X_gamma = np.zeros([self.dim])
        self.gbest_X = np.zeros([self.dim])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.T)
        self.radius = radius
        self.lambda_num = lambda_num

    def opt(self):
        self.X = initialize_population()
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.pop, self.dim])
        radius = (self.ub[0] - self.lb[0]) / 2
        for g in range(self.T):
            radius = radius / (g + 1) ** 2
            self.loss_curve[g] = self.gbest_F
            self.X, F = self.OBL()
            F = np.zeros(self.pop)
            for i in range(self.pop):
                F[i] = self.fitness(self.X[i, :])
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            for i in range(self.pop):
                if F[i] < self.F_alpha:
                    self.F_alpha = F[i].copy()
                    self.X_alpha = self.X[i].copy()
                elif F[i] < self.F_beta:
                    self.F_beta = F[i].copy()
                    self.X_beta = self.X[i].copy()
                elif F[i] < self.F_delta:
                    self.F_delta = F[i].copy()
                    self.X_delta = self.X[i].copy()
                elif F[i] < self.F_gamma:
                    self.F_gamma = F[i].copy()
                    self.X_gamma = self.X[i].copy()
            # # 收斂曲线
            # self.loss_curve[g] = self.gbest_F
            rand = np.random.uniform(0, 1)
            randn = np.random.randn()
            R = g / self.T
            a = 2 * (R ** 4 - R ** 3 - R + 1)
            a2 = self.a2_max - (self.a2_max - self.a2_min) * (g / self.T)
            sigma = np.abs(1 - rand)
            a3 = 5 * np.exp(-16 * (R ** 2)) / np.sqrt(2 * np.pi)
            for i in range(self.pop):
                p = np.random.uniform()
                if p < 0.7:
                    p = p / 0.7
                else:
                    p = (1 - p) / 0.3
                q = np.random.uniform()
                if q < 0.7:
                    q = q / 0.7
                else:
                    q = (1 - q) / 0.3
                k = np.random.uniform()
                if k < 0.7:
                    k = k / 0.7
                else:
                    k = (k - k) / 0.3
                r1 = np.random.uniform()
                r2 = np.random.uniform()
                r7 = np.random.uniform()
                r4 = np.random.uniform()
                r5 = np.random.uniform()
                r6 = np.random.uniform()
                A = a * (2 * r1 - 1)
                C = 2 * r2

                r3 = np.random.uniform()
                l = (a2 - 1) * r3 + 1
                b = np.random.randint(low=self.b_min, high=self.b_max)

                num = max(1, self.dim // 2)
                num = min(num, 6)

                y = local_optim(self.X[i, :], F[i], radius, num)

                alpha_r1 = np.random.uniform()
                alpha_r2 = np.random.uniform()
                alpha_A = 2 * a * alpha_r1 - a
                alpha_C = 2 * alpha_r2
                alpha_D = self.X_alpha - self.X
                X1 = self.X_alpha + alpha_A * alpha_D

                beta_r1 = np.random.uniform()
                beta_r2 = np.random.uniform()
                beta_A = 2 * a * beta_r1 - a
                beta_C = 2 * beta_r2
                beta_D = self.X_beta - self.X
                X2 = self.X_beta + beta_A * beta_D

                delta_r1 = np.random.uniform()
                delta_r2 = np.random.uniform()
                delta_A = 2 * a * delta_r1 - a
                delta_C = 2 * delta_r2
                delta_D = self.X_delta - self.X
                X3 = self.X_delta + delta_A * delta_D

                w1 = (1 - g / self.T) ** (1 - np.tan(np.pi * (np.random.standard_cauchy() - 0.5) * g / self.T))
                w2 = (2 - 2 * g / self.T) ** (1 - np.tan(np.pi * (np.random.standard_cauchy() - 0.5) * g / self.T))
                if (g / self.T < 0.5):
                    w = w1
                else:
                    w = w2
                if p < 0.5:
                    p1 = 0.5 - (0.5 - 0.25) * np.sin(0.5 * np.pi * g / self.T)
                    rand = np.random.uniform()
                    if np.abs(A) < 1:
                        D = np.abs(C * self.gbest_X - self.X[i, :])
                        if (g / self.T < 0.5):
                            new_X = w * self.gbest_X - A * D
                        else:
                            new_X = self.gbest_X - w * A * D
                        mean_X = np.mean(self.X, axis=0)
                        beta = np.random.uniform()
                        new_X2 = new_X + mean_X * round(beta ** 2) * np.random.standard_cauchy()
                        if self.fitness(new_X) < self.fitness(new_X2):
                            pos_new = new_X
                        else:
                            pos_new = new_X2
                    else:
                        if k < 0.5:
                            P = F / np.sum(F)
                            PP = np.cumsum(P)
                            idx = np.random.randint(low=0, high=self.pop)
                            for i in range(len(PP)):
                                if PP[i - 1] <= r and r < PP[i]:
                                    idx = i
                                    break
                            X_rand = self.X[idx]
                            D = np.abs(C * X_rand - self.X[i, :])
                            if (g / self.T < 0.5):
                                pos_new = w * X_rand - A * D  # (8)
                            else:
                                pos_new = X_rand - w * A * D  # (8)
                        else:
                            self.X = elite_study(self.X, self.F, self.pop, self.dim, self.fitness)
                else:
                    if q < 0.5:
                        u = 1 - (1 - 0.25) * np.sin(0.5 * np.pi * g / self.T)
                        D = np.abs(self.gbest_X - self.X[i, :])
                        if (g / self.T < 0.5):
                            pos_new = w * self.gbest_X + D * np.exp(b * l) * np.cos(2 * np.pi * l)
                        else:
                            pos_new = self.gbest_X + w * D * np.exp(b * l) * np.cos(2 * np.pi * l)

                    else:
                        gwo_x = (X1[i, :] + X2[i, :] + X3[i, :]) / 3
                        L_Delta = abs(C * y - self.X[i, :])
                        xv = np.vstack((gwo_x, gwo_x + L_Delta, 0.8 * gwo_x + 0.2 * L_Delta, 0.7 * gwo_x + 0.3 * L_Delta,
                                        0.6 * gwo_x + 0.4 * L_Delta, 0.5 * gwo_x + 0.5 * L_Delta))
                        fx = np.array([self.fitness(xi) for xi in xv])
                        lv = np.argmin(fx)
                        pos_new = xv[lv]
                self.X[i, :] = pos_new

            self.X = Reflect(self.X)
            if (R > 0.75):
                for i in range(self.pop):
                    X_rand1 = self.X[np.random.randint(low=0, high=self.pop, size=self.dim), :]
                    X_rand1 = np.diag(X_rand1).copy()
                    X_rand2 = self.X[np.random.randint(low=0, high=self.pop, size=self.dim), :]
                    X_rand2 = np.diag(X_rand2).copy()
                    new_X = self.X[i, :] + lambda_num * (X_rand1 - X_rand2)
                    new_X = np.clip(new_X, self.lb, self.ub)
                    if (self.fitness(new_X) < F[i]):
                        self.X[i, :] = new_X

            self.X *= (1 + Levyflight(X))

            self.Reflect2()
            self.Reflect()
            self.X = np.clip(self.X, self.lb, self.ub)

            for i in range(self.pop):
                F[i] = self.fitness(self.X[i, :])

            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
