import numpy as np


class Kernel:
    @staticmethod
    def linear(sigma):
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2)))

    @classmethod
    def dispatch(cls, name, sigma):
        try:
            return getattr(cls, name)(sigma)
        except AttributeError:
            raise ValueError("kernel choice must in list: " + str(cls._get_kernel_list()))

    @classmethod
    def _get_kernel_list(cls):
        return [method for method, _type_str in cls.__dict__.items() if "staticmethod object" in str(_type_str)]


class SVC:
    def __init__(self, c=1, epsilon=1e-3, verbose=False):
        self.C = c
        self.epsilon = epsilon
        self.verbose = verbose

        # during fitting
        self._X = None
        self._y = None
        self._alpha = None
        self._b = None
        # monitor
        self.fit_reach_epochs_end = None

    def predict(self, X):
        y_pred = self.decision_function(X)
        return np.sign(np.sign(y_pred) + 0.1)

    def decision_function(self, X):
        w = np.dot(self._alpha * self._y, self._X)
        y_pred = np.dot(X, w) + self._b
        return y_pred

    def _fit_init(self, X, y):
        self._X = np.atleast_2d(X)
        self._y = np.asarray(y)
        self._alpha = np.random.random(len(X)) * min(self.C, 1)
        self._b = np.random.random(1) * min(self.C, 1)

    def fit(self, X, y, epochs=10 ** 4):
        self._fit_init(X, y)
        self.fit_reach_epochs_end = False
        for epoch in range(epochs):
            idx1 = self._pick_first_alpha()
            if idx1 is None:
                if self.verbose:
                    print('fit iter:', epoch)
                return self
            idx2 = self._pick_second_alpha(idx1)
            if self.verbose:
                print(idx1, idx2)
            self._update_alpha(idx1, idx2)
        self.fit_reach_epochs_end = True
        if self.verbose:
            print('fit iter:', epochs)
            print('alpha error:', self._pick_first_alpha(return_err=True))
        return self

    def _pick_first_alpha(self, return_err=False):
        assert (self._alpha >= -1).all(), self._alpha
        cond1 = self._alpha <= 0
        cond2 = np.logical_and(0 < self._alpha, self._alpha < self.C)
        cond3 = self._alpha >= self.C
        err = np.asarray(self._y * self.decision_function(self._X) - 1)
        err[np.logical_and(cond1, err > 0)] = 0
        err[np.logical_and(cond2, err == 0)] = 0
        err[np.logical_and(cond3, err < 0)] = 0
        err = err ** 2
        idx = np.argmax(err)
        if self.verbose:
            print(err[idx], end=':')
            print(self._alpha, end=' ')
        if return_err:
            return err[idx]
        if err[idx] < self.epsilon:
            return
        return idx

    def _pick_second_alpha(self, idx1):
        idx2 = np.random.randint(len(self._y))
        while idx2 == idx1:
            idx2 = np.random.randint(len(self._y))
        return idx2

    def _update_alpha(self, idx1, idx2):
        """
        P145 in <<统计学习方法>>
        """
        l, h = self._get_lower_bound(idx1, idx2), self._get_upper_bound(idx1, idx2)
        y1, y2 = self._y[idx1], self._y[idx2]
        alpha1, alpha2 = self._alpha[idx1], self._alpha[idx2]
        E1 = self.decision_function(self._X[idx1]) - y1
        E2 = self.decision_function(self._X[idx2]) - y2
        eta = np.inner(idx1, idx1) + np.inner(idx2, idx2) - 2 * np.inner(idx1, idx2)
        alpha2_new_unc = self._alpha[idx2] + y2 * (E1 - E2) / eta
        #
        alpha2_new = np.clip(alpha2_new_unc, l, h)
        alpha1_new = self._alpha[idx1] + y1 * y2 * (self._alpha[idx2] - alpha2_new)
        self._alpha[idx1] = alpha1_new
        self._alpha[idx2] = alpha2_new
        self._update_b_cache(E1, E2, idx1, idx2, alpha1, alpha2)

    def _get_lower_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return max(0, self._alpha[idx2] - self._alpha[idx1])
        return max(0, self._alpha[idx2] + self._alpha[idx1] - self.C)

    def _get_upper_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return min(self.C, self.C + self._alpha[idx2] - self._alpha[idx1])
        return min(self.C, self._alpha[idx2] + self._alpha[idx1])

    def _update_b_cache(self, E1, E2, idx1, idx2, alpha_old_1, alpha_old_2):
        y1, y2 = self._y[idx1], self._y[idx2]
        alpha_new_1, alpha_new_2 = self._alpha[idx1], self._alpha[idx2]
        db1 = -E1 - y1 * np.inner(idx1, idx1) * (alpha_new_1 - alpha_old_1) \
              - y2 * np.inner(idx2, idx1) * (alpha_new_2 - alpha_old_2)
        db2 = -E2 - y1 * np.inner(idx1, idx2) * (alpha_new_1 - alpha_old_1) \
              - y2 * np.inner(idx2, idx2) * (alpha_new_2 - alpha_old_2)
        self._b += (db1 + db2) / 2


if __name__ == '__main__':
    x = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [2, 0]])
    y = np.array([-1, 1, -1, -1, 1])

    from sklearn import svm

    svc = SVC(c=100, verbose=True)
    # svc = svm.SVC(C=100, kernel='linear')
    svc.fit(x, y)
    print(svc.predict(x))
    # print(svc.fit_reach_epochs_end)
