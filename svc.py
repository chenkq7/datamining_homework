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
    def __init__(self, c=1, kernel='linear', sigma=1.0, epsilon=1e-3):
        self.kernel_fn = Kernel.dispatch(kernel, sigma)
        self.C = c
        self.epsilon = epsilon
        # during fitting
        self._alpha = None
        self._b = None
        self._w = None
        self._predict_cache = None
        # g(x) in <<统计学习方法>>
        self._X = None
        self._y = None
        self._gram = None
        # monitor
        self.fit_reach_epochs_end = None

    def predict(self, X, return_raw=False):
        X = self.kernel_fn(self._X, np.atleast_2d(X))
        y_pred = np.dot(self._w, X) + self._b
        if return_raw:
            return y_pred
        return np.sign(y_pred)

    def _fit_init(self, X, y):
        self._X = np.atleast_2d(X)
        self._y = np.asarray(y)
        self._gram = self.kernel_fn(X, X)
        #
        self._alpha = np.random.random(len(X)) * min(self.C, 1)
        self._b = np.random.random(len(X)) * min(self.C, 1)
        self._w = self._alpha * self._y
        self._predict_cache = np.dot(self._w, self._gram) + self._b

    def fit(self, X, y, epochs=10 ** 4):
        self._fit_init(X, y)
        self.fit_reach_epochs_end = False
        for epoch in range(epochs):
            idx1 = self._pick_first_alpha()
            if idx1 is None:
                return self
            idx2 = self._pick_second_alpha(idx1)
            self._update_alpha(idx1, idx2)
        self.fit_reach_epochs_end = True
        return self

    def _pick_first_alpha(self):
        assert (self._alpha >= 0).all()
        cond1 = self._alpha <= 0
        cond2 = np.logical_and(0 < self._alpha, self._alpha < self.C)
        cond3 = self._alpha >= self.C
        err = np.asarray(self._y * self._predict_cache - 1)
        err[np.logical_and(cond1, err > 0)] = 0
        err[np.logical_and(cond2, err == 0)] = 0
        err[np.logical_and(cond3, err < 0)] = 0
        err = err ** 2
        idx = np.argmax(err)
        # print(err[idx])
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
        E1 = self._predict_cache[idx1] - y1
        E2 = self._predict_cache[idx2] - y2
        eta = self._gram[idx1][idx1] + self._gram[idx2][idx2] - 2 * self._gram[idx1][idx2]
        alpha2_new_unc = self._alpha[idx2] + y2 * (E2 - E1) / eta
        #
        alpha2_new = np.clip(alpha2_new_unc, l, h)
        alpha1_new = self._alpha[idx1] + y1 * y2 * (self._alpha[idx2] - alpha2_new)
        self._alpha[idx1] = alpha1_new
        self._alpha[idx2] = alpha2_new
        self._update_b_cache(E1, E2, idx1, idx2, alpha1, alpha2)
        self._update_w_cache(idx1, idx2)
        self._update_predict_cache()
        pass

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
        db1 = -E1 - y1 * self._gram[idx1][idx1] * (alpha_new_1 - alpha_old_1) \
              - y2 * self._gram[idx2][idx1] * (alpha_new_2 - alpha_old_2)
        db2 = -E2 - y1 * self._gram[idx1][idx2] * (alpha_new_1 - alpha_old_1) \
              - y2 * self._gram[idx2][idx2] * (alpha_new_2 - alpha_old_2)
        self._b += (db1 + db2) / 2

    def _update_w_cache(self, idx1, idx2):
        self._w[idx1] = self._alpha[idx1] * self._y[idx1]
        self._w[idx2] = self._alpha[idx2] * self._y[idx2]

    def _update_predict_cache(self):
        self._predict_cache = self._b + np.dot(self._w, self._gram)


if __name__ == '__main__':
    x = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [2, 1]])
    y = np.array([-1, 1, -1, -1, 1])

    from sklearn import svm

    # svc = SVC()
    svc = svm.SVC(C=100, kernel='linear')
    svc.fit(x, y)
    print(svc.predict(x))
    # print(svc.fit_reach_epochs_end)
