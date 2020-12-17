import numpy as np
import pandas as pd


def nominal2binary(data, nominals):
    """
    :param data: pd.DataFrame.
    :param nominals: columns whose data type is Nominal.
    :return: convert Nominal data type to one-hot binary data type.
    """
    assert isinstance(data, pd.DataFrame)
    for nominal in nominals:
        max_value = max(data[nominal])
        if max_value >= 2:
            for v in range(max_value + 1):
                data.loc[:, nominal + '_' + str(v)] = (data.loc[:, nominal] == v).astype(int)
    columns = []
    for col in data.columns.values:
        if col not in nominals:
            columns.append(col)
    return data[columns]


def f1_score(y_true, y_pred, pos_label=1, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = set(y_true)
    assert not set(y_true).difference(labels)
    assert not set(y_pred).difference(labels)
    pos_pred = (y_pred == pos_label).sum()
    pos_true = (y_true == pos_label).sum()
    tp = np.logical_and(y_true == pos_label, y_pred == pos_label).sum()
    precision = tp / pos_pred if pos_pred else 0
    recall = tp / pos_true if pos_true else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


class Kernel:
    @staticmethod
    def linear(sigma):
        return lambda x, y: np.inner(x, y)

    # @staticmethod
    # def gaussian(sigma):
    #     return lambda x, y: np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2)))

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
    def __init__(self, c=1, kernel='linear', sigma=1.0, epsilon=1e-3, verbose=False):
        self.kernel_fn = Kernel.dispatch(kernel, sigma)
        self.C = c
        self.epsilon = epsilon
        self.verbose = verbose
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
        self._b = np.random.random(1) * min(self.C, 1)
        self._w = self._alpha * self._y
        self._predict_cache = np.dot(self._w, self._gram) + self._b

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
            self._update_alpha(idx1, idx2)
        self.fit_reach_epochs_end = True
        if self.verbose:
            print('fit iter:', epochs)
            print('alpha error:', self._pick_first_alpha(return_err=True))
        return self

    def _pick_first_alpha(self, return_err=False):
        # assert (self._alpha >= 0).all(), self._alpha
        cond1 = self._alpha <= 0
        cond2 = np.logical_and(0 < self._alpha, self._alpha < self.C)
        cond3 = self._alpha >= self.C
        err = np.asarray(self._y * self._predict_cache - 1)
        err[np.logical_and(cond1, err > 0)] = 0
        err[np.logical_and(cond2, err == 0)] = 0
        err[np.logical_and(cond3, err < 0)] = 0
        err = self._alpha * err + self.C * np.maximum(-err, 0)
        err = np.abs(err)
        idx = np.argmax(err)
        if self.verbose:
            print(err[idx])
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
        E1 = self._predict_cache[idx1] - y1
        E2 = self._predict_cache[idx2] - y2
        eta = self._gram[idx1][idx1] + self._gram[idx2][idx2] - 2 * self._gram[idx1][idx2]
        if eta == 0:
            return
        alpha2_new_unc = self._alpha[idx2] + y2 * (E1 - E2) / eta
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
    data = pd.read_csv("./dataset/线下/svm/svm_training_set.csv")
    idx_train = np.random.choice(len(data), int(9 / 10 * len(data)))
    idx_val = list(set(list(range(len(data)))).difference(idx_train))
    nominals = ['x1', 'x4', 'x6', 'x7', 'x8', 'x9']
    ord_rat = ['x5', 'x2', 'x3', 'x10', 'x11', 'x12']
    x_train = nominal2binary(data.iloc[idx_train, :12], nominals=nominals)
    x_val = nominal2binary(data.iloc[idx_val, :12], nominals=nominals)
    y_train = data.iloc[idx_train, 12]
    y_val = data.iloc[idx_val, 12]

    y_neg_idx = np.arange(len(y_train))[y_train < 0]
    y_neg_idx_sam = np.random.choice(y_neg_idx, 6000, replace=False)
    y_pos_idx_sam = np.arange(len(y_train))[y_train > 0]
    idx_sample = np.concatenate((y_neg_idx_sam, y_pos_idx_sam))
    idx_sample = np.random.choice(idx_sample, 2000, replace=False)

    x_train = x_train.iloc[idx_sample]
    y_train = y_train.iloc[idx_sample]

    for col in x_train.columns.values:
        if str(col).startswith(tuple(nominals)):
            max_v = np.max(x_train[col])
            min_v = np.min(x_train[col])
            if max_v != min_v:
                x_train[col] = (x_train[col] - min_v) / (max_v - min_v)
                x_val[col] = (x_val[col] - min_v) / (max_v - min_v)
        else:
            mean = np.mean(x_train[col])
            std = np.std(x_train[col])
            x_train[col] = (x_train[col] - mean) / std
            x_val[col] = (x_val[col] - mean) / std

    print(pd.DataFrame(x_train).columns.values)
    print(pd.DataFrame(y_train).columns.values)

    svc = SVC(c=1, verbose=True)
    svc.fit(x_train, y_train, epochs=int(1e5))

    val_pred = svc.predict(x_val)
    print(np.asarray(val_pred == y_val).mean())

    from sklearn import metrics

    print(f1_score(y_val, val_pred))
    print(metrics.f1_score(y_val, val_pred))
    print(metrics.recall_score(y_val, val_pred))
    print(metrics.accuracy_score(y_val, val_pred))
