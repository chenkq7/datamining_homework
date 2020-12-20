import numpy as np


class SVC:
    def __init__(self, c=1, epsilon=1e-3):
        self.C = c
        self.epsilon = epsilon

        # during fitting
        self._X = None
        self._y = None
        self._alpha = None
        self._b = None
        # cache
        self.err_cache = None
        self.decision_cache = None
        # monitor
        self.fit_reach_epoch = -1

    def get_decision_cache(self):
        if self.decision_cache is not None:
            return self.decision_cache
        self.decision_cache = self.decision_function(self._X)
        return self.decision_cache

    def get_err_cache(self):
        if self.err_cache is not None:
            return self.err_cache
        self.err_cache = np.asarray(self._y * self.get_decision_cache() - 1)
        return self.err_cache

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
        self._alpha = np.ones(len(X)) * min(self.C, 1)
        self._b = np.random.random(1) * min(self.C, 1)

    def fit(self, X, y, epochs=10 ** 4):
        self._fit_init(X, y)
        self.fit_reach_epoch = epochs
        for epoch in range(epochs):
            idx1 = self._pick_first_alpha()
            if idx1 is None:
                self.fit_reach_epoch = epoch
                break
            for i in range(len(self._y)):
                idx2 = self._pick_second_alpha(idx1)
                if self._update_alpha(idx1, idx2):
                    break
        return self

    def _pick_first_alpha(self):
        assert (self._alpha >= -1).all(), self._alpha
        cond1 = self._alpha <= 0
        cond2 = np.logical_and(0 < self._alpha, self._alpha < self.C)
        cond3 = self._alpha >= self.C
        err = self.get_err_cache().copy()
        err[np.logical_and(cond1, err > 0)] = 0
        err[np.logical_and(cond2, err == 0)] = 0
        err[np.logical_and(cond3, err < 0)] = 0
        err = err ** 2
        idx = np.argmax(err)
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
        L, H = self._get_lower_bound(idx1, idx2), self._get_upper_bound(idx1, idx2)
        if L == H:
            return False
        eta = np.inner(idx1, idx1) + np.inner(idx2, idx2) - 2 * np.inner(idx1, idx2)
        if eta <= 0:
            return False
        y1, y2 = self._y[idx1], self._y[idx2]
        alpha1, alpha2 = self._alpha[idx1], self._alpha[idx2]
        E1 = self.get_decision_cache()[idx1] - y1
        E2 = self.get_decision_cache()[idx2] - y2
        alpha2_new_unc = self._alpha[idx2] + y2 * (E1 - E2) / eta
        alpha2_new = np.clip(alpha2_new_unc, L, H)
        alpha1_new = self._alpha[idx1] + y1 * y2 * (self._alpha[idx2] - alpha2_new)
        self._alpha[idx1] = alpha1_new
        self._alpha[idx2] = alpha2_new
        self._update_b(E1, E2, idx1, idx2, alpha1, alpha2)
        self.err_cache = None
        self.decision_cache = None
        return True

    def _get_lower_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return max(0, self._alpha[idx2] - self._alpha[idx1])
        return max(0, self._alpha[idx2] + self._alpha[idx1] - self.C)

    def _get_upper_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return min(self.C, self.C + self._alpha[idx2] - self._alpha[idx1])
        return min(self.C, self._alpha[idx2] + self._alpha[idx1])

    def _update_b(self, E1, E2, idx1, idx2, alpha_old_1, alpha_old_2):
        y1, y2 = self._y[idx1], self._y[idx2]
        alpha_new_1, alpha_new_2 = self._alpha[idx1], self._alpha[idx2]
        db1 = -E1 - y1 * np.inner(idx1, idx1) * (alpha_new_1 - alpha_old_1) \
              - y2 * np.inner(idx2, idx1) * (alpha_new_2 - alpha_old_2)
        db2 = -E2 - y1 * np.inner(idx1, idx2) * (alpha_new_1 - alpha_old_1) \
              - y2 * np.inner(idx2, idx2) * (alpha_new_2 - alpha_old_2)
        self._b += (db1 + db2) / 2


if __name__ == '__main__':
    # x = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [2, 0]])
    # y = np.array([-1, 1, -1, -1, 1])
    #
    # svc = SVC(c=5)
    # svc.fit(x, y, epochs=10 ** 4)
    # print(svc.predict(x))

    from sklearn import svm
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


    data = pd.read_csv("../dataset/线下/svm/svm_training_set.csv")
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

    svc = SVC(c=1, epsilon=1e-3)
    # svc = svm.SVC(kernel='linear')
    svc.fit(x_train, y_train, epochs=int(5e4))
    # svc.fit(x_train, y_train)

    train_pred = svc.predict(x_train)
    print(np.asarray(train_pred == y_train).mean())
    val_pred = svc.predict(x_val)
    print(np.asarray(val_pred == y_val).mean())
    # print(svc.fit_reach_epoch)
    from sklearn import metrics

    print(metrics.f1_score(y_val, val_pred))
    print(metrics.recall_score(y_val, val_pred))
    print(metrics.accuracy_score(y_val, val_pred))
