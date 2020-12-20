import numpy as np
import pandas as pd


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


class LinearSVC:
    def __init__(self, c=1, epsilon=1e-3, verbose=False):
        self.kernel_fn = np.inner
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

    def state_dict(self):
        ret = {
            'C': self.C,
            'epsilon': self.epsilon,
            'verbose': self.verbose,
            '_alpha': self._alpha.tolist(),
            '_b': self._b.tolist(),
            '_X': self._X.tolist(),
            '_y': self._y.tolist(),
        }
        return ret

    @staticmethod
    def load(state_dict):
        self = LinearSVC(c=state_dict['C'], epsilon=state_dict['epsilon'], verbose=state_dict['verbose'])
        self._alpha = np.asarray(state_dict['_alpha'])
        self._b = np.asarray(state_dict['_b'])
        self._X = np.asarray(state_dict['_X'])
        self._y = np.asarray(state_dict['_y'])
        self._w = self._alpha * self._y
        return self

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


class PreProcess:
    def __init__(self, nominal):
        self.nominal = tuple(nominal)
        self.columns = np.asarray([])
        self.normalization_params = {}

    def state_dict(self):
        ret = {
            'nominal': self.nominal,
            'columns': self.columns.tolist(),
            'dict': self.normalization_params
        }
        return ret

    @staticmethod
    def load(state_dict):
        self = PreProcess(state_dict['nominal'])
        self.columns = state_dict['columns']
        self.normalization_params = state_dict['dict']
        return self

    def nominal2binary(self, data_):
        """
        :param data_: pd.DataFrame.
        :return: convert Nominal data type to one-hot binary data type.
        """
        data = pd.DataFrame(data_).copy()
        assert isinstance(data, pd.DataFrame)
        for nom in self.nominal:
            max_value = max(data[nom])
            if max_value >= 2:
                for v in range(max_value + 1):
                    data.loc[:, nom + '_' + str(v)] = (data.loc[:, nom] == v).astype(int)
        columns = []
        for col in data.columns.values:
            if col not in self.nominal:
                columns.append(col)
        return data[columns]

    def normalization_init(self, x_train):
        self.columns = np.asarray(x_train.columns.values)
        for col in x_train.columns.values:
            if str(col).startswith(tuple(self.nominal)):
                max_v = np.max(x_train[col])
                min_v = np.min(x_train[col])
                self.normalization_params[str(col)] = np.asarray([min_v, max_v]).tolist()
            else:
                mean = np.mean(x_train[col])
                std = np.std(x_train[col])
                self.normalization_params[str(col)] = np.asarray([mean, std]).tolist()

    def normalize(self, x_):
        x = x_.copy()
        assert (self.columns == np.asarray(x.columns.values)).all()
        for col in x.columns.values:
            if str(col).startswith(tuple(self.nominal)):
                min_v, max_v = self.normalization_params[str(col)]
                if min_v != max_v:
                    x[col] = (x[col] - min_v) / (max_v - min_v)
            else:
                mean, std = self.normalization_params[str(col)]
                if std != 0:
                    x[col] = (x[col] - mean) / std
        return x


class Model:
    def __init__(self, nominal=(), c=1, epsilon=1e-3, verbose=False):
        self.pp = PreProcess(nominal)
        self.svc = LinearSVC(c, epsilon=epsilon, verbose=verbose)

    def state_dict(self):
        ret = {
            'pp': self.pp.state_dict(),
            'svc': self.svc.state_dict(),
        }
        return ret

    @staticmethod
    def load(state_dict):
        self = Model()
        self.pp = PreProcess.load(state_dict['pp'])
        self.svc = LinearSVC.load(state_dict['svc'])
        return self

    def fit(self, x, y, epochs=int(1e5)):
        x = self.pp.nominal2binary(x)
        self.pp.normalization_init(x)
        x = self.pp.normalize(x)
        print(pd.DataFrame(x).columns.values)
        self.svc.fit(x, y, epochs=epochs)

    def predict(self, x):
        x = self.pp.nominal2binary(x)
        x = self.pp.normalize(x)
        return self.svc.predict(x)


def train(train_set="./dataset/线下/svm/svm_training_set.csv", train_rate=0.9, sample_num=2000, C=1, epochs=int(1e4),
          save_model=None, verbose=True):
    # div data into trainset valset
    data = pd.read_csv(train_set)
    idx_train = np.random.choice(len(data), int(train_rate * len(data)), replace=False)
    x_train = data.iloc[idx_train, :12]
    y_train = data.iloc[idx_train, 12]
    idx_val = list(set(list(range(len(data)))).difference(idx_train))
    x_val = data.iloc[idx_val, :12]
    y_val = data.iloc[idx_val, 12]

    # sample among trainset to
    # 1. balance the pos/neg data
    # 2. make dataset small enough to feed into my svc
    y_neg_idx = np.arange(len(y_train))[y_train < 0]
    y_pos_idx_sam = np.arange(len(y_train))[y_train > 0]
    y_neg_idx_sam = np.random.choice(y_neg_idx, len(y_pos_idx_sam), replace=False)
    idx_sample = np.concatenate((y_neg_idx_sam, y_pos_idx_sam))
    idx_sample = np.random.choice(idx_sample, sample_num, replace=False)
    x_train = x_train.iloc[idx_sample]
    y_train = y_train.iloc[idx_sample]

    # create model
    nominal = ['x1', 'x4', 'x6', 'x7', 'x8', 'x9']
    ord_rat = ['x5', 'x2', 'x3', 'x10', 'x11', 'x12']
    svc = Model(nominal, c=C, verbose=verbose)

    # train model
    svc.fit(x_train, y_train, epochs=epochs)

    # val model
    val_pred = svc.predict(x_val)
    print("val mean():", np.asarray(val_pred == y_val).mean())
    print("val f1:", f1_score(y_val, val_pred))

    # save model
    if save_model is not None:
        save_filename = str(save_model)
        if not save_filename.endswith('.json'):
            save_filename += '.json'
        with open(save_filename, 'w') as f:
            import json
            sd = svc.state_dict()
            js = json.dumps(sd)
            f.write(js)

    return svc


def test(test_set, model_path, result_path):
    # data
    data = pd.read_csv(test_set)
    into_data = data.iloc[:, :12]

    # model
    import json
    with open(model_path, 'r') as f:
        js = f.read()
        sd = json.loads(js)
    svc = Model.load(sd)

    # pred
    pred = svc.predict(into_data)
    data['pred'] = pred
    data.to_csv(result_path, index=False)

    # ret
    return pred


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    train_group = parser.add_argument_group('train model')
    train_group.add_argument('--train_set', help='train dataset path')
    train_group.add_argument('--train_rate', type=float, help='the rate of (train num)/(tot num). other is val set.',
                             default=0.9)
    train_group.add_argument('--sample_num', type=int, help='sample num for training.', default=2000)
    train_group.add_argument('--C', type=float, help='the C param in svc', default=1.0)
    train_group.add_argument('--epochs', type=int, help='max epochs for svc fitting', default=int(1e5))
    train_group.add_argument('--save_model', help='if not None, SAVE_MODEL will be the file to save the model')
    train_group.add_argument('--verbose', action='store_true', default=False)

    test_group = parser.add_argument_group('test model')
    test_group.add_argument("--test_set", help='test set path')
    test_group.add_argument("--model_path", help='saved model path')
    test_group.add_argument("--result_path", help='result file')

    args = parser.parse_args()
    if args.train:
        train(args.train_set, args.train_rate, args.sample_num, args.C, args.epochs, args.save_model, args.verbose)
    elif args.test:
        test(args.test_set, args.model_path, args.result_path)
    else:
        parser.print_help()
