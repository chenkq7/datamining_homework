import pandas as pd
import numpy as np


def sample(x_pd, y_pd, x_num=None, attr_num=None):
    """
    sample with replacement.
    :param x_pd: pd.DataFrame (id,attr_1,...,attr_n).
    :param y_pd: pd.DataFrame (id,label).
    :param x_num: the sample num for data. if None, use suggested num: len(x_pd)
    :param attr_num: the sample num for attrs. if None, use suggested num: sqrt(attrs_num)
    :return: x_sample(id)  x_sample(attr1,...)  y_sample(label)
             x_reverse(id) x_reverse(attr1,...) y_reverse(label)
             attr_sample: the sampled attr list [attr1,...]
    """
    assert isinstance(x_pd, pd.DataFrame)
    assert isinstance(y_pd, pd.DataFrame)
    x_head = x_pd.columns.values
    y_head = y_pd.columns.values
    _id, attrs, label = x_head[0], x_head[1:], y_head[1]
    if x_num is None:
        x_num = len(x_pd)
    if attr_num is None:
        from numpy import sqrt
        attr_num = int(sqrt(len(attrs)) + 1)

    # align ids so that sample easily
    if not pd.Series(x_pd[_id] == y_pd[_id]).all():
        y_pd = x_pd.merge(y_pd, on=_id, how='left')[y_pd.columns.values]

    # sample data
    from numpy.random import choice
    idx_sample = choice(len(x_pd), x_num)
    idx_reverse = list(set(list(range(len(x_pd)))).difference(idx_sample))
    x_sample = x_pd.iloc[idx_sample].reset_index()
    y_sample = y_pd.iloc[idx_sample].reset_index()
    x_reverse = x_pd.iloc[idx_reverse].reset_index()
    y_reverse = y_pd.iloc[idx_reverse].reset_index()

    # sample attrs
    attr_sample = choice(attrs, attr_num, replace=False)

    # return
    return ((x_sample[_id], x_sample[attr_sample], y_sample[label]),
            (x_reverse[_id], x_reverse[attr_sample], y_reverse[label]),
            attr_sample)


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


class PredVoter:
    def __init__(self):
        self.dict = {}

    def clear(self):
        self.dict = {}

    def accumulate(self, ids, preds):
        """
        :param ids: array-like.
        :param preds: array-like. value in {-1, +1}
        """
        for _id, _pred in zip(ids, preds):
            self.dict.setdefault(_id, [])
            self.dict[_id].append(_pred)

    def length(self):
        lengths = [len(v) for k, v in self.dict.items()]
        length2cnt = {}
        for length in lengths:
            length2cnt[length] = length2cnt.get(length, 0) + 1
        max_len = max(lengths)
        min_len = min(lengths)
        mean_len = sum(lengths) / len(lengths)
        mode_len = max(length2cnt, key=lambda x: length2cnt[x])
        return min_len, mode_len, mean_len, max_len

    def vote(self):
        """
        :return: np.array (ids, preds)
        """
        voted = {}
        for _id in self.dict:
            voted[_id] = int(np.sign(sum(self.dict[_id])))
            if voted[_id] == 0:
                voted[_id] = self.dict[_id][-1]  # if draw, random pick one
        ret = np.asarray([[X, y] for X, y in voted.items()])
        return ret


class TreeNode:
    def __init__(self):
        self.is_leaf = None
        # leaf
        self.prob = None
        # inner
        self.feature = None
        self.children = {}
        # build
        self.entropy = None

    def state_dict(self):
        child_state_dict = {}
        for child, node in self.children.items():
            child_state_dict[child] = node.state_dict()
        state_dict = {
            'is_leaf': self.is_leaf,
            'prob': list(self.prob.items()),
            'feature': self.feature,
            'entropy': self.entropy,
            'children': list(child_state_dict.items())
        }
        return state_dict

    @staticmethod
    def load(state_dict):
        self = TreeNode()
        self.is_leaf = state_dict['is_leaf']
        self.prob = dict(state_dict['prob'])
        self.feature = state_dict['feature']
        self.entropy = state_dict['entropy']
        child_state_dict = dict(state_dict['children'])
        for child, child_state_dict in child_state_dict.items():
            self.children[child] = TreeNode.load(child_state_dict)
        return self

    def predict(self, x):
        if self.is_leaf:
            return max(self.prob, key=self.prob.get)
        if x[self.feature] not in self.children:
            return max(self.prob, key=self.prob.get)
        return self.children[x[self.feature]].predict(x)

    def build_tree(self, X, y, features, epsilon=0):
        X = np.asarray(X)
        y = np.asarray(y)
        features = np.asarray(list(features)).tolist()
        assert isinstance(features, list)
        features = set(features)
        self.prob = self.calculate_probability(y)
        self.entropy = self.calculate_entropy(y)

        # only one label  or  features is empty set
        if len(set(y)) == 1 or (not features):
            self.is_leaf = True
            return
        # calculate gain
        max_gain, best_fea = -1, None
        for fea in features:
            gain = self.calculate_gain_rate(X, y, fea)
            if gain > max_gain:
                max_gain = gain
                best_fea = fea
        if max_gain < epsilon:
            self.is_leaf = True
            return
        # split and build recursive
        self.feature = best_fea
        fea_child = features.difference({best_fea})
        fea_value = np.asarray(X[:, best_fea]).tolist()
        assert isinstance(fea_value, list)
        fea_value = set(fea_value)
        for value in fea_value:
            idx_ = X[:, best_fea] == value
            self.children[value] = TreeNode()
            self.children[value].build_tree(X[idx_], y[idx_], fea_child, epsilon)
        return self

    def calculate_gain_rate(self, X, y, feature):
        if len(set(X[:, feature])) == 1:
            return 0
        old_entropy = self.entropy
        new_entropy = self.calculate_condition_entropy(X, y, feature)
        gain = old_entropy - new_entropy
        feature_entropy = self.calculate_entropy(X[:, feature])
        return gain / feature_entropy

    @staticmethod
    def calculate_entropy(y):
        y = np.asarray(y)
        value = set(y)
        entropy = 0
        for v in value:
            v_num = (y == v).sum()
            prob = v_num / len(y)
            entropy -= prob * np.log2(prob)
        return float(entropy)

    @staticmethod
    def calculate_condition_entropy(X, y, feature):
        fea_value = set(X[:, feature])
        condition_entropy = 0
        for value in fea_value:
            idx_ = np.asarray(X[:, feature] == value)
            child_entropy = TreeNode.calculate_entropy(y[idx_])
            condition_entropy -= idx_.sum() / len(y) * child_entropy
        return condition_entropy

    @staticmethod
    def calculate_probability(y):
        y = np.asarray(y).tolist()
        assert isinstance(y, list)
        predict = {}
        for y_item in y:
            predict[y_item] = predict.get(y_item, 0) + 1
        for key in predict:
            predict[key] = float(predict[key] / len(y))
        return predict


class DecisionTree:
    def __init__(self, columns):
        self.columns = list(columns) if columns is not None else None
        self.tree = None

    def fit(self, X, y, epsilon=0):
        """
        :param X: [attr_1,attr_2,...,attr_n]
        :param y: [label]
        :param epsilon: if gain < epsilon then stop build the tree
        """
        X = np.asarray(X)
        y = np.asarray(y)
        features = set(list(range(X.shape[-1])))
        self.tree = TreeNode().build_tree(X, y, features, epsilon)
        return self

    def state_dict(self):
        return {
            'columns': self.columns,
            'tree': self.tree.state_dict()
        }

    @staticmethod
    def load(state_dict):
        self = DecisionTree(columns=None)
        self.columns = state_dict['columns']
        self.tree = TreeNode.load(state_dict['tree'])
        return self

    def predict(self, X):
        X = np.asarray(X)
        y = []
        for x in X:
            y.append(self.tree.predict(x))
        return np.asarray(y)

    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_pred = self.predict(X)
        return (y == y_pred).mean()


class RandomForest:
    def __init__(self, tree_num, attr_num=None):
        """
        :param tree_num:
        :param attr_num: if None, use suggest num: sqrt(attrs_num)
        """
        self.tree_num = tree_num
        self.attr_num = attr_num
        self.voter_during_fit = PredVoter()
        self.trees = []
        self._id = None
        self._label = None
        self._pred = 'pred'

    def state_dict(self):
        state_dict = {
            'tree_num': self.tree_num,
            'attr_num': self.attr_num,
            'trees': [tree.state_dict() for tree in self.trees],
            '_id': self._id,
            '_label': self._label,
            '_pred': self._pred,
        }
        return state_dict

    @staticmethod
    def load(state_dict):
        self = RandomForest(tree_num=None)
        self.tree_num = state_dict['tree_num']
        self.attr_num = state_dict['attr_num']
        self.trees = [DecisionTree.load(tree_str) for tree_str in state_dict['trees']]
        self._id = state_dict['_id']
        self._label = state_dict['_label']
        self._pred = state_dict['_pred']
        return self

    def fit(self, x_data, y_data):
        """
        :param x_data: pd.DataFrame (id,attr_1,...attr_n)
        :param y_data: pd.DataFrame (id,label)
        :return: self
        """
        self.trees.clear()
        self.voter_during_fit.clear()
        self._id, self._label = y_data.columns.values

        for i in range(self.tree_num):
            (xs_id, x_sample, y_sample), (xr_id, x_reverse, y_reverse), attr_sample = sample(x_data, y_data,
                                                                                             attr_num=self.attr_num)
            clf = DecisionTree(attr_sample)
            clf = clf.fit(x_sample, y_sample)
            self.trees.append(clf)
            pred = clf.predict(x_reverse)
            self.voter_during_fit.accumulate(xr_id, pred)
        return self

    def predict(self, x_data):
        """
        :param x_data: pd.DataFrame (id,attr_1,...attr_n)
        :return: np.array (id,pred)
        """
        voter = PredVoter()
        x_id = x_data[self._id]
        for clf in self.trees:
            x_clf = x_data[clf.columns]
            pred = clf.predict(x_clf)
            voter.accumulate(x_id, pred)
        voted = voter.vote()
        return voted

    def f1_score(self, x_data, y_data):
        """
        :param x_data: pd.DataFrame (id,attr_1,...attr_n)
        :param y_data: pd.DataFrame (id,label)
        :return: f1_score
        """
        voted = self.predict(x_data)
        df = pd.DataFrame(voted, columns=[self._id, self._pred])
        df = df.merge(y_data, how='left')
        right, pred = df[self._label], df[self._pred]
        return f1_score(right, pred)

    def f1_score_on_oob(self, y_data):
        """
        :return: the Out-Of-Bag(oob) f1 score.
        """
        voted = self.voter_during_fit.vote()
        if voted.size > 0:
            df = pd.DataFrame(voted, columns=[self._id, self._pred])
            df = df.merge(y_data, how='left')
            right, pred = df[self._label], df[self._pred]
            return f1_score(right, pred)
        return 0


def save_state_dict(state_dict, filename):
    import json
    js = json.dumps(state_dict)
    if not str(filename).endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as f:
        f.write(js)


def load_state_dict(filename):
    import json
    if not str(filename).endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as f:
        js = f.read()
        return json.loads(js)


def train(x_data_path="./dataset/线下/rf/x_train.csv", y_data_path="./dataset/线下/rf/y_train.csv", train_data_rate=0.9,
          tree_num=200, attr_num=None, save_model=None):
    x_data = pd.read_csv(x_data_path)
    y_data = pd.read_csv(y_data_path)

    idx_train = np.random.choice(len(x_data), int(len(x_data) * train_data_rate), replace=False)
    idx_val = list(set(list(range(len(x_data)))).difference(set(idx_train)))
    x_train = x_data.iloc[idx_train]
    y_train = y_data.iloc[idx_train]
    x_val = x_data.iloc[idx_val]
    y_val = y_data.iloc[idx_val]

    rf = RandomForest(tree_num, attr_num)
    rf = rf.fit(x_train, y_train)
    print("oob f1:", rf.f1_score_on_oob(y_train))
    print("val f1:", rf.f1_score(x_val, y_val))
    print("whole dataset f1:", rf.f1_score(x_data, y_data))

    if save_model is not None:
        save_state_dict(rf.state_dict(), save_model)
    return rf


def test(csv_data_path, model_state_dict, result_path):
    x_data = pd.read_csv(csv_data_path)
    sd = load_state_dict(model_state_dict)
    rf = RandomForest.load(sd)
    pred = rf.predict(x_data)
    result = pd.DataFrame(pred, columns=[rf._id, rf._pred])
    if result_path is not None:
        if not str(result_path).endswith('.csv'):
            result_path = str(result_path) + '.csv'
        result.to_csv(result_path, index=False)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='random_forests train or test.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store_true', help='train model')

    train_group = parser.add_argument_group('train model')
    train_group.add_argument('--x_path', help='x csv data path')
    train_group.add_argument('--y_path', help='y csv data path')
    train_group.add_argument('--train_rate', type=float, help='train data rate.',
                             default=0.9)
    train_group.add_argument('--tree_num', type=int,
                             help='the tree num of the random forest.', default=200)
    train_group.add_argument('--attr_num', type=int,
                             help='the attr num per tree has. if not specified, use suggested num.', default=None)
    train_group.add_argument('--save_model', help='save model to file SAVE_MODEL', default=None)

    parser.add_argument('--test', action='store_true', help='test model')
    test_group = parser.add_argument_group('test model')
    test_group.add_argument('--data_path', help='test csv data path')
    test_group.add_argument('--model_path', help='saved model path')
    test_group.add_argument('--result_path', help='result csv path')

    args = parser.parse_args()

    if args.train:
        train(args.x_path, args.y_path, args.train_rate, args.tree_num, args.attr_num, args.save_model)
    elif args.test:
        test(args.data_path, args.model_path, args.result_path)
    else:
        parser.print_help()
