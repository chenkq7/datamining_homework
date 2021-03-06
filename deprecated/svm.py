import pandas as pd
import numpy as np


def t_sne(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    import time
    start = time.time()
    X_tsne = TSNE(n_components=2).fit_transform(X)
    end = time.time()
    print(end - start)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


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


if __name__ == '__main__':
    data = pd.read_csv("../dataset/线下/svm/svm_training_set.csv")
    idx_train = np.random.choice(len(data), int(9 / 10 * len(data)))
    idx_val = list(set(list(range(len(data)))).difference(idx_train))
    nominals = ['x1', 'x4', 'x6', 'x7', 'x8', 'x9']
    ord_rat = ['x5', 'x2', 'x3', 'x10', 'x11', 'x12']
    x_train = nominal2binary(data.iloc[idx_train, :12], nominals=nominals)
    x_val = nominal2binary(data.iloc[idx_val, :12], nominals=nominals)
    y_train = data.iloc[idx_train, 12]
    y_val = data.iloc[idx_val, 12]

    y_neg_idx = np.arange(len(y_train))[y_train<0]
    y_neg_idx_sam = np.random.choice(y_neg_idx,6000,replace=False)
    y_pos_idx_sam = np.arange(len(y_train))[y_train>0]
    idx_sample = np.concatenate((y_neg_idx_sam,y_pos_idx_sam))

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

    from sklearn import svm
    import time

    svc = svm.SVC(C=1, kernel='linear', verbose=True)
    start = time.time()
    svc.fit(x_train, y_train)
    end = time.time()
    print(end - start)
    pred = svc.predict(x_val)
    # print(f1_score(y_val, pred))
    print((np.asarray(y_val) == np.asarray(pred)).mean())

    from sklearn import metrics

    val_pred = pred
    print(metrics.f1_score(y_val, val_pred))
    print(metrics.recall_score(y_val, val_pred))
    print(metrics.accuracy_score(y_val, val_pred))

    """
    me notices:
    1. if use nominal2binary, the linear SVM is enough.
    2. if not use, the SVM kernel should be poly or rbf.
    3. even use nominal2binary, doesn't improve the poly/rbf kernel accuracy.
    4. forget use f1_score to evaluate the result.
    5. oh. f1_score is really low.
    """
