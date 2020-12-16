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

    print(pd.DataFrame(x_train).columns.values)
    print(pd.DataFrame(y_train).columns.values)

    from sklearn import svm
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    import time

    svc = make_pipeline(MinMaxScaler(), svm.SVC(kernel='linear'))
    start = time.time()
    svc.fit(x_train, y_train)
    end = time.time()
    print(end - start)
    print(svc.score(x_train, y_train))
    print(svc.score(x_val, y_val))
