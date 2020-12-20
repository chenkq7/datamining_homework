<h1><center> Random Forest 说明文档

[TOC]

# 包依赖

python3.8, pandas, numpy, json, argparse

# 运行方法

## 帮助

```shell
python random_forest.py
```

## 示例

### 训练

指定参数进行训练

```shell
python random_forests.py --train --x_path ./dataset/线下/rf/x_train.csv --y_path ./dataset/线下/rf/y_train.csv --train_rate 0.9 --tree_num 100 --attr_num 5 --save_model model_checkpoint.json
```

使用默认参数进行训练

```shell
python random_forests.py --train --x_path ./dataset/线下/rf/x_train.csv --y_path ./dataset/线下/rf/y_train.csv
```

### 测试

```shell
python random_forests.py --test --data_path ./dataset/线下/rf/x_train.csv --model_path rf_checkpoint.json --result_path RESULT.csv
```

注: 输出文件RESULT.csv 应为列索引为 [index, pred] 的csv文件. index 为记录的 id, pred为模型的预测输出.

# 代码功能说明

## PredVoter类

主要功能: 二分类投票器.对多个分类器的结果进行记录,输出.

主要函数:

| 函数                    | 参数要求                                                     | 返回值                      | 功能                              |
| ----------------------- | ------------------------------------------------------------ | --------------------------- | --------------------------------- |
| acccumulate(ids ,preds) | ids: array-like<br />preds: array-like. value in {-1, +1}<br />preds 和 ids 长度相同. pred[k] 是对 id[k] 的预测结果 | 无                          | 记录此组预测结果                  |
| vote()                  | 无                                                           | type=np.array. (ids, preds) | 多数表决的投票结果,平手则随机输出 |
| clear()                 | 无                                                           | 无                          | 清空统计结果                      |

## TreeNode类

主要功能: 决策树节点类. 负责决策树递归的构建,预测.

主要函数:

| 函数                             | 参数要求                                                     | 返回值               | 功能                                                         |
| -------------------------------- | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| state_dict()                     | 无                                                           | type=dict.           | 返回描述该节点及其子节点的所有信息的字典.<br />且其中所有变量均可以使用json进行dumps.<br />便于节点对象可移植的序列化. |
| load(state_dict)                 | state_dict 描述该节点的所有信息的字典                        | TreeNode对象         | 根据状态字典生成节点对象.                                    |
| build_tree(X,y,features,epsilon) | X: array-like. 特征序列<br />y: array-like. label序列<br />features: set. X中哪些列还可以用来分类<br />epsilon:float. 信息增益小于该阈值时作为叶节点. | TreeNode对象         | 使用C4.5算法递归地建立决策树.无剪枝.                         |
| predict(X)                       | X: 特征序列                                                  | np.array. label 序列 | 使用以该节点为根结点的子树进行预测.                          |

计算辅助函数:

| 函数                                     | 参数要求                                                     | 返回值 | 功能                                             |
| ---------------------------------------- | ------------------------------------------------------------ | ------ | ------------------------------------------------ |
| calculate_gain_rate(X, y, feature)       | X: array-like. 特征序列<br />y: array-like. label序列<br />feature: int. 选择的特征<br /> | float  | 在选择feature作为分类条件下,得到的信息增益比     |
| calculate_condition_entropy(X,y,feature) | X: array-like. 特征序列<br />y: array-like. label序列<br />feature: int. 选择的特征<br /> | float  | 在选择feature作为分类条件下,得到的条件经验信息熵 |
| calculate_entropy(y)                     | y: array-like. label序列                                     | float  | 分布y的经验信息熵                                |
| calculate_probability(y)                 | y: array-like. label序列                                     | dict   | 仅以该节点做预测时,各个类别的概率.               |

## DecisionTree类

主要功能: 决策树类, 对TreeNode类的使用,以及对接口的简单包装. 仅暴露用户关心的函数.

主要函数:

| 函数               | 参数要求                                                     | 返回值               | 功能                                                         |
| ------------------ | ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| state_dict()       | 无                                                           | type=dict.           | 返回描述决策树所有信息的字典.<br />且其中所有变量均可以使用json进行dumps.<br />便于节点对象可移植的序列化. |
| load(state_dict)   | state_dict 描述该节点的所有信息的字典                        | TreeNode对象         | 根据状态字典生成节点对象.                                    |
| fit(X,y,epsilon=0) | X: array-like. 特征序列<br />y: array-like. label序列<br />psilon:float. 建树时信息增益率小于该阈值时作为叶节点.默认值为0 | TreeNode对象         | 构建决策树.                                                  |
| predict(X)         | X: array-like. 特征序列                                      | np.array. label 序列 | 根据输入特征进行预测.                                        |
| score(X,y)         | X: array-like. 特征序列<br />y: array-like. label序列        | float                | 准确率.                                                      |

## RandomForest类

主要功能: 随机森林类.

主要函数:

| 函数                           | 参数要求                                                     | 返回值                         | 功能                                                         |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------ |
| state_dict()                   | 无                                                           | type=dict.                     | 返回描述随机森林所有信息的字典.<br />且其中所有变量均可以使用json进行dumps.<br />便于节点对象可移植的序列化. |
| load(state_dict)               | state_dict 描述该节点的所有信息的字典                        | RandomForest对象               | 根据状态字典生成对象.                                        |
| \__init__(tree\_num,attr\_num) | tree_num: int 森林中树的总数<br />attr_num: 每个树使用的属性的数量 | 无                             | 构造函数                                                     |
| fit(x_data,y_data)             | x_data: pd.DataFrame 列索引形如(id,attr_1,...attr_n)<br />y_data: pd.DataFrame 列索引形如(id,label) | RandomForest对象               | 构建森林                                                     |
| predict(x_data)                | x_data: pd.DataFrame (id,attr_1,...attr_n)                   | np.array 每行记录形如(id,pred) | 返回预测结果                                                 |
| f1_score(x_data,y_data)        | 同上                                                         | float                          | 返回正类的f1得分                                             |
| f1_score_on_oob(y_data)        | y_data: pd.DataFrame 列索引形如(id,label). 所有训练时使用过的样本的label | float                          | 返回训练时out of bag集合的平均f1                             |

辅助计算函数:

| 函数                                               | 参数要求                                                     | 返回值         | 功能                                                   |
| -------------------------------------------------- | ------------------------------------------------------------ | -------------- | ------------------------------------------------------ |
| sample(x_pd, y_pd, x_num=None, attr_num=None)      | x_pd: pd.DataFrame (id,attr_1,...,attr_n).<br />y_pd: pd.DataFrame (id,label).<br />x_num: the sample num for data. default: len(x_pd)<br />attr_num: the sample num for attrs. default: sqrt(attrs_num) | 元组: 采样结果 | 在原数据集中对记录的有放回采样.以及对特征的无放回采样. |
| f1_score(y_true, y_pred, pos_label=1, labels=None) | y_true: array-like<br />y_pred: array-like<br />pos_label: 正类标签. default: 1<br />labels: 所有类别标签. default: 从y_true中推断 | float: f1得分  | 计算正类的f1得分                                       |

## 其他函数

| 函数                                                         | 参数要求                                                     | 返回值           | 功能                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- | ----------------------------------------- |
| save_state_dict(state_dict, filename)                        | state_dict: 字典<br />filename: 文件名                       | 无               | 将状态字典序列化为json文件                |
| load_state_dict(filename)                                    | filename: 文件名                                             | state_dict: 字典 | 将json文件加载为状态字典                  |
| train(x_data_path, y_data_path, train_data_rate,tree_num, attr_num, save_model) | x_data_path: 特征文件路径<br /> y_data_path: 标记文件路径<br />train_data_rate: 训练集在全体所占比例 (其他为验证集)<br />tree_num: 随机森林中树的总数<br />attr_num: 随机森林中每颗树使用的特征数<br />save_model: 保存模型的路径 | RandomForest对象 | 训练随机森林                              |
| test(csv_data_path, model_state_dict, result_path)           | csv_data_path: 特征文件路径<br />model_state: 模型状态序列化文件路径<br />result_path: 结果输出的csv文件地址. 结果列索引形如 (id,pred) | pd:DataFrame     | 使用保存的模型对测试集进行测试,并输出结果 |

