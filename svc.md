<h1><center>svm 说明文档

[TOC]

# 包依赖

python3.8, numpy, pandas, json, argparse

# 运行方法

## 帮助

```shell
python svc.py 
```

## 示例

### 训练

指定参数进行训练

```shell
python svc.py --train --train_set ./dataset/线下/svm/svm_training_set.csv --train_rate 0.9 --sample_num 2000 --C 1 --epochs 100000 --save_model checkpoint.json
```

使用默认参数进行训练

```shell
python svc.py --train --train_set ./dataset/线下/svm/svm_training_set.csv --save_model checkpoint.json
```

### 测试

```shell
python svc.py --test --test_set ./dataset/线下/svm/svm_training_set.csv --model_path ./svc_checkpoint.json --result_path RESULT.csv
```

# 代码说明

代码整体包括3个类: linearSVC类, Prerocess类, Model类.

1. LinearSVC 实现软间隔线性支持向量机. 是一个通用的,数据集无关的模型. 模仿sklearn的接口风格.
2. PreProcess 实现对数据的预处理,包括将标称类型转化为多个binary变量,变量归一化. 是一个数据集相关的类,根据数据集的特征来进行一些预处理工作.当更换数据集时,或需要指定新的参数,甚至重写.
3. Model 将PreProcess 和 LinearSVC 整合到一起. 保证同一次训练中的PreProcess对象和LinearSVC对象被同时保存,加载.避免混乱搭配.

模型的保存和加载为保证跨平台,跨python版本的可用性,均使用python内置类型结合json文件的方法.

另外为了代码结果更容易被复现与使用,实现了train函数,test函数,以及用于评估的f1_score函数. 具体而言, 可以参考示例使用命令行参数. 或在其他文件中import并调用.

main函数主要负责命令行参数的解析和命令分发.

## LinearSVC类

主要功能: 实现软间隔线性支持向量机. 使用SMO算法来进行模型训练. 此外还实现预测,保存,加载等功能.

主要函数:

| 函数                                        | 参数要求                                                     | 返回值              | 功能                                                         |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ |
| \__init__(c=1, epsilon=1e-3, verbose=False) | c: int 软间隔参数<br />epsilon: float 误差阈值. 误差小于该值时停止训练<br />verbose: bool 打印详细训练过程. | 无                  | 构造函数                                                     |
| state_dict()                                | 无                                                           | dict                | 返回描述模型的所有信息的状态字典<br />且其中所有值类型可以被json进行dumps |
| load(state_dict)                            | state_dict: dict                                             | LinearSVC对象       | 根据状态字典生成对应LinearSVC对象                            |
| fit(X, y, epochs=10 ** 4)                   | X: array-like 特征序列<br />y: array-like 标记序列<br />epochs:  int 最大迭代次数 | 无                  | 根据输入训练模型                                             |
| predict(self, X, return_raw=False)          | X: array-like 特征序列<br />return_raw: bool                 | array-like 标记序列 | 若return_raw为true, 返回线性函数计算结果.否则返回类别.       |

辅助计算函数:

| 函数                                                         | 参数要求                                                     | 返回值 | 功能                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| _fit_init(X, y)                                              | X: array-like 特征序列<br />y: array-like 标记序列           | 无     | 根据训练数据初始化相应训练参数                               |
| _pick_first_alpha()                                          | 无                                                           | int    | 返回"最"违背KKT条件的对偶变量alpha索引                       |
| _pick_second_alpha(idx1)                                     | idx1: int 选定的第一个alpha                                  | int    | 根据第一个选定的alpha来确定第二个alpha变量                   |
| _update_alpha(self, idx1, idx2)                              | idx1: int 选定的第一个alpha<br />idx2: int 选定的第二个alpha | 无     | 根据所选择的对偶变量alpha, 计算并更新**所有参数以及各种缓存** |
| _update_b_cache(E1, E2, idx1, idx2, alpha_old_1, alpha_old_2) | idx1: int 选定的第一个alpha<br />idx2: int 选定的第二个alpha<br />alpha_old_1: float 第一个alpha更新前的值<br />alpha_old_2: float 第二个alpha更新前的值<br />E1,E2: float $E_{idx_1},E_{idx_2}$具体含义参见注 | 无     | 更新分类器偏差b                                              |
| _update_w_cache(idx1, idx2)                                  | idx1: int 选定的第一个alpha<br />idx2: int 选定的第二个alpha | 无     | 若将g(x)写为w*x+b的形式.更新其中的w缓存.                     |
| _update_predict_cache()                                      | 无                                                           | 无     | 更新g(x)对训练集特征X的预测结果缓存                          |

注: $E_i$为函数$g(x) $对输入$x_i$的预测值与真实输出$ y_i $之差。其中$g(x)=\sum_1^N{\alpha_iy_iK(x_i,x)+b}$.$N$为样本点个数. (具体参见<<统计学习方法第2版>> 7.4.1 两个变量二次规划的求解方法 P145.)

## PreProcess类

主要功能: 对原始DataFrame数据进行预处理. 具体为:

1. 将多值标称类型转化为多个one-hot的二元变量.
2. 根据训练样本分布特点,对训练样本以及测试样本进行归一化.具体实现是对二元变量,序数变量使用最大最小归一化.对比率类型变量采用均值方差标准化.

主要函数:

| 函数                        | 参数要求                         | 返回值         | 功能                                                         |
| --------------------------- | -------------------------------- | -------------- | ------------------------------------------------------------ |
| state_dict()                | 无                               | dict           | 返回描述模型的所有信息的状态字典<br />且其中所有值类型可以被json进行dumps |
| load(state_dict)            | state_dict: dict                 | PreProcess对象 | 根据状态字典生成对应PreProcess对象                           |
| \__init__(nominal)          | nominal: list 标称类型数据       | 无             | 构造函数.                                                    |
| nominal2binary(data)        | data: pd.DataFrame 样本集        | pd.DataFrame   | 将多值标称类型转化为多个one-hot的二元变量                    |
| normalization_init(x_train) | x_train: pd.DataFrame 训练样本集 | 无             | 记录训练样本集的某些分布特点. 用于之后对数据的归一化         |
| normalize(x)                | x: pd.DataFrame                  | pd.DataFrame   | 对数据进行归一化                                             |

## Model类

主要功能:

1. 将PreProcess对象和LinearSVC对象进行pipeline. 保证PreProcess对象和**对应的**LinearSVC对象同时被使用,保存.
2. 对外提供统一接口,可以直接处理**原始**的pd.DataFrame数据.

主要函数:

| 函数                               | 参数要求                                                     | 返回值            | 功能                                                         |
| ---------------------------------- | ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ |
| state_dict()                       | 无                                                           | dict              | 返回描述模型的所有信息的状态字典<br />且其中所有值类型可以被json进行dumps |
| load(state_dict)                   | state_dict: dict                                             | model对象         | 根据状态字典生成对应model对象                                |
| fit(X, y, epochs=10 ** 5)          | X: pd.DataFrame 特征序列<br />y: pd.DataFrame 标记序列<br />epochs:  int 最大迭代次数 | 无                | 根据输入确定预处理参数,训练模型                              |
| predict(self, X, return_raw=False) | X: array-like 特征序列<br />return_raw: bool                 | np.array 标记序列 | 若return_raw为true, 返回线性函数计算结果.否则返回类别.       |

## 其他

| 函数                                                         | 参数要求                                                     | 返回值                 | 功能                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- | --------------------------------------------------- |
| f1_score(y_true, y_pred, pos_label=1, labels=None)           | y_true: array-like<br />y_pred: array-like<br />pos_label: 正类标签. default: 1<br />labels: 所有类别标签. default: 从y_true中推断 | float: f1得分          | 计算正类的f1得分                                    |
| train(train_set, train_rate, sample_num, C,epochs,save_model, verbose) | train_set: 训练集csv文件路径<br />train_rate: float训练数据和val数据划分比例.<br />sample_num: int 从训练数据中采样数目用于训练<br />C: int 软间隔参数C<br />epochs: int 最大迭代次数<br />save_model: 保存训练模型的路径<br />verbose: int 打印详细训练信息 | Model对象              | 训练模型                                            |
| test(test_set, model_path, result_path)                      | test_set: 测试集csv文件路径<br />model_path: 保存的模型参数路径<br />result_path: 目标保存预测结果的路径 | pd.DataFrame: 预测结果 | 将预测结果添加到文件最后一列,并保存到result_path处. |

train函数细节说明:

1. train_rate指明训练集合占总数据集的比例,其他数据作为验证集.

一些问题:

1. 训练集过大,个人实现版本中由于内存限制无法一次性利用所有样本.原因在于需要cache核函数计算结果.大小是sizeof(float)\*data_num\*data_num.对于2w+条样本,需要一次性申请大约4G+的内存.观察sklearn版本的svc训练过程中并没有申请如此巨大的空间.更优秀的cache算法或能够缓解空间占用高的问题.但实现起来代码也会更加复杂.鉴于cache并不是本次作业的重点,所以没有花费大精力去实现.
2. 过多样本时,个人实现SMO版本收敛缓慢.原因可能在于 1.实现中的第二个alpha的启发式选择还不够优秀.仅使用了随机选择.显然还不够好. 2.收敛的判别准则过于严苛,仅使用违反KKT条件的程度作为判别条件.且训练后期大部分情况违反程度在0.03附近,而低于0.0001则很困难,,或许数据量较大时违反程度很难低于0.0001.或许可以引入更多的更优秀的收敛判别准则.
3. 但幸运的是,在该问题的训练集上进行部分采样和使用原数据集的效果相当.且能够加速训练时间.train函数的sample_num参数指明最终sample的小训练集的大小.

