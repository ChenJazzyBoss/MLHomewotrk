import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

# pd.set_option('max_columns', None)
pd.options.display.max_columns = None
# pd.set_option('max_rows', 200)
pd.options.display.max_rows = 200
# pd.set_option('float_format', lambda x: '%.3f' % x)
pd.options.display.float_format = lambda x: '%.3f' % x

train = pd.read_csv('data/data112151/train_dataset.csv', sep='\t')
print("在训练集中，共有{}条数据，其中每条数据有{}个特征".format(train.shape[0], train.shape[1]))

test = pd.read_csv('data/data112151/test_dataset.csv', sep='\t')
print("在测试集中，共有{}条数据，其中每条数据有{}个特征".format(test.shape[0], test.shape[1]))

# 同时处理训练数据与测试数据
# 将两个数据集连起来
data = pd.concat([train, test])
# print(data.shape)

# 将三段式的地址分成三个数据
data['location_first_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
data['location_sec_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
data['location_third_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

# 客户端类型和浏览器来源对检测没有实际意义，故删除
data.drop(['client_type', 'browser_source'], axis=1, inplace=True)

# 对首次认证方式为空的数据进行填充
# 即对数据表中没有数据的部分填空,为后面编码做准备
data['auth_type'].fillna('__NaN__', inplace=True)

# 将数据编码
for col in tqdm(['user_name', 'action', 'auth_type', 'ip',
                 'ip_location_type_keyword', 'ip_risk_level', 'location', 'device_model',
                 'os_type', 'os_version', 'browser_type', 'browser_version',
                 'bus_system_code', 'op_target', 'location_first_lvl', 'location_sec_lvl',
                 'location_third_lvl']):
    # 对每列进行编码
    lbl = LabelEncoder()
    data[col] = lbl.fit_transform(data[col])

# 处理认证时间
data['op_date'] = pd.to_datetime(data['op_date'])
# 将时间换成毫秒
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9

# 重排数据
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
# 新增数据列 列值为data.groupby(['user_name'])['op_ts']向下挪动一位
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
# ts_diff1列的值为操作时间减去结束时间
data['ts_diff1'] = data['op_ts'] - data['last_ts']

# 通过对各列的遍历，获取数据中不同的值
for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
    data[f'user_{f}_nunique'] = data.groupby(['user_name'])[f].transform('nunique')

# 双重循环遍历获取不同的方法（'mean', 'max', 'min', 'std', 'sum', 'median'）值
for method in ['mean', 'max', 'min', 'std', 'sum', 'median']:
    for col in ['user_name', 'ip', 'location', 'device_model', 'os_version', 'browser_version']:
        data[f'ts_diff1_{method}_' + str(col)] = data.groupby(col)['ts_diff1'].transform(method)
        # 查看数值
        print(data[f'ts_diff1_{method}_' + str(col)])

# 根据risk_label字段将训练集与测试集分开
train = data[data['risk_label'].notna()]
test = data[data['risk_label'].isna()]

# print(train.shape, test.shape)
ycol = 'risk_label'

# filter函数能够从可迭代对象（如字典、列表）中筛选某些元素，并生成一个新的迭代器 并用list()函数将其转化为列表，这个列表包含过滤器对象中返回的所有的项。
# 过滤
feature_names = list(
    filter(lambda x: x not in [ycol, 'session_id', 'op_date', 'last_ts'], train.columns))   # 过滤掉已经使用过的特征

# 设置模型参数
model = lgb.LGBMClassifier(objective='binary',  # 目标为二进制
                           boosting_type='gbdt',    # 树的类型为梯度提升树gbdt
                           tree_learner='serial',   # 单个machine tree 学习器
                           num_leaves=2 ** 8,   # 树的最大叶子数为256
                           max_depth=16,    # 树的最大深度为16
                           learning_rate=0.1,   # 学习率为0.1
                           n_estimators=10000,  # 拟合的树的棵树，相当于训练轮数，这个为10000轮
                           subsample=0.5,   # 训练样本采样率
                           feature_fraction=0.4,    # 使用特征的子抽样
                           reg_alpha=0.,    # L1正则化系数
                           reg_lambda=0.,   # L2正则化系数
                           random_state=2021,   # 随机种子数
                           is_unbalance=True,   # 算法将尝试自动平衡占主导地位的标签的权重
                           metric='auc')    # 模型度量标准

# 初始化
oof = []    # 折外预测数组
prediction = test[['session_id']]
prediction[ycol] = 0
df_importance_list = []     # 特征重要性

# KFold是用于生成交叉验证的数据集的，而StratifiedKFold则是在KFold的基础上，
# 加入了分层抽样的思想，使得测试集和训练集有相同的数据分布，因此表现在算法上，
# StratifiedKFold需要同时输入数据和标签，便于统一训练集和测试集的分布
# 将数据分成5份 打乱顺序 随机数种子个数为2022
# StratifiedKFold能确保训练集，测试集中各类别样本的比例与原始数据集中相同。
# 5折遍
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

# 数据集进行划分，并获取索引值，然后进行遍历
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]
    print(X_train)
    print("------------------------------")

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]
    # 从一开始
    print('Fold_{} Training'.format(fold_id + 1))
    # 开始训练
    lgb_model = model.fit(X_train,  # array, DataFrame 类型
                          Y_train,  # array, Series 类型
                          eval_names=['train', 'valid'],    # 数据集名字
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],    # 用于评估的训练集及验证集
                          verbose=100,
                          eval_metric='auc',    # 评估函数，二分类
                          early_stopping_rounds=5)  # 在5轮间指标不提升，那就提前停止！！！
    # 开始预测
    # 预测验证集
    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)     # 预测值选取最好的那次
    df_oof = train.iloc[val_idx][['session_id', ycol]].copy()   # 保存每一次预测验证集的结果
    df_oof['pred'] = pred_val[:, 1]
    oof.append(df_oof)
    # 预测测试集
    pred_test = lgb_model.predict_proba(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    # 保存预测结果
    prediction[ycol] += pred_test[:, 1] / kfold.n_splits
    # 创建表两列
    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,       #
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
    gc.collect()

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()  # 聚合，整体求均值

# print(df_importance)
df_importance
# 处理结果
df_oof = pd.concat(oof)
# 评估
print('roc_auc_score', roc_auc_score(df_oof[ycol], df_oof['pred']))

prediction['id'] = range(len(prediction))
prediction['id'] = prediction['id'] + 1
prediction = prediction[['id', 'risk_label']].copy()
prediction.columns = ['id', 'ret']
prediction.head()
prediction.to_csv('submit.csv', index=False)
