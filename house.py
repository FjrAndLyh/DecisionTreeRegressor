import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 读取数据集(数据集在本地)
dataset = pd.read_csv('house_data.csv')

# 将数据集拆分成自变量和因变量
X = dataset.iloc[:, 0].values.reshape(-1, 1) # 年份
y = dataset.iloc[:, 1].values.reshape(-1, 1) # 房价

# 拟合决策树回归模型
#寻找合适的深度，避免深度过高递归次数过多出现过拟合
regressor = DecisionTreeRegressor(random_state=0, max_depth=3)
regressor.fit(X, y)

# 预测房价趋势
X_test = np.arange(min(X), max(X)+1, 1).reshape(-1, 1) # 生成预测数据集
y_pred = regressor.predict(X_test)

# 可视化预测结果
plt.scatter(X, y, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Prediction')
plt.xlabel('years')
plt.ylabel('prices')
plt.show()