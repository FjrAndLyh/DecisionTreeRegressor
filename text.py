import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 生成随机数据(伪随机，实际上是固定的)
areas = np.random.randint(low=800, high=3000, size=70)
prices = areas * 150 + np.random.normal(loc=0, scale=20000, size=70)
data = np.column_stack((areas, prices))

print(data)
# 创建决策树回归模型
from sklearn.tree import DecisionTreeRegressor
#设置深度，防止过度递归造成过拟合，过拟合对于预测并不是十分有效
#会造成对于某一种情况的极度偏向，导致预测结果大幅度偏离实际
model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
#投喂数据
model.fit(data[:, 0].reshape(-1, 1), data[:, 1])

# 绘制决策树回归可视化图形
x = np.arange(800, 3000, 1).reshape(-1, 1)
y = model.predict(x)
#用绿色点阵代表数据集
plt.scatter(data[:, 0], data[:, 1], c='green', label='training data')
#用红色线代表拟合曲线
plt.plot(x, y, c='red', label='decision tree regression')
plt.xlabel('area')
plt.ylabel('price')
plt.legend()
plt.show()
