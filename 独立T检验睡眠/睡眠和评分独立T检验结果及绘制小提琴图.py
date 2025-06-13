import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据到变量'data'中
data = pd.read_csv('D:\\生信python期末作业\\all_data.csv')

# 将睡眠时长分为两组
group1 = data[data['Sleepduration'] == 1]  # 每天平均不超过7小时
group2 = data[data['Sleepduration'] == 2]  # 每天平均超过7小时

# 执行独立样本T检验 - 抑郁评分和焦虑评分（代码同前，这里省略）

# 设置颜色调色板
palette = {"<=7 hours": "skyblue", ">7 hours": "lightgreen"}

# 绘制小提琴图 - 抑郁评分
plt.figure(figsize=(10, 6))
# 更新数据框中的'Sleepduration'列为描述性字符串
data['Sleepduration'] = data['Sleepduration'].map({1: '<=7 hours', 2: '>7 hours'})
sns.violinplot(x='Sleepduration', y='Depression_score', data=data, palette=palette)
plt.title('Violin Plot of Depression Score by Sleep Duration')
plt.xlabel('Sleep Duration')
plt.ylabel('Depression Score')
plt.savefig('depression_violin_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制小提琴图 - 焦虑评分
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sleepduration', y='Anxiety_score', data=data, palette=palette)
plt.title('Violin Plot of Anxiety Score by Sleep Duration')
plt.xlabel('Sleep Duration')
plt.ylabel('Anxiety Score')
plt.savefig('anxiety_violin_plot.png', dpi=300, bbox_inches='tight')
plt.show()
