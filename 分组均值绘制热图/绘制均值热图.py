import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('D:\\生信python期末作业\\all_data.csv')

# 定义映射字典
family_income_mapping = {
    1: '很富裕',
    2: '中等偏上',
    3: '中等',
    4: '中等偏下',
    5: '比较困难'
}

parents_relationship_mapping = {
    1: '非常和睦',
    2: '比较和睦',
    3: '一般',
    4: '不太和睦',
    5: '很不和睦'
}

# 应用映射
df['Family_income'] = df['Family_income'].map(family_income_mapping)
df['Parents_relationship'] = df['Parents_relationship'].map(parents_relationship_mapping)

# 计算抑郁筛查评分的分组均值
depression_mean = df.pivot_table(values='Depression_score', index='Parents_relationship', columns='Family_income', aggfunc='mean')

# 保存抑郁筛查评分的分组均值
depression_mean.to_csv('depression_mean.csv')

# 计算焦虑评分的分组均值
anxiety_mean = df.pivot_table(values='Anxiety_score', index='Parents_relationship', columns='Family_income', aggfunc='mean')

# 保存焦虑评分的分组均值
anxiety_mean.to_csv('anxiety_mean.csv')

# 绘制并保存抑郁筛查评分的热图
plt.figure(figsize=(10, 8))
sns.heatmap(depression_mean, annot=True, cmap='Blues', fmt='.2f')
plt.title('Depression Score Mean by Family Income and Parents Relationship')
plt.xlabel('Family Income')
plt.ylabel('Parents Relationship')
plt.savefig('depression_heatmap.png')  # 保存抑郁筛查评分的热图
plt.show()

# 绘制并保存焦虑评分的热图
plt.figure(figsize=(10, 8))
sns.heatmap(anxiety_mean, annot=True, cmap='YlOrBr', fmt='.2f')
plt.title('Anxiety Score Mean by Family Income and Parents Relationship')
plt.xlabel('Family Income')
plt.ylabel('Parents Relationship')
plt.savefig('anxiety_heatmap.png')  # 保存焦虑评分的热图
plt.show()
