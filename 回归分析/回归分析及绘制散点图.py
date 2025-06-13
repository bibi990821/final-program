import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = pd.read_csv('D:\\生信python期末作业\\all_data.csv')  # CSV文件路径

# 将数值型变量转换为分类变量
data['Gender'] = data['Gender'].map({1: 'Male', 2: 'Female'})
data['Age_group'] = data['Age_group'].map({1: '≤14 years', 2: '15-18 years'})
data['Family_income'] = data['Family_income'].map({
    1: 'Very Rich',
    2: 'Upper Middle',
    3: 'Middle',
    4: 'Lower Middle',
    5: 'Poor'
})
data['Parents_relationship'] = data['Parents_relationship'].map({
    1: 'Very Harmonious',
    2: 'Harmonious',
    3: 'Average',
    4: 'Not Harmonious',
    5: 'Very Not Harmonious'
})
data['Sleepduration'] = data['Sleepduration'].map({1: '≤7 hours', 2: '>7 hours'})
data['Depression_risk'] = data['Depression_risk'].map({1: 'High Risk', 2: 'Low Risk'})
data['Anxiety_risk'] = data['Anxiety_risk'].map({1: 'High Risk', 2: 'Low Risk'})

# 显示前几行数据以确认转换是否正确
print(data.head())

# 准备特征和目标变量
X_depression = pd.get_dummies(data.drop(columns=['Depression_score', 'Anxiety_score']), drop_first=True)
y_depression = data['Depression_score']

# 划分训练集和测试集
X_train_depression, X_test_depression, y_train_depression, y_test_depression = train_test_split(X_depression, y_depression, test_size=0.2, random_state=42)

# 训练模型
model_depression = LinearRegression()
model_depression.fit(X_train_depression, y_train_depression)

# 预测
y_pred_depression = model_depression.predict(X_test_depression)

# 评估模型
mse_depression = mean_squared_error(y_test_depression, y_pred_depression)
r2_depression = r2_score(y_test_depression, y_pred_depression)

print(f"Depression Score Regression Results:")
print(f"Mean Squared Error: {mse_depression:.2f}")
print(f"R^2 Score: {r2_depression:.2f}")

# 输出系数
coefficients_depression = pd.DataFrame({
    'Feature': X_train_depression.columns,
    'Coefficient': model_depression.coef_
})
print("\nCoefficients for Depression Score:")
print(coefficients_depression)

# 准备特征和目标变量
X_anxiety = pd.get_dummies(data.drop(columns=['Depression_score', 'Anxiety_score']), drop_first=True)
y_anxiety = data['Anxiety_score']

# 划分训练集和测试集
X_train_anxiety, X_test_anxiety, y_train_anxiety, y_test_anxiety = train_test_split(X_anxiety, y_anxiety, test_size=0.2, random_state=42)

# 训练模型
model_anxiety = LinearRegression()
model_anxiety.fit(X_train_anxiety, y_train_anxiety)

# 预测
y_pred_anxiety = model_anxiety.predict(X_test_anxiety)

# 评估模型
mse_anxiety = mean_squared_error(y_test_anxiety, y_pred_anxiety)
r2_anxiety = r2_score(y_test_anxiety, y_pred_anxiety)

print(f"\nAnxiety Score Regression Results:")
print(f"Mean Squared Error: {mse_anxiety:.2f}")
print(f"R^2 Score: {r2_anxiety:.2f}")

# 输出系数
coefficients_anxiety = pd.DataFrame({
    'Feature': X_train_anxiety.columns,
    'Coefficient': model_anxiety.coef_
})
print("\nCoefficients for Anxiety Score:")
print(coefficients_anxiety)

import csv

# 准备输出数据
output_depression = {
    'Feature': X_train_depression.columns,
    'Coefficient': model_depression.coef_
}
output_df_depression = pd.DataFrame(output_depression)
output_df_depression.loc[len(output_df_depression)] = ['Intercept', model_depression.intercept_]
output_df_depression.loc[len(output_df_depression)] = ['Mean Squared Error', mse_depression]
output_df_depression.loc[len(output_df_depression)] = ['R^2 Score', r2_depression]

# 将结果写入CSV文件
output_df_depression.to_csv('regression_results_depression.csv', index=False)  # 保存的文件名和路径

# 准备输出数据
output_anxiety = {
    'Feature': X_train_anxiety.columns,
    'Coefficient': model_anxiety.coef_
}
output_df_anxiety = pd.DataFrame(output_anxiety)
output_df_anxiety.loc[len(output_df_anxiety)] = ['Intercept', model_anxiety.intercept_]
output_df_anxiety.loc[len(output_df_anxiety)] = ['Mean Squared Error', mse_anxiety]
output_df_anxiety.loc[len(output_df_anxiety)] = ['R^2 Score', r2_anxiety]

# 将结果写入CSV文件
output_df_anxiety.to_csv('regression_results_anxiety.csv', index=False)  # 保存的文件名和路径

# 抑郁评分与各特征的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Depression_score', y='Anxiety_score', hue='Gender', data=data)
plt.title('Scatter Plot of Depression Score vs Anxiety Score by Gender')
# 保存图像
plt.savefig('scatter_plot_depression_anxiety_gender.png')  # 保存的文件名和路径
plt.show()

# 焦虑评分与各特征的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Anxiety_score', y='Depression_score', hue='Age_group', data=data)
plt.title('Scatter Plot of Anxiety Score vs Depression Score by Age Group')
# 保存图像
plt.savefig('scatter_plot_depression_anxiety_age.png')  # 保存的文件名和路径
plt.show()
