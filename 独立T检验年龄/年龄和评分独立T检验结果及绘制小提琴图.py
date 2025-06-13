import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 加载数据
data = pd.read_csv('D:\\生信python期末作业\\all_data.csv')  

# 分组
group_young_depression = data[data['Age_group'] == 1]['Depression_score']
group_old_depression = data[data['Age_group'] == 2]['Depression_score']

group_young_anxiety = data[data['Age_group'] == 1]['Anxiety_score']
group_old_anxiety = data[data['Age_group'] == 2]['Anxiety_score']

# 执行独立样本T检验
t_dep, p_dep = ttest_ind(group_young_depression, group_old_depression, nan_policy='omit')
t_anx, p_anx = ttest_ind(group_young_anxiety, group_old_anxiety, nan_policy='omit')

# 构建结果表格
results = pd.DataFrame({
    'Variable': ['Depression Score', 'Anxiety Score'],
    'Group1 (≤14 years) Mean': [round(group_young_depression.mean(), 2), round(group_young_anxiety.mean(), 2)],
    'Group2 (15-18 years) Mean': [round(group_old_depression.mean(), 2), round(group_old_anxiety.mean(), 2)],
    't-statistic': [round(t_dep, 2), round(t_anx, 2)],
    'p-value': [round(p_dep, 4), round(p_anx, 4)]
})

# 输出并保存结果到当前目录
print(results)
results.to_csv(os.path.join(os.getcwd(), 'age_group_t_test_results.csv'), index=False)
print("\n✅ T检验结果已保存为 'age_group_t_test_results.csv'")


# 设置颜色调色板
palette = {'1': "#FF7F0E", '2': "#1F77B4"}

# 抑郁评分图
plt.figure(figsize=(8, 6))
sns.violinplot(x='Age_group', y='Depression_score', data=data, palette=palette)
plt.title('Violin Plot of Depression Score by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Depression Score')
save_path = os.path.join(os.getcwd(), 'depression_violin_plot_age.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
if os.path.exists(save_path):
    print(f"抑郁评分图像已成功保存至: {save_path}")
else:
    print("抑郁评分图像保存失败，请检查路径及权限。")

# 焦虑评分图
plt.figure(figsize=(8, 6))
sns.violinplot(x='Age_group', y='Anxiety_score', data=data, palette=palette)
plt.title('Violin Plot of Anxiety Score by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Anxiety Score')
save_path = os.path.join(os.getcwd(), 'anxiety_violin_plot_age.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
if os.path.exists(save_path):
    print(f"焦虑评分图像已成功保存至: {save_path}")
else:
    print("焦虑评分图像保存失败，请检查路径及权限。")
