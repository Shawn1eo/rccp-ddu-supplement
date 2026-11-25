# import copy
#
# import pandas as pd
#
# # Load the data from the provided CSV file
# df_ori = pd.read_excel('CaseStudy.xlsx')
#
# df = copy.deepcopy(df_ori)
#
# # Remove rows where originalObj is 0
# df_filtered = df[df['originalObj'] != 0].copy()
#
# # Calculate the ratio of kNNobj to originalObj
# df_filtered['ratio'] = df_filtered['kNNobj'] / df_filtered['originalObj']
#
# # Group by 'v' and 'alpha' and calculate the mean of the ratio, originRate, and kNNrate
# result = df_filtered.groupby(['v', 'alpha']).agg({
#     'ratio': 'mean',
#     'originRate': 'mean',
#     'kNNrate': 'mean'
# }).reset_index()
import copy

import pandas as pd
import matplotlib.pyplot as plt

# 计算比值
df_ori = pd.read_excel('CaseStudy.xlsx')

df = copy.deepcopy(df_ori)

# Remove rows where originalObj is 0
df_filtered = df[df['originalObj'] != 0].copy()

df_10 = df_filtered[df_filtered['v'] == 10].copy()

# Calculate the ratio of kNNobj to originalObj
df_10['ratio'] = df_10['kNNobj'] / df_10['originalObj']
mean_ratio = df_10['ratio'].mean()


# 绘制直方图
plt.figure(figsize=(12, 8))
plt.hist(df_10['ratio'], bins=40, alpha=0.7, color='lightblue', edgecolor='black')

# 添加平均比值线
plt.axvline(x=mean_ratio, color='g', linestyle='--', linewidth=2, label=f'Mean Ratio={mean_ratio:.2f}')
plt.axvline(x=1, color='r', linestyle='--', linewidth=2, label=f'ratio = 1 (Base line)')
# 设置字体大小
plt.xlabel('Final-to-Original loss ratio', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xlim(0.7, 5)
plt.title('Histogram of Final-to-Original Lost Ratios', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

plt.show()