import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 路径配置
# 请修改为你的输入数据路径 / Change to your data paths
DATA_PATH = '../outputs/健美操关节角度_完整版.csv'
SAVE_DIR = '../figures'
os.makedirs(SAVE_DIR, exist_ok=True)

# 绘图风格配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

colors = ['#486FB3', '#82B0D2', '#E9A58E', '#8DD3C7', '#FFFFB3']
angle_cols = [
    '左膝关节角度', '右膝关节角度',
    '左踝关节角度', '右踝关节角度',
    '左髋关节角度', '右髋关节角度',
    '躯干倾斜角',
    '左肘关节角度', '右肘关节角度',
    '分腿角度',
    '左身体直线度角', '右身体直线度角',
    '肩躯干对称角'
]

# 加载数据
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print("数据加载完成，共 {} 条记录".format(len(df)))

# 按动作+人员ID分组，计算评价指标
def compute_metrics(group):
    group = group.sort_values('帧号').reset_index(drop=True)
    metrics = {}

    # 缓冲幅度
    for col in angle_cols:
        if col in group.columns:
            initial = group[col].iloc[0]
            final = group[col].min()
            metrics[f'{col}_缓冲幅度'] = final - initial

    # 缓冲时长（帧差/30fps）
    metrics['缓冲时长(s)'] = (group['帧号'].max() - group['帧号'].min()) / 30

    # 角度变化速率
    if metrics['缓冲时长(s)'] > 0:
        for col in angle_cols:
            if col in group.columns:
                metrics[f'{col}_变化速率'] = metrics[f'{col}_缓冲幅度'] / metrics['缓冲时长(s)']

    # 动作对称差
    if '左膝关节角度' in group.columns and '右膝关节角度' in group.columns:
        metrics['膝关节对称差'] = abs(group['左膝关节角度'].mean() - group['右膝关节角度'].mean())
    if '左髋关节角度' in group.columns and '右髋关节角度' in group.columns:
        metrics['髋关节对称差'] = abs(group['左髋关节角度'].mean() - group['右髋关节角度'].mean())
    if '左踝关节角度' in group.columns and '右踝关节角度' in group.columns:
        metrics['踝关节对称差'] = abs(group['左踝关节角度'].mean() - group['右踝关节角度'].mean())
    if '左肘关节角度' in group.columns and '右肘关节角度' in group.columns:
        metrics['肘关节对称差'] = abs(group['左肘关节角度'].mean() - group['右肘关节角度'].mean())

    # 最大/最小角度
    for col in angle_cols:
        if col in group.columns:
            metrics[f'{col}_最大值'] = group[col].max()
            metrics[f'{col}_最小值'] = group[col].min()

    # 角度标准差（动作稳定性）
    for col in angle_cols:
        if col in group.columns:
            metrics[f'{col}_标准差'] = group[col].std()

    return pd.Series(metrics)

# 计算所有人员的评价指标
metrics_df = df.groupby(['动作', '人员ID']).apply(compute_metrics).reset_index()
metrics_df.to_csv(os.path.join(SAVE_DIR, '动作评价指标.csv'), index=False, encoding='utf-8-sig')
print("评价指标计算完成，已保存到 {}".format(os.path.join(SAVE_DIR, '动作评价指标.csv')))


# 图1：分动作绘制关节角度相关性热力图
action_list = ['团身跳', '屈体分腿跳', '纵劈腿跳']

for action in action_list:
    df_action = df[df['动作'] == action].copy()

    plt.figure(figsize=(14, 12))
    corr = df_action[angle_cols].corr()

    sns.heatmap(
        corr,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        vmin=-1, vmax=1,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8, "aspect": 30},
        square=True
    )

    plt.title(f'{action} - 关节角度 Pearson 相关性热力图', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'3_{action}_相关性热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 图2：缓冲幅度与缓冲时长散点图（团身跳膝关节示例）
if '团身跳' in metrics_df['动作'].values:
    tj_metrics = metrics_df[metrics_df['动作'] == '团身跳']
    plt.figure(figsize=(8, 5))
    plt.scatter(
        tj_metrics['缓冲时长(s)'], tj_metrics['左膝关节角度_缓冲幅度'],
        color=colors[0], alpha=0.7, s=60, edgecolors='white'
    )
    plt.title('团身跳左膝关节缓冲幅度与缓冲时长关系', fontweight='bold')
    plt.xlabel('缓冲时长 (s)')
    plt.ylabel('缓冲幅度 (deg)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '5_团身跳膝关节缓冲特征散点图.png'), bbox_inches='tight')
    plt.close()

# 图3：角度标准差对比（稳定性）
std_cols = [col for col in metrics_df.columns if '标准差' in col and col in ['左膝关节角度_标准差', '左髋关节角度_标准差', '躯干倾斜角_标准差']]
std_data = metrics_df.melt(id_vars=['动作'], value_vars=std_cols, var_name='关节', value_name='标准差')
plt.figure(figsize=(10, 6))
sns.barplot(x='关节', y='标准差', hue='动作', data=std_data, palette=colors[:3])
plt.title('不同动作关键关节角度标准差（稳定性）对比', fontweight='bold')
plt.ylabel('角度标准差 (deg)')
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '6_关节角度标准差对比图.png'), bbox_inches='tight')
plt.close()

# 图4：分腿角度密度分布
if '纵劈腿跳' in df['动作'].values:
    plt.figure(figsize=(8, 5))
    for i, act in enumerate(['团身跳', '屈体分腿跳', '纵劈腿跳']):
        if act in df['动作'].values:
            sns.kdeplot(df[df['动作'] == act]['分腿角度'].dropna(), fill=True, color=colors[i], alpha=0.5, label=act)
    plt.title('不同动作分腿角度密度分布', fontweight='bold')
    plt.xlabel('分腿角度 (deg)')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '7_分腿角度密度分布.png'), bbox_inches='tight')
    plt.close()

print(f"所有图表已生成，保存路径：{SAVE_DIR}")
