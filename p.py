import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# ==================== 配置 ====================
train_folder = "amp_hacking"  # 训练奖励模型结果
indep_folder = "amp_toxinpred3"  # 独立测试模型结果
output_file = "spearman_results.csv"


# ==================== 辅助函数：自动识别列名 ====================
def find_id_and_score_columns(df):
    """自动识别ID列和毒性得分列"""
    id_candidates = ['Entry', 'entry', 'ID', 'Id', 'id', 'Peptide', 'peptide', 'Sequence', 'sequence']
    score_candidates = ['Toxin Probabilities', 'ToxinProbability', 'Toxin_Prob', 'toxin_prob',
                        'Probability', 'Score', 'score', 'Toxicity', 'toxicity']

    id_col = None
    score_col = None

    for col in df.columns:
        col_clean = col.strip()
        if col_clean in id_candidates or col_clean.lower() in [c.lower() for c in id_candidates]:
            id_col = col
        if col_clean in score_candidates or col_clean.lower() in [c.lower() for c in score_candidates]:
            score_col = col

    # 如果没找到，使用第一列作为ID，最后一列作为得分
    if id_col is None:
        id_col = df.columns[0]
        print(f"  未识别ID列，使用: {id_col}")
    if score_col is None:
        score_col = df.columns[-1]
        print(f"  未识别得分列，使用: {score_col}")

    return id_col, score_col


# ==================== 主程序 ====================
results = []

for filename in os.listdir(train_folder):
    if not filename.endswith('.csv'):
        continue

    train_path = os.path.join(train_folder, filename)
    indep_path = os.path.join(indep_folder, filename)

    if not os.path.exists(indep_path):
        print(f"警告: {filename} 在独立测试文件夹中不存在，已跳过")
        continue

    # 读取数据
    train_df = pd.read_csv(train_path)
    indep_df = pd.read_csv(indep_path)

    # 获取生成模型名称（去掉.csv后缀）
    model_name = filename.replace('.csv', '')

    print(f"\n处理: {model_name}")

    # 识别列名
    train_id_col, train_score_col = find_id_and_score_columns(train_df)
    indep_id_col, indep_score_col = find_id_and_score_columns(indep_df)

    # 重命名以便合并
    train_df = train_df.rename(columns={train_id_col: 'ID', train_score_col: 'Score_train'})
    indep_df = indep_df.rename(columns={indep_id_col: 'ID', indep_score_col: 'Score_indep'})

    # ===== 关键修复：将 ID 列统一转换为字符串类型 =====
    train_df['ID'] = train_df['ID'].astype(str)
    indep_df['ID'] = indep_df['ID'].astype(str)

    # 按ID合并
    merged = pd.merge(train_df[['ID', 'Score_train']],
                      indep_df[['ID', 'Score_indep']],
                      on='ID', how='inner')

    n_peptides = len(merged)

    if n_peptides < 3:
        print(f"  警告: 只有 {n_peptides} 条肽，无法计算可靠的相关性")
        spearman_corr = np.nan
        pearson_corr = np.nan
        spearman_p = np.nan
        pearson_p = np.nan
    else:
        spearman_corr, spearman_p = spearmanr(merged['Score_train'], merged['Score_indep'])
        pearson_corr, pearson_p = pearsonr(merged['Score_train'], merged['Score_indep'])
        print(f"  Spearman ρ = {spearman_corr:.4f} (n={n_peptides})")

    # 保存结果
    results.append({
        '生成模型': model_name,
        '肽数量': n_peptides,
        'Spearman_ρ': spearman_corr,
        'Spearman_p值': spearman_p,
        'Pearson_r': pearson_corr,
        'Pearson_p值': pearson_p,
    })

# ==================== 输出结果 ====================
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n========== 结果汇总 ==========")
print(results_df.to_string(index=False))
print(f"\n结果已保存至: {output_file}")

# ==================== 可视化 ====================
# 过滤掉 NaN 值用于绘图
plot_df = results_df.dropna(subset=['Spearman_ρ'])

if len(plot_df) > 0:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df['生成模型'], plot_df['Spearman_ρ'], color='steelblue')
    plt.axhline(y=0.7, color='green', linestyle='--', label='高一致性阈值 (0.7)')
    plt.axhline(y=0.5, color='orange', linestyle='--', label='中等一致性阈值 (0.5)')
    plt.axhline(y=0.3, color='red', linestyle='--', label='弱一致性阈值 (0.3)')
    plt.ylabel('Spearman ρ')
    plt.title('各生成模型在两个毒性预测模型上的一致性')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('spearman_comparison.png', dpi=150)
    plt.show()
    print(f"\n柱状图已保存至: spearman_comparison.png")
else:
    print("\n没有足够数据生成柱状图")