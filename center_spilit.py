import pandas as pd

csv_file = "/home/lin01231/song0760/CancerMoE/data/TCGA_9523sample_label_4-2-4_Censorship_HKUST.csv"
df = pd.read_csv(csv_file)

# 提取group_id
df['group_id'] = df['patient_id'].str.split('-').str[1]

# 你想要统计的cancer type
selected_types = ['UCEC', 'LUAD', 'LGG', 'BRCA', 'BLCA', 'PAAD', 'COAD', 'READ', 'KIRC', 'GBM']
df = df[df['cancer_type'].isin(selected_types)]

# 记录新的split
new_split = []

# 记录分组统计
split_stat = []

for ct in df['cancer_type'].unique():
    df_ct = df[df['cancer_type'] == ct].copy()
    group_df = df_ct.groupby('group_id').size().reset_index(name='count')
    group_df = group_df.sample(frac=1, random_state=42)  # 打乱顺序

    total = group_df['count'].sum()
    train_target = int(total * 0.4)
    val_target = int(total * 0.2)
    test_target = total - train_target - val_target

    splits = []
    split_names = []
    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    split_groups = {'train': [], 'valid': [], 'test': []}

    for _, row in group_df.iterrows():
        group_id = row['group_id']
        count = row['count']
        # 优先保证train:val:test ≈ 4:2:4
        # 动态分配，保证不会超太多
        if split_counts['train'] + count <= train_target:
            split = 'train'
        elif split_counts['valid'] + count <= val_target:
            split = 'valid'
        else:
            split = 'test'
        split_counts[split] += count
        split_groups[split].append((group_id, count))
        # 标记所有属于该group_id的样本
        splits.extend([split] * count)
        split_names.extend([split] * count)

        # 把该group下所有index都赋给对应split
        df_ct.loc[df_ct['group_id'] == group_id, 'split'] = split

    # 保存分组统计
    for split in ['train', 'valid', 'test']:
        group_list = [g[0] for g in split_groups[split]]
        count_list = [g[1] for g in split_groups[split]]
        split_stat.append({
            'cancer_type': ct,
            'split': split,
            'group_ids': ','.join(group_list),
            'group_num': len(group_list),
            'patient_num': sum(count_list)
        })
    # 更新总表
    new_split.append(df_ct)

# 拼接总表
df_new = pd.concat(new_split, axis=0)
# 保持原csv列顺序，只有split这一列替换为新的
col_order = ['patient_filename', 'text', 'patient_id', 'cancer_type', 'label',
             'raw_censorship', 'raw_survival_months', 'days_to_death',
             'censorship', 'survival_months', 'split']
df_new = df_new[col_order]

# 生成统计表
split_stat_df = pd.DataFrame(split_stat)

# 保存结果
df_new.to_csv('TCGA_9523sample_label_4-2-4_grouped_split.csv', index=False)
split_stat_df.to_csv('TCGA_grouped_split_stat.csv', index=False)
print("划分完成，结果已保存！")


