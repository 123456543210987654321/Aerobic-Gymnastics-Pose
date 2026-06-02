import pandas as pd
import os

# 路径配置
# 请修改为你的输入/输出文件路径 / Change to your file paths
score_file = r"../outputs/judge_scores.xlsx"
result_file = r"../outputs/final_scores.csv"
output_file = r"../outputs/final_scores_with_judges.csv"

# 读取评分表（Excel）
print("正在读取评分数据...")
df_score = pd.read_excel(score_file, engine="openpyxl")

# 重命名统一列名
df_score = df_score.rename(columns=lambda x: str(x).strip())

# 自动识别动作列
action_col = None
for c in df_score.columns:
    if "动作" in str(c):
        action_col = c
        break

df_score_clean = df_score[["person_ID", action_col, "评委评分"]].copy()
df_score_clean.columns = ["person_ID", "动作", "评委评分"]

# 清洗
df_score_clean["person_ID"] = df_score_clean["person_ID"].astype(str).str.strip()
df_score_clean["动作"] = df_score_clean["动作"].astype(str).str.strip()
df_score_clean["评委评分"] = pd.to_numeric(df_score_clean["评委评分"], errors="coerce")
df_score_clean = df_score_clean.dropna(subset=["person_ID", "动作", "评委评分"])

# 去重：同一人+同一动作只保留一个评分
df_score_clean = df_score_clean.groupby(["person_ID", "动作"], as_index=False)["评委评分"].mean().round(1)

# 读取最终结果表（CSV）
print("正在读取最终评分结果...")
df_result = pd.read_csv(result_file, encoding="utf-8-sig")
df_result = df_result.rename(columns=lambda x: str(x).strip())

# 自动识别CSV里的动作列
res_action_col = None
for c in df_result.columns:
    if "动作" in str(c):
        res_action_col = c
        break

df_result = df_result.rename(columns={res_action_col: "动作"})
df_result["person_ID"] = df_result["person_ID"].astype(str).str.strip()
df_result["动作"] = df_result["动作"].astype(str).str.strip()

# 核心：按 person_ID + 动作 双条件匹配
print("开始按 person_ID + 动作 双列匹配评委评分...")
final = pd.merge(
    df_result,
    df_score_clean,
    on=["person_ID", "动作"],
    how="left"
)

# 保存
final.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n匹配完成！")
print(f"新文件已保存到：{output_file}")
print("成功将【评委评分】按 person_ID + 动作 匹配完成！")
