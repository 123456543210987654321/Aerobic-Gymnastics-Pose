import os
import re
import numpy as np
import pandas as pd

# 路径配置
# 请修改为你的 YOLO 标签文件夹路径 / Change to your labels directory
LABELS_ROOT = r'../YOLO_Dataset/labels'
SAVE_DIR = '../outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# 正/负理想解（标准动作、最差动作）
STANDARD_CONFIG = {
    "团身跳": "person_ID_9_1",
    "屈体分腿跳": "person_ID_12_1",
    "纵劈腿跳": "person_ID_13_1"
}

NEG_STANDARD_CONFIG = {
    "团身跳": "person_ID_4_9",
    "屈体分腿跳": "person_ID_9_1",
    "纵劈腿跳": "person_ID_6_1"
}


# 文件名解析
def parse_filename(filename):
    match = re.search(r'^(.+)_(person_ID_\d+_\d+)_frame_(\d+)\.txt$', filename)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None, None, None


# 角度计算
def angle_between(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot = np.dot(v1, v2)
    mod = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
    cos_ang = np.clip(dot / mod, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


def calculate_all_angles(kpts):
    try:
        kpts = np.array(kpts, dtype=np.float32)
        ls = kpts[5]
        rs = kpts[6]
        lh = kpts[11]
        rh = kpts[12]
        lk = kpts[13]
        rk = kpts[14]
        la = kpts[15]
        ra = kpts[16]

        left_knee = angle_between(lh, lk, la)
        right_knee = angle_between(rh, rk, ra)
        left_hip = angle_between(ls, lh, lk)
        right_hip = angle_between(rs, rh, rk)
        left_ankle = angle_between(lk, la, lh)
        right_ankle = angle_between(rk, ra, rh)
        split_angle = angle_between(lk, (lh + rh) / 2, rk)

        return {
            "左膝关节角度": round(left_knee, 2),
            "右膝关节角度": round(right_knee, 2),
            "左髋关节角度": round(left_hip, 2),
            "右髋关节角度": round(right_hip, 2),
            "左踝关节角度": round(left_ankle, 2),
            "右踝关节角度": round(left_ankle, 2),
            "分腿角度": round(split_angle, 2),
        }
    except:
        return None


# 评价指标提取
def extract_metrics(df_person):
    try:
        df_person = df_person.sort_values("帧号").reset_index(drop=True)
        metrics = {}
        joints = ["左髋关节角度", "右髋关节角度", "左膝关节角度", "右膝关节角度", "左踝关节角度", "右踝关节角度"]

        for j in joints:
            initial = df_person[j].iloc[0]
            min_val = df_person[j].min()
            metrics[f"{j}_缓冲幅度"] = round(initial - min_val, 2)

        frame_count = df_person["帧号"].max() - df_person["帧号"].min()
        metrics["缓冲时长"] = round(frame_count / 30, 3) if frame_count > 0 else 0.01

        for j in joints:
            amp = metrics[f"{j}_缓冲幅度"]
            metrics[f"{j}_变化速率"] = round(amp / metrics["缓冲时长"], 2)

        metrics["膝关节对称差"] = round(abs(df_person["左膝关节角度"].mean() - df_person["右膝关节角度"].mean()), 2)
        metrics["髋关节对称差"] = round(abs(df_person["左髋关节角度"].mean() - df_person["右髋关节角度"].mean()), 2)
        metrics["踝关节对称差"] = round(abs(df_person["左踝关节角度"].mean() - df_person["右踝关节角度"].mean()), 2)

        for c in df_person.columns:
            if "角度" in c:
                metrics[f"{c}_标准差"] = round(df_person[c].std(), 2) if len(df_person) > 1 else 0.0
        return metrics
    except:
        return None


# 中间指标正向化
def middle_normalize(col, best_val):
    return 1 - np.abs(col - best_val) / (np.max(col) - np.min(col) + 1e-12)


# 输出 X_best 和区间 [a,b]
def print_xbest_and_ab(df_metrics, action_name):
    best_id = STANDARD_CONFIG[action_name]
    df_best = df_metrics[df_metrics["人员ID"] == best_id]
    df_act = df_metrics[df_metrics["动作"] == action_name]

    cols = [c for c in df_metrics.columns if c not in ["动作", "人员ID"]]
    positive_cols = ["分腿角度"]
    negative_cols = [c for c in cols if "对称差" in c or "标准差" in c]
    middle_cols = [c for c in cols if "缓冲幅度" in c or "变化速率" in c or "缓冲时长" in c]

    print(f"\n" + "-" * 62)
    print(f"动作：{action_name}    |    标准样本：{best_id}")
    print("-" * 62)
    print(f"{'指标类型':<10} {'指标名称':<22} {'X_best':<12} {'区间 [a, b]'}")
    print("-" * 62)

    for name in cols:
        a = round(df_act[name].min(), 4)
        b = round(df_act[name].max(), 4)
        x_best = round(df_best[name].values[0], 4) if not df_best.empty else "-"

        if name in positive_cols:
            print(f"{'极大型':<10} {name:<22} {'-':<12} [{a}, {b}]")
        elif name in negative_cols:
            print(f"{'极小型':<10} {name:<22} {'-':<12} [{a}, {b}]")
        elif name in middle_cols:
            print(f"{'中间型':<10} {name:<22} {x_best:<12} [{a}, {b}]")


# 熵权TOPSIS双理想解评分
def entropy_topsis_double_ideal(df_action, action_name):
    best_id = STANDARD_CONFIG[action_name]
    worst_id = NEG_STANDARD_CONFIG[action_name]
    df_best = df_action[df_action["人员ID"] == best_id]
    df_worst = df_action[df_action["人员ID"] == worst_id]
    if df_best.empty or df_worst.empty:
        return None

    cols = [c for c in df_action.columns if c not in ["动作", "人员ID"]]
    X = df_action[cols].values.copy()
    n, m = X.shape

    positive_cols = ["分腿角度"]
    negative_cols = [c for c in cols if "对称差" in c or "标准差" in c]
    middle_cols = [c for c in cols if "缓冲幅度" in c or "变化速率" in c or "缓冲时长" in c]

    X_norm = np.zeros_like(X, dtype=np.float64)
    for j in range(m):
        col = X[:, j]
        c_min, c_max = col.min(), col.max()
        name = cols[j]

        if name in positive_cols:
            X_norm[:, j] = (col - c_min) / (c_max - c_min + 1e-12)
        elif name in negative_cols:
            X_norm[:, j] = (c_max - col) / (c_max - c_min + 1e-12)
        elif name in middle_cols:
            best_val = df_best[name].values[0]
            X_norm[:, j] = middle_normalize(col, best_val)

    p = X_norm / (X_norm.sum(axis=0) + 1e-12)
    p[p < 1e-12] = 1e-12
    e = -(1 / np.log(n + 1e-12)) * np.sum(p * np.log(p), axis=0)
    w = (1 - e) / (np.sum(1 - e) + 1e-12)
    weighted = X_norm * w

    best_vec = weighted[df_action["人员ID"] == best_id].flatten()
    worst_vec = weighted[df_action["人员ID"] == worst_id].flatten()

    scores = []
    for row in weighted:
        d_best = np.linalg.norm(row - best_vec)
        d_worst = np.linalg.norm(row - worst_vec)
        sim = d_worst / (d_best + d_worst + 1e-12)
        scores.append(round(8 + sim * 2, 2))

    df_out = df_action.copy()
    df_out["得分"] = scores
    df_out["排名"] = df_out["得分"].rank(ascending=False).astype(int)
    return df_out


# 主程序
def main():
    data = []
    for root, _, files in os.walk(LABELS_ROOT):
        for f in files:
            if not f.endswith(".txt"): continue
            action, pid, frame = parse_filename(f)
            if not action or not pid: continue
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8") as ff:
                    nums = list(map(float, ff.readline().strip().split()))
                if len(nums) < 34: continue
                kpts = [[nums[1 + 2 * i], nums[2 + 2 * i]] for i in range(17)]
                angles = calculate_all_angles(kpts)
                if angles:
                    data.append({"动作": action, "人员ID": pid, "帧号": frame, **angles})
            except:
                continue

    df = pd.DataFrame(data)
    df.to_csv(f"{SAVE_DIR}/关节角度.csv", index=False, encoding="utf-8-sig")

    metrics_list = []
    for action in df["动作"].unique():
        for pid in df[df["动作"] == action]["人员ID"].unique():
            sub = df[(df["动作"] == action) & (df["人员ID"] == pid)]
            met = extract_metrics(sub)
            if met:
                metrics_list.append({"动作": action, "人员ID": pid, **met})

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(f"{SAVE_DIR}/评价指标.csv", index=False, encoding="utf-8-sig")

    # 输出 X_best 和 a,b 区间
    print("\n各指标 X_best 与区间 [a,b] 输出：")
    for action in df_metrics["动作"].unique():
        if action in STANDARD_CONFIG:
            print_xbest_and_ab(df_metrics[df_metrics["动作"] == action].copy(), action)

    all_scores = []
    for action in df_metrics["动作"].unique():
        if action not in STANDARD_CONFIG: continue
        res = entropy_topsis_double_ideal(df_metrics[df_metrics["动作"] == action].copy(), action)
        if res is not None:
            all_scores.append(res)

    if all_scores:
        final = pd.concat(all_scores, ignore_index=True)
        final.to_csv(f"{SAVE_DIR}/最终评分结果.csv", index=False, encoding="utf-8-sig")
        print("\nTOPSIS 得分结果：")
        print(final[["动作", "人员ID", "得分", "排名"]].to_string(index=False))

    print("\n全部完成！")


if __name__ == "__main__":
    main()
