import json
import pulp
import pandas as pd
import numpy as np

with open('processed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

I = data['sets']['invigilators']
J = data['sets']['shifts']
dates = data['sets']['dates']
campuses = data['sets']['campuses']
R = [
    'LTK_Trưởng HĐ', 'LTK_Thư ký', 'LTK_CBCT',
    'DiAn_Trưởng HĐ', 'DiAn_Thư ký', 'DiAn_CBCT'
]

R_j = data['parameters']['req_staff'] #Số staff tối thiểu cho 1 ca thi
L_j = data['parameters']['shift_campus'] #Cơ sở
T_j = data['parameters']['shift_time']
Date_j = data['parameters']['shift_date'] #Ngày
Duration_j = data['parameters']['shift_duration'] #Thời gian 1 ca thi
P_ic = data['parameters']['staff_pref'] #Staff thích cơ sở nào hơn

model = pulp.LpProblem("EduSched_Problem", pulp.LpMinimize)

X = pulp.LpVariable.dicts("Assign", (I, J, R), cat=pulp.LpBinary)

# thêm workload balance trong soft constraints
# =====================================================================
# Soft Constraints (Ràng buộc mềm)
# =====================================================================

# a. Một người không trực hai ca liên tiếp nhau bất kể cơ sở
# Xác định hai ca liên tiếp là những cặp nào
consecutive_pairs = []
for d in dates:
    shifts_today = [j for j in J if Date_j[j] == d]
    shifts_today.sort()
    for k in range(len(shifts_today) - 1):
        consecutive_pairs.append((shifts_today[k], shifts_today[k + 1]))

Y_fatigue = pulp.LpVariable.dicts("Fatigue", ((i, pair) for i in I for pair in consecutive_pairs), cat=pulp.LpBinary)

# +1 nếu có staff làm hai ca liên tục bất kể cơ sở
fatigue_pen = pulp.lpSum(Y_fatigue[(i, pair)] for i in I for pair in consecutive_pairs)

# b. +1 nếu cơ sở được phân không gần staff (Đã sửa lỗi X[i][j] thành X[i][j][r])
travel_pen = pulp.lpSum(X[i][j][r] for i in I for j in J for r in R if L_j[j] != 'Chưa xác định' and P_ic[i] != 'Neutral' and P_ic[i] != L_j[j])

# c. Cân bằng khối lượng công việc (Workload Fairness)
# Khai báo biến dư (d_plus) và thiếu (d_minus) cho từng người
d_plus = pulp.LpVariable.dicts("d_plus", I, lowBound=0, cat=pulp.LpContinuous)
d_minus = pulp.LpVariable.dicts("d_minus", I, lowBound=0, cat=pulp.LpContinuous)

# Tính mức gác ca lý tưởng trung bình (Tổng số nhân sự cần / Tổng số cán bộ)
ideal_workload = sum(R_j.values()) / len(I)

# Thêm phương trình cân bằng cho từng giám thị
for i in I:
    total_worked = pulp.lpSum(X[i][j][r] for j in J for r in R)
    model += total_worked + d_minus[i] - d_plus[i] == ideal_workload

# Tổng điểm phạt chênh lệch khối lượng công việc
workload_pen = pulp.lpSum(d_plus[i] + d_minus[i] for i in I)

# Xác định trọng số của các soft constraints (cái nào quan trọng hơn)
WEIGHT_TRAVEL = 2
WEIGHT_FATIGUE = 5
WEIGHT_WORKLOAD = 15 # Đặt trọng số cao nhất để ưu tiên san sẻ công bằng

# Vì mục tiêu của hàm là minimize cho nên sẽ cố gắng đạt giá trị gần 0 nhất có thể
model += (WEIGHT_TRAVEL * travel_pen) + (WEIGHT_FATIGUE * fatigue_pen) + (WEIGHT_WORKLOAD * workload_pen)


# =====================================================================
# Hard Constraints
# =====================================================================

# a. Đảm bảo đủ số lượng cán bộ cho mỗi ca thi (update)
for j in J:
    model += pulp.lpSum(X[i][j][r] for i in I for r in R) == R_j[j]
    campus = L_j[j]
    if campus == 'Cơ sở 1':
        role_cap = 'LTK_Trưởng HĐ'
        role_sec = 'LTK_Thư ký'
        model += pulp.lpSum(X[i][j][r] for i in I for r in ['DiAn_Trưởng HĐ', 'DiAn_Thư ký', 'DiAn_CBCT']) == 0
    elif campus == 'Cơ sở 2':
        role_cap = 'DiAn_Trưởng HĐ'
        role_sec = 'DiAn_Thư ký'
        model += pulp.lpSum(X[i][j][r] for i in I for r in ['LTK_Trưởng HĐ', 'LTK_Thư ký', 'LTK_CBCT']) == 0
    else:
        continue

    if R_j[j] >= 3:
        #Trưởng HĐ chỉ có 1
        model += pulp.lpSum(X[i][j][role_cap] for i in I) == 1
        #Thư ký có 1 tới 2 người
        model += pulp.lpSum(X[i][j][role_sec] for i in I) >= 1
        model += pulp.lpSum(X[i][j][role_sec] for i in I) <= 2
    else:
        model += pulp.lpSum(X[i][j][role_cap] for i in I) == 0
        model += pulp.lpSum(X[i][j][role_sec] for i in I) == 0

#Chỉ được giao 1 nhiệm vụ 1 ca
for i in I:
    for j in J:
        model += pulp.lpSum(X[i][j][r] for r in R) <= 1

# b. Ràng buộc trùng lịch (Clash-free): Một người không trực 2 ca cùng lúc 
# Giả sử ca 1 và ca 2 trùng giờ nhau
for i in I:
    for d in dates:
        shifts_today = [j for j in J if Date_j[j] == d] #Lấy tất cả ca trong ngày
        for j1 in shifts_today:
            for j2 in shifts_today:
                if j1 != j2 and T_j[j1] == T_j[j2]:
                    worked_j1 = pulp.lpSum(X[i][j1][r] for r in R)
                    worked_j2 = pulp.lpSum(X[i][j2][r] for r in R)
                    model += worked_j1 + worked_j2 <= 1


# c. Ràng buộc Di chuyển Cơ sở (Campus Travel - Gợi ý thêm)
# Nếu ca 1 (CS1) và ca 2 (CS2) diễn ra sát nhau, không thể phân cùng 1 người
for i in I:
    for pair in consecutive_pairs:
        shift_A = pair[0]
        shift_B = pair[1]
        campus_A = L_j[shift_A]
        campus_B = L_j[shift_B]
        if campus_A != campus_B and campus_A != 'Chưa xác định' and campus_B != 'Chưa xác định':
            worked_A = pulp.lpSum(X[i][shift_A][r] for r in R)
            worked_B = pulp.lpSum(X[i][shift_B][r] for r in R)
            model += worked_A + worked_B <= 1

for i in I:
    for pair in consecutive_pairs:
        shift_A = pair[0]
        shift_B = pair[1]
        worked_A = pulp.lpSum(X[i][shift_A][r] for r in R)
        worked_B = pulp.lpSum(X[i][shift_B][r] for r in R)
        model += Y_fatigue[(i, pair)] >= worked_A + worked_B - 1


# sửa format file output 

status = model.solve()
print(f"\nFinal Status: {pulp.LpStatus[status]}")

if pulp.LpStatus[status] == 'Optimal':
    print(f"Total Penalty Score: {pulp.value(model.objective)}")
    
    sched = []
    
    # Tạo từ điển dịch Thứ sang Tiếng Việt
    weekday_map = {
        'Monday': 'Thứ 2', 'Tuesday': 'Thứ 3', 'Wednesday': 'Thứ 4', 
        'Thursday': 'Thứ 5', 'Friday': 'Thứ 6', 'Saturday': 'Thứ 7', 'Sunday': 'Chủ nhật'
    }

    for i in I:
        for j in J:
            for r in R:
                if pulp.value(X[i][j][r]) == 1:
                    # 1. Xử lý cột Thứ sang Tiếng Việt
                    eng_day = pd.to_datetime(Date_j[j]).strftime('%A')
                    vie_day = weekday_map[eng_day]
                    
                    # 2. Xử lý cột Ca thi chi tiết
                    date_obj = pd.to_datetime(Date_j[j])
                    date_str = date_obj.strftime('%d/%m/%Y')
                    ca_number = str(j).split('_')[-1] 
                    
                    start_time_str = T_j[j].replace('g', ':')
                    end_time_obj = pd.to_datetime(start_time_str, format='%H:%M') + pd.Timedelta(minutes=Duration_j[j])
                    end_time_str = end_time_obj.strftime('%Hg%M')
                    
                    full_cathi_name = f"{vie_day}, {date_str}, Ca {ca_number}: {T_j[j]}-{end_time_str}"

                    sched.append({
                        "Ca thi": full_cathi_name,
                        "Ngày": Date_j[j],
                        "GIỜ": T_j[j],
                        "MS Ca thi": j,
                        "Nhiệm vụ": r,
                        "MS của CÁN BỘ COI THI": i,
                        "Thời gian": Duration_j[j],
                        "Thứ": vie_day,
                        "Cơ sở": L_j[j]
                    })
                    
    output_df = pd.DataFrame(sched)
    output_df = output_df.sort_values(by=['Ngày', 'GIỜ', 'MS Ca thi'])
    
    columns_order = [
        "Ca thi", "Ngày", "GIỜ", "MS Ca thi", "Nhiệm vụ", 
        "MS của CÁN BỘ COI THI", "Thời gian", "Thứ", "Cơ sở"
    ]
    output_df = output_df[columns_order]
    
    output_df.to_csv('Output.csv', index=False, encoding='utf-8-sig')
    print("Đã xuất file kết quả: Output.csv")

else:
    print("Unable to solve - Mô hình bị vô nghiệm.")
        

#Performance evaluation
BASELINE_CSV = 'IAPDataset.csv'
OPTIMIZED_CSV = 'Output.csv'
PROCESSED_JSON = 'processed_data.json'
STAFF_COL = 'MS của CÁN BỘ COI THI'


def load_preferences(json_path):
    """Load staff campus preferences from processed_data.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['parameters']['staff_pref']


def compute_consecutive_pairs(df):
    """
    Replicate the solver's logic for identifying consecutive shift pairs.
    For each date, sort shifts by MS Ca thi and pair adjacent ones.
    """
    pairs = []
    for date, group in df.groupby('Ngày'):
        shifts_today = sorted(group['MS Ca thi'].unique())
        for k in range(len(shifts_today) - 1):
            pairs.append((shifts_today[k], shifts_today[k + 1]))
    return pairs


def evaluate_schedule(filepath, staff_pref, label):
    """Compute all 9 metrics for a given schedule CSV."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['Cơ sở'] = df['Cơ sở'].fillna('Chưa xác định')

    results = {}

    # -----------------------------------------------------------------
    # 1. Total number of assigned duties
    # -----------------------------------------------------------------
    results['total_duties'] = len(df)

    # -----------------------------------------------------------------
    # 2-4. Workload metrics
    # -----------------------------------------------------------------
    workloads = df[STAFF_COL].value_counts()
    all_staff = list(staff_pref.keys())
    # Include staff with 0 assignments
    workloads = workloads.reindex(all_staff, fill_value=0)

    results['max_workload'] = int(workloads.max())
    results['min_workload'] = int(workloads.min())

    ideal_workload = results['total_duties'] / len(all_staff)
    results['ideal_workload'] = ideal_workload
    results['mad_workload'] = float(np.mean(np.abs(workloads - ideal_workload)))

    # -----------------------------------------------------------------
    # 5. Number of simultaneous-shift conflicts
    #    (same person, same date, same time slot → assigned to >1 shift)
    # -----------------------------------------------------------------
    simultaneous = df.groupby([STAFF_COL, 'Ngày', 'GIỜ']).size()
    results['simultaneous_conflicts'] = int((simultaneous > 1).sum())

    # -----------------------------------------------------------------
    # 6. Number of cross-campus consecutive conflicts
    #    Same person assigned to two consecutive shifts on different campuses
    # -----------------------------------------------------------------
    consecutive_pairs = compute_consecutive_pairs(df)
    shift_campus = df.drop_duplicates('MS Ca thi').set_index('MS Ca thi')['Cơ sở'].to_dict()

    cross_campus_count = 0
    for (s1, s2) in consecutive_pairs:
        c1 = shift_campus.get(s1, 'Chưa xác định')
        c2 = shift_campus.get(s2, 'Chưa xác định')
        if c1 == 'Chưa xác định' or c2 == 'Chưa xác định' or c1 == c2:
            continue
        staff_s1 = set(df[df['MS Ca thi'] == s1][STAFF_COL])
        staff_s2 = set(df[df['MS Ca thi'] == s2][STAFF_COL])
        cross_campus_count += len(staff_s1 & staff_s2)

    results['cross_campus_conflicts'] = cross_campus_count

    # -----------------------------------------------------------------
    # 7. Total fatigue penalty
    #    Count how many (person, consecutive-pair) have both shifts assigned
    # -----------------------------------------------------------------
    fatigue_count = 0
    for (s1, s2) in consecutive_pairs:
        staff_s1 = set(df[df['MS Ca thi'] == s1][STAFF_COL])
        staff_s2 = set(df[df['MS Ca thi'] == s2][STAFF_COL])
        fatigue_count += len(staff_s1 & staff_s2)

    results['fatigue_penalty'] = fatigue_count

    # -----------------------------------------------------------------
    # 8. Total travel preference penalty
    #    Count assignments where campus ≠ staff preference (excluding
    #    Neutral preference and unknown campus), matching solver logic
    # -----------------------------------------------------------------
    travel_count = 0
    for _, row in df.iterrows():
        campus = row['Cơ sở']
        staff = row[STAFF_COL]
        pref = staff_pref.get(staff, 'Neutral')
        if campus != 'Chưa xác định' and pref != 'Neutral' and pref != campus:
            travel_count += 1

    results['travel_penalty'] = travel_count

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  {label}: {filepath}")
    print(f"{'='*60}")
    print(f"  1. Total assigned duties        : {results['total_duties']}")
    print(f"  2. Max workload per invigilator  : {results['max_workload']}")
    print(f"  3. Min workload per invigilator  : {results['min_workload']}")
    print(f"     (Ideal workload               : {results['ideal_workload']:.2f})")
    print(f"  4. Mean abs deviation (MAD)      : {results['mad_workload']:.4f}")
    print(f"  5. Simultaneous-shift conflicts  : {results['simultaneous_conflicts']}")
    print(f"  6. Cross-campus consec. conflicts: {results['cross_campus_conflicts']}")
    print(f"  7. Total fatigue penalty         : {results['fatigue_penalty']}")
    print(f"  8. Total travel preference penalty: {results['travel_penalty']}")
    print(f"{'='*60}")

    return results


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    staff_pref = load_preferences(PROCESSED_JSON)

    baseline = evaluate_schedule(BASELINE_CSV, staff_pref, 'BASELINE')
    optimized = evaluate_schedule(OPTIMIZED_CSV, staff_pref, 'OPTIMIZED OUTPUT')

    # -----------------------------------------------------------------
    # Side-by-side comparison table
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<40} {'Baseline':>10} {'Optimized':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")

    rows = [
        ('Total assigned duties',       'total_duties'),
        ('Max workload',                'max_workload'),
        ('Min workload',                'min_workload'),
        ('MAD from ideal workload',     'mad_workload'),
        ('Simultaneous-shift conflicts','simultaneous_conflicts'),
        ('Cross-campus consec. conflicts','cross_campus_conflicts'),
        ('Total fatigue penalty',       'fatigue_penalty'),
        ('Total travel pref. penalty',  'travel_penalty'),
    ]

    for label, key in rows:
        bval = baseline[key]
        oval = optimized[key]
        if isinstance(bval, float):
            print(f"  {label:<40} {bval:>10.4f} {oval:>10.4f}")
        else:
            print(f"  {label:<40} {bval:>10} {oval:>10}")
solver.py
