import json
import pulp
import pandas as pd

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

#Soft Constraints

#a. Một người không trực hai ca liên tiếp nhau bất kể cơ sở
#Xác định hai ca liên tiếp là những cặp nào
consecutive_pairs = []
for d in dates:
    shifts_today = [j for j in J if Date_j[j] == d]
    shifts_today.sort()
    for k in range(len(shifts_today) - 1):
        consecutive_pairs.append((shifts_today[k], shifts_today[k + 1]))

Y_fatigue = pulp.LpVariable.dicts("Fatigue", ((i, pair) for i in I for pair in consecutive_pairs), cat=pulp.LpBinary)

# +1 nếu cơ sở được phân không gần staff
travel_pen = pulp.lpSum(X[i][j] for i in I for j in J for r in R if L_j[j] != 'Chưa xác định' and P_ic[i] != 'Neutral' and P_ic[i] != L_j[j])

# +1 nếu có staff làm hai ca liên tục bất kể cơ sở
fatigue_pen = pulp.lpSum(Y_fatigue[(i, pair)] for i in I for pair in consecutive_pairs)

#Xác định trọng số của các soft constraints (cái nào quan trọng hơn)
WEIGHT_TRAVEL = 2
WEIGHT_FATIGUE = 5

#Vì mục tiêu của hàm là minimize cho nên sẽ cố gắng đạt giá trị gần 0 nhất có thể
model += (WEIGHT_TRAVEL * travel_pen) + (WEIGHT_FATIGUE * fatigue_pen)

#Hard Constraints

# a. Đảm bảo đủ số lượng cán bộ cho mỗi ca thi (update)
for j in J:
    model += pulp.lpSum(X[i][j] for i in I) == R_j[j]
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

status = model.solve()
print(f"Final Status: {pulp.LpStatus[status]}")

if pulp.LpStatus[status] == 'Optimal':
    print(f"Total Penalty Score: {pulp.value(model.objective)}")
    sched = []
    for i in I:
        for j in J:
            for r in R:
                if pulp.value(X[i][j][r]) == 1:
                    sched.append({
                        "MS của CÁN BỘ COI THI": i,
                        "MS Ca thi": j,
                        "Nhiệm vụ": r,
                        "Ngày": Date_j[j],
                        "GIỜ": T_j[j],
                        "Cơ sở": L_j[j]
                    })
    
    final_df = pd.DataFrame(sched)
    final_df = final_df.sort_values(by=["Ngày", "GIỜ", "MS Ca thi"])
    final_df.to_csv("Output.csv", index=False, encoding='utf-8-sig')
else:
    print("Unable to solve")
        