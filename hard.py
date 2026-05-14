import pulp

# 1. Khởi tạo bài toán
prob = pulp.LpProblem("Invigilator_Assignment_Problem", pulp.LpMinimize)

# 2. Định nghĩa Tập hợp (Sets) - Dữ liệu giả định từ Excel
# Trong thực tế, dùng pandas để đọc file này
exams = [1, 2, 3, 4]  # Danh sách ID ca thi
staffs = ['CB001', 'CB002', 'CB003']  # Danh sách cán bộ

# Tham số giả định
# campus: {exam_id: 1 hoặc 2}
exam_campus = {1: 1, 2: 2, 3: 1, 4: 2} 
# slots: {exam_id: khung giờ}
exam_slots = {1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4'}
# required_staff: Số lượng nhân viên hành chính cần cho mỗi ca 
req_staff = {1: 1, 2: 2, 3: 1, 4: 1}

# 3. Biến quyết định (Decision Variables)
# x[i, j] = 1 nếu cán bộ j trực ca i [cite: 66, 215]
x = pulp.LpVariable.dicts("x", (exams, staffs), cat=pulp.LpBinary)

# 4. Ràng buộc cứng (Hard Constraints)

# a. Đảm bảo đủ số lượng cán bộ cho mỗi ca thi 
for i in exams:
    prob += pulp.lpSum([x[i][j] for j in staffs]) == req_staff[i]

# b. Ràng buộc trùng lịch (Clash-free): Một người không trực 2 ca cùng lúc 
# Giả sử ca 1 và ca 2 trùng giờ nhau (E_o)
clash_exams = [(1, 2)] 
for j in staffs:
    for (i, k) in clash_exams:
        prob += x[i][j] + x[k][j] <= 1

# c. Ràng buộc Di chuyển Cơ sở (Campus Travel - Gợi ý thêm)
# Nếu ca 1 (CS1) và ca 2 (CS2) diễn ra sát nhau, không thể phân cùng 1 người
travel_conflict = [(1, 2)] # Cặp ca thi sát giờ nhưng khác cơ sở
for j in staffs:
    for (i, k) in travel_conflict:
        prob += x[i][j] + x[k][j] <= 1

# d. Giới hạn số ca tối đa một ngày (ví dụ k=2) [cite: 93]
max_shifts_per_day = 2
for j in staffs:
    prob += pulp.lpSum([x[i][j] for i in exams]) <= max_shifts_per_day

# 5. Hàm mục tiêu (Objective Function)
# Ở đây tạm thời để 0 để check tính khả thi (Feasibility)
# Sau này bạn sẽ thêm các biến Alpha, Beta cho Soft Constraints vào đây [cite: 116]
prob += 0

# 6. Giải bài toán
status = prob.solve()
print(f"Trạng thái: {pulp.LpStatus[status]}")

# Xuất kết quả
if pulp.LpStatus[status] == 'Optimal':
    for i in exams:
        for j in staffs:
            if pulp.value(x[i][j]) == 1:
                print(f"Cán bộ {j} trực ca {i} tại Cơ sở {exam_campus[i]}")