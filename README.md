# 🎓 Invigilator Assignment Problem (IAP) Solver

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PuLP](https://img.shields.io/badge/PuLP-Optimization-orange.svg)](https://coin-or.github.io/pulp/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green.svg)](https://pandas.pydata.org/)

Hệ thống tối ưu hóa và tự động hóa phân công lịch Cán bộ coi thi (CBCT) dành cho cấp Đại học. Dự án ứng dụng **Quy hoạch tuyến tính nguyên hỗn hợp (MILP)** kết hợp với **Quy hoạch mục tiêu (Goal Programming)** để không chỉ tìm ra lịch phân công hợp lệ mà còn đảm bảo tính công bằng và nhân văn cho đội ngũ giảng viên.

## 🌟 Tính năng nổi bật

Hệ thống được chia làm hai bộ ràng buộc chính, giải quyết triệt để bài toán thực tế của nhà trường:

### 🎯 Ràng buộc mềm (Soft Constraints - Goal Programming)
Tối ưu hóa sự thoải mái cho cán bộ dựa trên hệ số phạt (Penalty Weights):
* **Workload Fairness (Trọng số cao nhất):** Cân bằng khối lượng công việc, đảm bảo số ca gác của mọi người xấp xỉ mức trung bình, chống tình trạng dồn việc.
* **Fatigue Minimization:** Hạn chế tối đa việc phân công một người gác 2 ca liên tiếp gây mệt mỏi.
* **Campus Affinity:** Hạn chế xếp ca trái với sở thích cơ sở (Location Preference) của cán bộ.

""
pip3 install pandas pulp

1. Data processing: run python3 dataprocessing.py
  
2. Solve: run python3 solver.py
""
