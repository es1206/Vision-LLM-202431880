import numpy as np

# 학생 정보 및 국어, 수학, 과학 성적
raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
]

raw_scores = np.array([
    [95, 82, 88],  # 짱구
    [70, 65, 90],  # 철수
    [98, 100, 92], # 맹구
    [60, 75, 80],  # 유리
    [81, 88, 84]   # 훈이
])

subjects = ["국어", "수학", "과학"]

# 1. 학번과 이름을 분리하여 이름만 추출
student_names = [student.split("_")[1] for student in raw_students]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)

# 2. 2차원 성적 배열을 1차원으로 변환
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# 3. 85점 이상인 성적 개수 계산
over_85 = np.sum(all_scores >= 85)
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# 4. 각 학생의 세 과목 평균 계산
student_means = np.mean(raw_scores, axis=1)
mean_results = [f"{name} {mean:.2f}점" for name, mean in zip(student_names, student_means)]
print(f"4. 학생별 평균 점수: {', '.join(mean_results)}", end="\n\n")

# 5. 평균이 가장 높은 학생 찾기
top_student_idx = np.argmax(student_means)
top_student_name = student_names[top_student_idx]
print(f"5. 최고 평균 수석 학생: {top_student_name} ({student_means[top_student_idx]:.2f}점)", end="\n\n")

# 6. 최고 점수의 위치를 학생 번호와 과목 번호로 변환
overall_max_score = np.max(all_scores)
overall_max_idx = np.argmax(all_scores)
overall_max_subject = subjects[overall_max_idx % len(subjects)]
overall_max_student = student_names[overall_max_idx // len(subjects)]
print(f"6. 단일 과목 최고 점수: {overall_max_score}점 (과목: {overall_max_subject}, 학생: {overall_max_student})")
