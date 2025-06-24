import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="졸피뎀 복용량 예측기")
st.title("💊 졸피뎀 복용량 및 수면 지속 예측기")

# 사용자 입력
st.sidebar.header("🧑 사용자 정보 입력")
나이 = st.sidebar.number_input("나이", min_value=10, max_value=100, value=25)
체중 = st.sidebar.number_input("체중 (kg)", min_value=30, max_value=150, value=60)
수면시각 = st.sidebar.number_input("수면 시작 시각 (예: 22.5)", min_value=0.0, max_value=24.0, value=22.0)

# 데이터셋
data = {
    "나이": [25, 35, 60, 45, 70, 30, 20, 50, 65, 40, 18, 55, 33, 42, 67, 38, 26, 48, 60, 31],
    "체중": [60, 70, 50, 65, 55, 68, 72, 59, 53, 61, 49, 54, 63, 58, 52, 66, 62, 60, 51, 67],
    "수면시작_시각": [23, 22, 21, 22.5, 20.5, 23.5, 24, 22, 21.5, 23, 22.5, 21, 23, 23.5, 20, 22, 22, 21, 21.5, 23],
    "추천_복용량": [5, 5, 2.5, 5, 2.5, 5, 5, 5, 2.5, 5, 5, 2.5, 5, 5, 2.5, 5, 5, 5, 2.5, 5],
    "추천_복용시각": [22.5, 21.5, 20, 21.5, 19.5, 22, 22.5, 21.5, 20, 22, 21.5, 20.5, 22, 22.5, 19.5, 21, 21.5, 20.5, 20, 22]
}
df = pd.DataFrame(data)

# 모델 학습
X = df[["나이", "체중", "수면시작_시각"]]
y = df[["추천_복용량", "추천_복용시각"]]
model = LinearRegression()
model.fit(X, y)

# 예측
입력값 = pd.DataFrame([[나이, 체중, 수면시각]], columns=["나이", "체중", "수면시작_시각"])
예측결과 = model.predict(입력값)
예측_복용량 = round(예측결과[0][0], 2)
예측_복용시각 = round(예측결과[0][1], 2)

st.subheader("✅ 예측 결과")
st.markdown(f"**추천 복용량:** {예측_복용량} mg")
st.markdown(f"**추천 복용 시각:** {예측_복용시각} 시")

# 경고 메시지
if 나이 >= 65 and 체중 <= 50 and 예측_복용량 > 5:
    st.error("⚠️ 고령 저체중 상태에서 복용량이 높습니다. 의사 상담 권장!")
elif 예측_복용량 > 7.5:
    st.warning("⚠️ 복용량이 일반 권장량보다 높습니다. 복용 주의!")
else:
    st.success("✅ 복용 안전 범위입니다.")

# 정규분포 기반 수면 지속 시간 예측
평균수면 = 예측_복용량
표준편차 = 1
P_4_to_6 = norm.cdf(6, 평균수면, 표준편차) - norm.cdf(4, 평균수면, 표준편차)
P_4_to_6 = round(P_4_to_6 * 100, 2)

st.subheader("📊 수면 지속 시간 예측")
st.markdown(f"**예측된 평균 수면 시간:** {평균수면} 시간")
st.markdown(f"**수면이 4~6시간 지속될 확률:** {P_4_to_6}%")

# 정규분포 시각화
x = [i/10 for i in range(int((평균수면-3)*10), int((평균수면+3)*10))]
y = [norm.pdf(val, 평균수면, 표준편차) for val in x]

fig1 = plt.figure(figsize=(8, 4))
plt.plot(x, y, label="수면 지속 시간 정규분포")
plt.axvline(x=평균수면, color='red', linestyle='--', label=f'예측 평균: {평균수면}시간')
plt.fill_between(x, 0, y, where=[4<=val<=6 for val in x], color='skyblue', alpha=0.4, label="4~6시간 구간")
plt.title("예상 수면 지속 시간 분포")
plt.xlabel("수면 지속 시간 (시간)")
plt.ylabel("확률 밀도")
plt.legend()
plt.grid(True)
st.pyplot(fig1)

# 사용자 위치 시각화
fig2 = plt.figure(figsize=(6, 5))
plt.scatter(df["체중"], df["나이"], c=df["추천_복용량"], cmap='coolwarm', label="훈련 데이터")
plt.scatter(체중, 나이, color='black', marker='X', s=120, label="당신")
plt.xlabel("체중 (kg)")
plt.ylabel("나이")
plt.title("사용자의 복용 위치 시각화")
plt.legend()
plt.colorbar(label="복용량 (mg)")
plt.grid(True)
st.pyplot(fig2)
