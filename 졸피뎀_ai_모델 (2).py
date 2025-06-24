import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zolpidem Dose Predictor")
st.title("💊 Zolpidem Dose & Sleep Duration Predictor")

# Sidebar - user input
st.sidebar.header("🧑 Enter Your Info")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=150, value=60)
sleep_time = st.sidebar.number_input("Sleep Start Time (e.g., 22.5)", min_value=0.0, max_value=24.0, value=22.0)

# Dataset
data = {
    "Age": [25, 35, 60, 45, 70, 30, 20, 50, 65, 40, 18, 55, 33, 42, 67, 38, 26, 48, 60, 31],
    "Weight": [60, 70, 50, 65, 55, 68, 72, 59, 53, 61, 49, 54, 63, 58, 52, 66, 62, 60, 51, 67],
    "Sleep_Start": [23, 22, 21, 22.5, 20.5, 23.5, 24, 22, 21.5, 23, 22.5, 21, 23, 23.5, 20, 22, 22, 21, 21.5, 23],
    "Dose": [5, 5, 2.5, 5, 2.5, 5, 5, 5, 2.5, 5, 5, 2.5, 5, 5, 2.5, 5, 5, 5, 2.5, 5],
    "Recommended_Time": [22.5, 21.5, 20, 21.5, 19.5, 22, 22.5, 21.5, 20, 22, 21.5, 20.5, 22, 22.5, 19.5, 21, 21.5, 20.5, 20, 22]
}
df = pd.DataFrame(data)

# Train model
X = df[["Age", "Weight", "Sleep_Start"]]
y = df[["Dose", "Recommended_Time"]]
model = LinearRegression()
model.fit(X, y)

# Predict
input_data = pd.DataFrame([[age, weight, sleep_time]], columns=["Age", "Weight", "Sleep_Start"])
prediction = model.predict(input_data)
predicted_dose = round(prediction[0][0], 2)
predicted_time = round(prediction[0][1], 2)

st.subheader("✅ Prediction Result")
st.markdown(f"**Recommended Dose:** {predicted_dose} mg")
st.markdown(f"**Recommended Time to Take:** {predicted_time} hr")

# Warning system
if age >= 65 and weight <= 50 and predicted_dose > 5:
    st.error("⚠️ Elderly & underweight: high dose detected. Please consult a doctor.")
elif predicted_dose > 7.5:
    st.warning("⚠️ High dose. Use with caution.")
else:
    st.success("✅ Safe dosage range.")

# Sleep duration probability based on normal distribution
mean_sleep = predicted_dose
std_dev = 1
P_4_to_6 = norm.cdf(6, mean_sleep, std_dev) - norm.cdf(4, mean_sleep, std_dev)
P_4_to_6 = round(P_4_to_6 * 100, 2)

st.subheader("📊 Sleep Duration Prediction")
st.markdown(f"**Predicted average sleep duration:** {mean_sleep} hours")
st.markdown(f"**Probability of sleeping 4–6 hours:** {P_4_to_6}%")

# Plot: Normal distribution curve
x = [i / 10 for i in range(int((mean_sleep - 3) * 10), int((mean_sleep + 3) * 10))]
y = [norm.pdf(val, mean_sleep, std_dev) for val in x]

fig1 = plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Sleep Duration Distribution")
plt.axvline(x=mean_sleep, color='red', linestyle='--', label=f'Predicted Mean: {mean_sleep} hr')
plt.fill_between(x, 0, y, where=[4 <= val <= 6 for val in x], color='skyblue', alpha=0.4, label="4–6 hr range")
plt.title("Estimated Sleep Duration Distribution")
plt.xlabel("Sleep Duration (hr)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
st.pyplot(fig1)

# Plot: User location visualization
fig2 = plt.figure(figsize=(6, 5))
plt.scatter(df["Weight"], df["Age"], c=df["Dose"], cmap='coolwarm', label="Training Data")
plt.scatter(weight, age, color='black', marker='X', s=120, label="You")
plt.xlabel("Weight (kg)")
plt.ylabel("Age")
plt.title("Your Position in Training Dataset")
plt.legend()
plt.colorbar(label="Dose (mg)")
plt.grid(True)
st.pyplot(fig2)

