

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import base64
import io

# Load models
kmeans_model = joblib.load('kmeans_model.pkl')
savings_model = joblib.load('savings_predictor.pkl')

# Set theme and title
st.set_page_config(page_title="Wealthy Ways", page_icon="💰", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f8;
        background-image: linear-gradient(to right, #fefcea, #f1da36);
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💸 Wealthy Ways - Financial Wellbeing Profiler")
st.write("Answer the following questions to assess and improve your financial health.")

# User info
name = st.text_input("👤 Enter your name")
age = st.number_input("🎂 Enter your age", min_value=10, max_value=100)

# Questionnaire
income = st.number_input("💼 Monthly Income (in ₹)", min_value=0)
expenses = st.number_input("🧾 Monthly Expenses (in ₹)", min_value=0)
savings = st.number_input("🏦 Monthly Savings (in ₹)", min_value=0)
debt = st.number_input("💳 Total Debt (in ₹)", min_value=0)

# Derived metrics
if income > 0:
    savings_rate = (savings / income) * 100
    debt_to_income = (debt / income) * 100
else:
    savings_rate = 0
    debt_to_income = 0

# Rule-based scoring
score = 0
if savings_rate >= 20:
    score += 2
elif savings_rate >= 10:
    score += 1

if debt_to_income < 20:
    score += 2
elif debt_to_income < 35:
    score += 1

if expenses < income:
    score += 2
elif expenses == income:
    score += 1

# KMeans persona prediction
user_data = np.array([[income, expenses, savings, debt]])
scaled_data = StandardScaler().fit_transform(user_data)
persona_cluster = kmeans_model.predict(scaled_data)[0]
persona_map = {0: "💰 Stable Saver", 1: "📉 Over-Spender", 2: "🔄 Break-Evener"}
financial_persona = persona_map.get(persona_cluster, "Unknown")

# Savings prediction
next_month_savings = savings_model.predict(user_data)[0]

# Display results
st.markdown("---")
st.subheader(f"Hi {name}, here's your Financial Wellness Snapshot 🧾")
st.metric("💹 Financial Wellness Score", f"{score} / 6")
st.metric("📊 Savings Rate", f"{savings_rate:.2f}%")
st.metric("📉 Debt-to-Income Ratio", f"{debt_to_income:.2f}%")
st.metric("🔮 Predicted Savings Next Month", f"₹{next_month_savings:.2f}")
st.markdown(f"**🧠 Financial Persona:** {financial_persona}")

# Pie chart
labels = ['Savings', 'Expenses']
sizes = [savings, expenses]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#6ee7b7', '#fca5a5'])
ax.axis('equal')
st.pyplot(fig)

# Tips
st.markdown("### 💡 Personalized Tips")
if financial_persona == "📉 Over-Spender":
    st.write("- Track expenses and create a strict monthly budget.")
    st.write("- Reduce high-interest debt aggressively.")
elif financial_persona == "🔄 Break-Evener":
    st.write("- Focus on saving a fixed amount monthly.")
    st.write("- Consider emergency fund and goal-based investing.")
elif financial_persona == "💰 Stable Saver":
    st.write("- Explore investment options like SIPs, PPF, etc.")
    st.write("- Diversify savings to grow wealth.")

# Downloadable PDF report
if st.button("📄 Download Your Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Financial Report for {name}, Age {age}", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Financial Wellness Score: {score}/6\nSavings Rate: {savings_rate:.2f}%\nDebt-to-Income Ratio: {debt_to_income:.2f}%\nPredicted Savings: ₹{next_month_savings:.2f}\nPersona: {financial_persona}")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    b64 = base64.b64encode(pdf_output.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{name}_WealthyWays_Report.pdf">Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Model evaluation snippet (run in Colab during training)
# from sklearn.metrics import mean_absolute_error
# y_pred = model.predict(X_test)
# print("MAE:", mean_absolute_error(y_test, y_pred))
