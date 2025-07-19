import streamlit as st
import pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.neighbors import NearestNeighbors

kmeans = joblib.load("kmeans_model.pkl")
clf = joblib.load("stress_classifier.pkl")
reg = joblib.load("savings_regressor.pkl")
scaler = joblib.load("scaler.pkl")
persona_map = {0:"Stable Saver",1:"Over-Spender",2:"At-Risk"}

def score(data):
    s=0
    s+=20 if data["savings"]>0.2*data["income"] else 0
    s+=20 if data["expenses"]<0.5*data["income"] else 0
    s+=20 if data["debt"]<0.3*data["income"] else 0
    s+=20 if data["emergency_fund"]>=3 else 0
    s+=20 if data["budgeting"] else 0
    return s

def nlp_tips(persona, stress, reg_sav, user_sav):
    tips=[]
    if persona=="Over-Spender":
        tips.append("Track daily expenses and set spending limits.")
    if stress=="High":
        tips.append("Consider debt consolidation or talking to a financial advisor.")
    if reg_sav>user_sav:
        tips.append("You're saving lower than predicted—review your budget categories.")
    tips.append("Automate savings if you can.")
    return tips

st.title("🌟 Wealthy Ways")
st.write("Your personalized financial health dashboard")

income=st.number_input("Income", 0)
expenses=st.number_input("Expenses", 0)
savings=st.number_input("Savings", 0)
debt=st.number_input("Debt", 0)
emergency=st.slider("Emergency Fund (months)", 0, 12, 3)
budget=st.selectbox("Track budget monthly?", ["Yes","No"])=="Yes"

if st.button("Analyze"):
    u={"income":income,"expenses":expenses,"savings":savings,"debt":debt,
       "emergency_fund":emergency,"budgeting":int(budget)}
    st.metric("Score",f"{score(u)}/100")
    arr=scaler.transform([[*u.values()]])
    persona=persona_map[kmeans.predict(arr)[0]]
    stress="High" if clf.predict(arr)[0] else "Low"
    pred_next=int(reg.predict(arr)[0])
    st.success(f"Persona: {persona}")
    st.warning(f"Stress level: {stress}")
    st.metric("Predicted next month’s savings", f"₹{pred_next}")
    fig,ax=plt.subplots()
    ax.pie([expenses, savings], labels=["Expenses","Savings"], autopct="%1.1f%%")
    st.pyplot(fig)
    tips=nlp_tips(persona,stress,pred_next,savings)
    st.subheader("💬 Recommendations")
    for t in tips: st.write("- " + t)
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    pdf.cell(0,10,"Wealthy Ways Report",ln=1,align="C")
    for k,v in u.items(): pdf.cell(0,8,f"{k}: {v}",ln=1)
    pdf.cell(0,8,f"Score: {score(u)}/100",ln=1)
    pdf.cell(0,8,f"Persona: {persona}",ln=1)
    pdf.cell(0,8,f"Stress: {stress}",ln=1)
    pdf.cell(0,8,f"Next Savings: ₹{pred_next}",ln=1)
    pdf.ln(5)
    pdf.cell(0,8,"Tips:",ln=1)
    for t in tips: pdf.cell(0,8,"- "+t,ln=1)
    pdf.output("report.pdf")
    with open("report.pdf","rb") as f:
        st.download_button("Download Report",f,"WealthyWays_Report.pdf")