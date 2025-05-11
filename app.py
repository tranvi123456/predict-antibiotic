import streamlit as st
import numpy as np
import pandas as pd  
import joblib
from pre_process import modify_antibiotic_column, age_columns, select_columns_1, filter_sensitivity, replace_sensitivity, remove_duplicates, merge_unique_prb, select_columns_2, encode_columns, preprocess_pipeline, xgb_model, age_columns_new_pt, select_columns, add_antibiotic_column, load_and_encode, decode_and_overwrite, preprocess_pipeline_new_pt
from xgboost import XGBClassifier
from datetime import datetime, date



model = joblib.load('xgb_model.pkl')





# GUI


# Tải model đã huấn luyện
model = joblib.load("xgb_model.pkl")

st.title("🔬 Antibiotic susceptibility prediction")

st.markdown("Enter the patient's details below to predict antibiotic sensitivity")

# Nhập từng thông tin
visible_patient_id = st.number_input("🆔 HN:")
sex_rcd = st.selectbox("👤 Sex:", options=["M", "F"])
date_of_birth = st.date_input("🎂 Date of Birth", value=date(2000, 1, 1), format="YYYY-MM-DD")
df_doctor = pd.read_excel("dimension.xlsx", sheet_name="order_owner")
doctor_list = df_doctor["order_owner"].dropna().unique().tolist()
order_owner = st.selectbox("👨‍⚕️ Order Owner:", options=doctor_list)
df_location = pd.read_excel("dimension.xlsx", sheet_name="location")
location_list = df_location["location"].dropna().unique().tolist()
location = st.selectbox("👨‍⚕️ Location:", options=location_list)
df_department_order = pd.read_excel("dimension.xlsx", sheet_name="department_ordered")
department_ordered_list = df_department_order["department_ordered"].dropna().unique().tolist()
department_ordered = st.selectbox("📍 Department Ordered:", options=department_ordered_list)
problem = st.text_input("🦠 Problem:")

if st.button("🚀 Predict"):
    # Tạo dataframe từ dữ liệu nhập
    new_patient = pd.DataFrame([{
        "VISIBLE_PATIENT_ID": visible_patient_id,
        "SEX_RCD": sex_rcd,
        "DATE_OF_BIRTH": date_of_birth,
        "ORDER_OWNER": order_owner,
        "LOCATION": location,
        "DEPARTMENT_ORDERED": department_ordered,
        "PROBLEM": problem,
        }])

    # Encode lại nếu cần (chú ý: cần giống lúc huấn luyện)
    new_patient_encoded = preprocess_pipeline_new_pt(new_patient)

  
    try:
    # Dự đoán phân lớp và xác suất
        y_pred = model.predict(new_patient_encoded)
        proba = model.predict_proba(new_patient_encoded)

        # Gán kết quả vào DataFrame
        new_patient_encoded['SENSITIVITY_INTERPRETION'] = y_pred
        new_patient_encoded['PROB_SENSITIVE'] = np.round(proba[:, 1], 2)  # Xác suất nhạy
        new_patient_encoded['PROB_RESISTANT'] = np.round(proba[:, 0], 2)  # Xác suất kháng
        
        new_patient_encoded = decode_and_overwrite(new_patient_encoded)
        pt_pred_show = new_patient_encoded[['SENSITIVITY_ANTIBIOTIC', 'SENSITIVITY_INTERPRETION', 'PROB_SENSITIVE', 'PROB_RESISTANT']]
        
        # Hiển thị kết quả
        st.subheader("📋 Kết quả dự đoán")
        st.dataframe(pt_pred_show)

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
