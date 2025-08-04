import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import date

from pre_process import (
    # preprocess_pipeline_df,
    # preprocess_pipeline_df1,
    preprocess_pipeline_new_pt_1_1,
    preprocess_pipeline_new_pt_1_2,
    preprocess_pipeline_new_pt,
    decode_and_overwrite_1_1,
    decode_and_overwrite_1_2,
    decode_and_overwrite_2,
    highlight_broad_spectrum
)

# Load models and data
@st.cache_resource
def load_models():
    return {
        "gram_group": joblib.load('xgb_model_gram_group.pkl'),
        "rs_1": joblib.load('voting_model_rs_1.pkl'),
        "rs_2": joblib.load('xgb_model_rs_2.pkl'),
        "gram_encoder": joblib.load('timeout_encoders.3/GRAM_GROUP_encoder.pkl')
    }

@st.cache_data
def load_dimensions():
    return {
        "doctor_list": pd.read_excel("dimension.xlsx", sheet_name="order_owner.1")["order_owner"].dropna().unique().tolist(),
        "source_list": pd.read_excel("dimension.xlsx", sheet_name="specimen_source")["Source"].dropna().unique().tolist(),
        "department_list": pd.read_excel("dimension.xlsx", sheet_name="department_ordered")["department_ordered"].dropna().unique().tolist(),
        "organism_list": pd.read_excel("dimension.xlsx", sheet_name="organism")["ISOLATE_ORGANISM"].dropna().unique().tolist(),
        "popular_by_gram": pd.read_excel("popular_by_gram.xlsx"),
        "popular_by_organism": pd.read_excel("popular_by_organism.xlsx")
    }

models = load_models()
dims = load_dimensions()

# App UI
st.title("🔬 Antibiotic Susceptibility Prediction")
st.markdown("Dự đoán khả năng nhạy cảm với kháng sinh của vi sinh vật.")

user_choice = st.radio("Bạn đã có kết quả định danh vi sinh vật chưa?", ["Chưa có", "Đã có"])

with st.form("patient_form"):
    visible_patient_id = st.number_input("🆔 Mã bệnh nhân (HN):")
    sex_rcd = st.selectbox("👤 Giới tính:", ["M", "F"])
    dob = st.date_input("🎂 Ngày sinh:", min_value=date(1920, 1, 1), value=date(2000, 1, 1), max_value=date.today())
    order_owner = st.selectbox("👨‍⚕️ Bác sĩ chỉ định:", dims["doctor_list"])
    specimen_source = st.selectbox("🧪 Nguồn mẫu:", dims["source_list"])
    department_ordered = st.selectbox("🏥 Khoa chỉ định:", dims["department_list"])
    problem = st.text_input("🦠 Chẩn đoán lâm sàng:")
    ordered_date_time = st.date_input("🎂 Ngày gửi yêu cầu xét nghiệm:", min_value=date(2025, 1, 1), value=date(2025, 1, 1), max_value=date.today())
    


    if user_choice == "Đã có":
        isolated_organism = st.selectbox("🔍 Vi sinh vật đã phân lập:", dims["organism_list"])

    submitted = st.form_submit_button("🚀 Dự đoán")

if submitted:
    try:
        # Chung
        new_patient = {
            "VISIBLE_PATIENT_ID": visible_patient_id,
            "SEX_RCD": sex_rcd,
            "DATE_OF_BIRTH": dob,
            "ORDER_OWNER.1": order_owner,
            "SPECIMEN_SOURCE": specimen_source,
            "DEPARTMENT_ORDERED": department_ordered,
            "PROBLEM": problem,
            "ORDERED_DATE_TIME": ordered_date_time,
        }

        if user_choice == "Chưa có":
            df_1 = pd.DataFrame([new_patient])
            df_1_1 = preprocess_pipeline_new_pt_1_1(df_1)
            y_pred_gram = models["gram_group"].predict(df_1_1)
            df_1_1['GRAM_GROUP'] = models["gram_encoder"].inverse_transform(y_pred_gram)
            # df_1_1_show = df_1_1[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'GRAM_GROUP']]
            df_1_1 = decode_and_overwrite_1_1(df_1_1)

            st.success("✅ Dự đoán nhóm Gram thành công!")
            st.subheader("📋 Kết quả nhóm vi sinh vật:")
            st.dataframe(df_1_1)

            df_1_2 = preprocess_pipeline_new_pt_1_2(df_1_1, dims["popular_by_gram"])
            df_1_2 = df_1_2[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'GRAM_GROUP', 'Antimicrobial Class', 'Month']]
            y_pred_sens = models["rs_1"].predict(df_1_2)
            prob_sens = models["rs_1"].predict_proba(df_1_2)

            df_1_2['SENSITIVITY_INTERPRETION'] = y_pred_sens
            df_1_2['PROB_SENSITIVE'] = np.round(prob_sens[:, 1], 2)
            df_1_2 = decode_and_overwrite_1_2(df_1_2)

            df_show = df_1_2[df_1_2["SENSITIVITY_INTERPRETION"] == 1]
            df_show = df_show[['GRAM_GROUP', 'Antimicrobial Class', 'PROB_SENSITIVE']].sort_values(by="PROB_SENSITIVE", ascending=False)

            st.subheader("💊 Kháng sinh có khả năng nhạy:")
            df_show = df_show.reset_index(drop=True)  # loại bỏ index cũ
            df_show.columns = [f"col_{i}" if col == '' else str(col) for i, col in enumerate(df_show.columns)]  # đảm bảo tên cột là duy nhất
            st.dataframe(df_show.style.background_gradient(cmap='Greens', subset=['PROB_SENSITIVE']))

        else:  # Đã có vi sinh vật
            new_patient["ISOLATE_ORGANISM"] = isolated_organism
            df_2 = pd.DataFrame([new_patient])
            df_encoded = preprocess_pipeline_new_pt(df_2, dims["popular_by_organism"])
            df_encoded = df_encoded[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC', 'Month']]
            y_pred = models["rs_2"].predict(df_encoded)
            prob = models["rs_2"].predict_proba(df_encoded)

            df_encoded['SENSITIVITY_INTERPRETION'] = y_pred
            df_encoded['PROB_SENSITIVE'] = np.round(prob[:, 1], 2)
            df_encoded = decode_and_overwrite_2(df_encoded)

            df_show = df_encoded[df_encoded["SENSITIVITY_INTERPRETION"] == 1]
            df_show = df_show[['ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC', 'PROB_SENSITIVE']]
            df_show = df_show.sort_values(by=["ISOLATE_ORGANISM", "PROB_SENSITIVE"], ascending=[True, False])

            st.subheader("📋 Kết quả kháng sinh phù hợp:")
            df_show = df_show.reset_index(drop=True)
            df_show.columns = [f"col_{i}" if col == '' else str(col) for i, col in enumerate(df_show.columns)]  # hoặc đảm bảo tên cột hợp lệ
            st.dataframe(df_show.style.applymap(highlight_broad_spectrum, subset=["SENSITIVITY_ANTIBIOTIC"]))

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")

