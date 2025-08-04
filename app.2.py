import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import date

from pre_process import (
    preprocess_pipeline_df,
    preprocess_pipeline_df1,
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

# Giao diện chính
st.title("🔬 Antibiotic Susceptibility Prediction")

# Sidebar bên trái
with st.sidebar:
    st.header("Thông tin chung")
    user_choice = st.radio("📌 Bạn đã có kết quả định danh vi sinh vật chưa?", ["Chưa có", "Đã có"])

if user_choice == "Chưa có":
    st.subheader("📋 Nhập thông tin bệnh nhân (Chưa có kết quả vi sinh vật)")
    
    with st.form("form_no_organism"):
        visible_patient_id = st.number_input("🆔 Mã bệnh nhân (HN):")
        sex_rcd = st.selectbox("👤 Giới tính:", ["M", "F"])
        dob = st.date_input("🎂 Ngày sinh:", min_value=date(1920, 1, 1), value=date(2000, 1, 1), max_value=date.today())
        order_owner = st.selectbox("👨‍⚕️ Bác sĩ chỉ định:", dims["doctor_list"])
        specimen_source = st.selectbox("🧪 Nguồn mẫu:", dims["source_list"])
        department_ordered = st.selectbox("🏥 Khoa chỉ định:", dims["department_list"])
        problem = st.text_input("🦠 Chẩn đoán lâm sàng:")
        ordered_date_time = st.date_input("📆 Ngày gửi mẫu:", min_value=date(2024, 1, 1), max_value=date.today())
        submitted = st.form_submit_button("🚀 Dự đoán")

    if submitted:
        try:
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

            # 👉 Phần xử lý dự đoán Gram và kháng sinh
            df_1 = pd.DataFrame([new_patient])
            df_1_1 = preprocess_pipeline_new_pt_1_1(df_1)
            y_pred_gram = models["gram_group"].predict(df_1_1)
            df_1_1['GRAM_GROUP'] = models["gram_encoder"].inverse_transform(y_pred_gram)
            df_1_1 = decode_and_overwrite_1_1(df_1_1)

            st.success("✅ Dự đoán nhóm Gram thành công!")
            st.dataframe(df_1_1)

            df_1_2 = preprocess_pipeline_new_pt_1_2(df_1_1, dims["popular_by_gram"])
            y_pred_sens = models["rs_1"].predict(df_1_2)
            prob_sens = models["rs_1"].predict_proba(df_1_2)

            df_1_2['SENSITIVITY_INTERPRETION'] = y_pred_sens
            df_1_2['PROB_SENSITIVE'] = np.round(prob_sens[:, 1], 2)
            df_1_2 = decode_and_overwrite_1_2(df_1_2)

            df_show = df_1_2[df_1_2["SENSITIVITY_INTERPRETION"] == 1]
            df_show = df_show[['GRAM_GROUP', 'Antimicrobial Class', 'PROB_SENSITIVE']].sort_values(by="PROB_SENSITIVE", ascending=False)
            df_show = df_show.reset_index(drop=True)

            st.subheader("💊 Kháng sinh nhạy:")
            st.dataframe(df_show.style.background_gradient(cmap='Greens', subset=['PROB_SENSITIVE']))
        
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

elif user_choice == "Đã có":
    st.subheader("📋 Bạn có bao nhiêu vi sinh vật phân lập?")
    num_micro = st.radio("🔬 Số lượng vi sinh vật:", ["1", "Nhiều"])

    if num_micro == "1":
        with st.form("form_one_micro"):
            visible_patient_id = st.number_input("🆔 Mã bệnh nhân:")
            sex_rcd = st.selectbox("👤 Giới tính:", ["M", "F"])
            dob = st.date_input("🎂 Ngày sinh:", min_value=date(1920, 1, 1), value=date(2000, 1, 1), max_value=date.today())
            order_owner = st.selectbox("👨‍⚕️ Bác sĩ:", dims["doctor_list"])
            specimen_source = st.selectbox("🧪 Mẫu:", dims["source_list"])
            department_ordered = st.selectbox("🏥 Khoa:", dims["department_list"])
            problem = st.text_input("🦠 Chẩn đoán:")
            ordered_date_time = st.date_input("📆 Ngày gửi mẫu:")
            isolated_organism = st.selectbox("🔍 Vi sinh vật:", dims["organism_list"])
            submitted_2 = st.form_submit_button("🚀 Dự đoán")

        if submitted_2:
            try:
                new_patient = {
                    "VISIBLE_PATIENT_ID": visible_patient_id,
                    "SEX_RCD": sex_rcd,
                    "DATE_OF_BIRTH": dob,
                    "ORDER_OWNER.1": order_owner,
                    "SPECIMEN_SOURCE": specimen_source,
                    "DEPARTMENT_ORDERED": department_ordered,
                    "PROBLEM": problem,
                    "ORDERED_DATE_TIME": ordered_date_time,
                    "ISOLATE_ORGANISM": isolated_organism
                }

                df_2 = pd.DataFrame([new_patient])
                df_encoded = preprocess_pipeline_new_pt(df_2, dims["popular_by_organism"])
                y_pred = models["rs_2"].predict(df_encoded)
                prob = models["rs_2"].predict_proba(df_encoded)

                df_encoded['SENSITIVITY_INTERPRETION'] = y_pred
                df_encoded['PROB_SENSITIVE'] = np.round(prob[:, 1], 2)
                df_encoded = decode_and_overwrite_2(df_encoded)

                df_show = df_encoded[df_encoded["SENSITIVITY_INTERPRETION"] == 1]
                df_show = df_show[['ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC', 'PROB_SENSITIVE']].sort_values(by=["ISOLATE_ORGANISM", "PROB_SENSITIVE"], ascending=[True, False])

                st.subheader("📋 Kháng sinh phù hợp:")
                st.dataframe(df_show.style.applymap(highlight_broad_spectrum, subset=["SENSITIVITY_ANTIBIOTIC"]))

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    else:  # Trường hợp "Nhiều"
        st.warning("🔢 Nhiều mẫu vi sinh vật được phân lập. Hãy nhập danh sách.")

        # 👇 Bạn có thể tạm nhập cứng hoặc dùng upload file:
        multi_micro = [
            {
                'ISOLATE_ORGANISM': 'Klebsiella pneumoniae',
                'ORDER_OWNER.1': 'Dr. Vu Ngoc Chuc',
                'DEPARTMENT_ORDERED': 'Medical Ward East ',
                'PROBLEM': '',
                'SPECIMEN_SOURCE': 'Set 1 Aerobic Bottle',
                'ORDERED_DATE_TIME': '3/23/2024'
            },
            {
                'ISOLATE_ORGANISM': 'Enterococcus faecalis',
                'ORDER_OWNER.1': 'Dr. Ho An Toan',
                'DEPARTMENT_ORDERED': 'ICU Department',
                'PROBLEM': 'pneumonia',
                'SPECIMEN_SOURCE': 'Fluid',
                'ORDERED_DATE_TIME': '4/22/2024'
            }
        ]

        # Thông tin chung
        common_info = {
            "VISIBLE_PATIENT_ID": 123456789,
            "SEX_RCD": "M",
            "DATE_OF_BIRTH": "1/4/1954"
        }

        multi_df = pd.DataFrame([{**common_info, **item} for item in multi_micro])
        st.dataframe(multi_df)

        st.info("📈 Bạn có thể xử lý thêm tại đây bằng mô hình nếu muốn.")


