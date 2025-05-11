import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import load
import os


def modify_antibiotic_column(data):
    data['SENSITIVITY_ANTIBIOTIC'] = data['SENSITIVITY_ANTIBIOTIC'].str.split('(').str[0].str.strip()
    data['SENSITIVITY_ANTIBIOTIC'] = data['SENSITIVITY_ANTIBIOTIC'].str.replace(r"\s*[+/\t]\s*", ";", regex=True)
    data['SENSITIVITY_ANTIBIOTIC'] = data['SENSITIVITY_ANTIBIOTIC'].str.lower()
    return data

def age_columns(data):
    data['date_of_birth'] = pd.to_datetime(data['DATE_OF_BIRTH'], errors='coerce')
    today = datetime.today()
    data['AGE'] = data['date_of_birth'].apply(lambda x: today.year -x.year - ((today.month, today.day) <(x.month, x.day)))
    return data

def select_columns_1(data):
    return data[['VISIBLE_PATIENT_ID', 'SEX_RCD', 'AGE', 'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED', 'PROBLEM', 'SENSITIVITY_ANTIBIOTIC', 'SENSITIVITY_INTERPRETION']]

def filter_sensitivity(data):
    return data[data['SENSITIVITY_INTERPRETION'].isin (['R', 'S', 'I'])]

def replace_sensitivity(data):
    data['SENSITIVITY_INTERPRETION'] = data['SENSITIVITY_INTERPRETION'].replace({
    'S': 1,
    'I': 1,
    'R': 0
})
    return data

def remove_duplicates(data):
  return data.drop_duplicates()

def merge_unique_prb (data):
  unq_prb = pd.read_excel('unique_problems.xlsx')
  return data.merge(unq_prb, left_on='PROBLEM', right_on='unique_problem', how='left')

def select_columns_2(data):
    return data[['VISIBLE_PATIENT_ID', 'SEX_RCD', 'AGE', 'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED', 'general_problem','SENSITIVITY_ANTIBIOTIC', 'SENSITIVITY_INTERPRETION']]

def encode_columns(data):
    """Mã hóa các cột bằng LabelEncoder."""
    os.makedirs('encoders', exist_ok=True)
    columns_to_encode = ['SEX_RCD', 'general_problem', 'SENSITIVITY_ANTIBIOTIC', 'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED']
    # Fit và lưu từng encoder
    for col in columns_to_encode:
        label_encoder = LabelEncoder()
        data[col] = data[col].astype(str)  # đảm bảo kiểu string
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])
        joblib.dump(label_encoder, f'encoders/{col}_encoder.pkl')
    return data



# 🧩 Pipeline xử lý
def preprocess_pipeline(df):
    df = modify_antibiotic_column(df)
    df = age_columns(df)
    df = select_columns_1(df)
    df = filter_sensitivity(df)
    df = replace_sensitivity(df)
    df = remove_duplicates(df)
    df = merge_unique_prb(df)
    df = select_columns_2(df)
    df = encode_columns(df)
    return df


def xgb_model (data):
    X = data[['VISIBLE_PATIENT_ID', 'SEX_RCD', 'AGE',
       'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED', 'general_problem','SENSITIVITY_ANTIBIOTIC']]
    y = data[["SENSITIVITY_INTERPRETION"]]

    # Tạo mô hình với tham số tốt nhất
    best_xgb = XGBClassifier(
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric='logloss'
    )

    # Huấn luyện mô hình
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_xgb.fit(X_train, y_train)

    # 3. Lưu mô hình vào file
    joblib.dump(best_xgb, 'xgb_model.pkl')

    # Dự đoán
    y_pred = best_xgb.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

     # Gộp kết quả dự đoán với dữ liệu test
    results = X_test.copy()
    results = results.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    results['Actual'] = y_test
    results['Predicted'] = y_pred

    return best_xgb, results



def age_columns_new_pt (data):
    data['date_of_birth'] = pd.to_datetime(data['DATE_OF_BIRTH'], errors='coerce')
    today = datetime.today()
    dob = data.at[0, 'date_of_birth']
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    data['AGE'] = age
    return data

def select_columns(data):
    return data[['VISIBLE_PATIENT_ID', 'SEX_RCD', 'AGE', 'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED', 'general_problem']]

def add_antibiotic_column(data, all_antibiotics):
    """ Thêm cột kháng sinh. """
    data = data.loc[data.index.repeat(len(all_antibiotics))].copy()
    # Reset index để gán lại chính xác
    data.reset_index(drop=True, inplace=True)
    data['SENSITIVITY_ANTIBIOTIC'] = all_antibiotics
    return data

def load_and_encode(data):
    columns_to_encode = ['SEX_RCD', 'general_problem', 'SENSITIVITY_ANTIBIOTIC',
                         'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED']

    for col in columns_to_encode:
        encoder_path = f'encoders/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.transform(data[col].astype(str))
        else:
            raise FileNotFoundError(f"Encoder for column {col} not found at {encoder_path}")

    return data



# thiếu 'ceftazidime;avibactam', 'ampicillin;sulbactam
def preprocess_pipeline_new_pt(data):
    all_antibiotics = ['ampicillin', 'amoxicillin sodium;clavulanic acid',
       'piperacillin;tazobactam', 'cefazolin', 'cefotaxime', 'cefepime',
       'ertapenem', 'imipenem', 'meropenem', 'amikacin sulfate',
       'gentamicin', 'tobramycin', 'ciprofloxacin', 'nitrofurantoin',
       'trimethoprim;sulfamethoxazole', 'fosfomycin', 'ceftazidime', 'fluconazole',
       'voriconazole', 'caspofungin', 'micafungin', 'amphotericin b',
       'benzylpenicillin', 'oxacillin', 'erythromycin', 'clindamycin',
       'quinupristin;dalfopristin', 'linezolid', 'vancomycin',
       'tetracycline', 'tigecycline', 'rifampicin', 'azithromycin',
       'aztreonam', 'ceftriaxone', 'streptomycin', 'levofloxacin',
       'moxifloxacin', 'chloramphenicol', 'cefoxitin', 'ticarcillin',
       'ticarcillin;clavulanic acid', 'piperacillin', 'colistin',
       'ceftolozane;tazobactam']
    data = age_columns_new_pt(data)
    data = merge_unique_prb(data)
    data = select_columns(data)
    data = add_antibiotic_column(data, all_antibiotics)
    data = load_and_encode(data)
    return data



def decode_and_overwrite(data):
    columns_to_decode = ['SEX_RCD', 'general_problem', 'SENSITIVITY_ANTIBIOTIC',
                         'ORDER_OWNER', 'LOCATION', 'DEPARTMENT_ORDERED']

    for col in columns_to_decode:
        encoder_path = f'encoders/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.inverse_transform(data[col])
        else:
            print(f"⚠️ Không tìm thấy encoder cho {col}")

    return data



