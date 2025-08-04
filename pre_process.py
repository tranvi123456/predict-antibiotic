import subprocess
subprocess.check_call(["pip", "install", "catboost"])
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import load
import os
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
# import shap
# import matplotlib.pyplot as plt
# from xgboost import plot_importance

def classify_organism_name(organism):
  gram_negative = [
    'Escherichia coli',
    'Pseudomonas aeruginosa',
    'Gram-negative bacilli (identification in process)',
    'Enterobacter cloacae',
    'Klebsiella pneumoniae',
    'Haemophilus parainfluenzae',
    'Haemophilus influenzae',
    'Acinetobacter baumannii',
    'Vibrio parahaemolyticus',
    'Acinetobacter baumannii complex',
    'Klebsiella pneumoniae ssp pneumoniae',
    'Bacteroides fragilis',
    'Stenotrophomonas maltophilia',
    'Moraxella catarrhalis (Branhamella catarrhalis)',
    'Neisseria gonorrhoeae',
    'Klebsiella aerogenes',
    'Achromobacter denitrificans',
    'Proteus mirabilis',
    'Proteus penneri',
    'Citrobacter freundii',
    'Citrobacter koserii',
    'Aeromonas punctata (caviae)',
    'Salmonella spp',
    'Salmonella enterica ssp. enterica',
    'Salmonella sp',
    'Citrobacter braakii',
    'Chryseobacterium indologenes',
    'Acinetobacter haemolyticus',
    'Enterobacter cloacae complex',
    'Serratia marcescens',
    'Morganella morganii',
    'Klebsiella oxytoca',
    'Burkholderia cepacia',
    'Providencia stuartii',
    'Sphingomonas paucimobilis',
    'Pseudomonas putida',
    'Comamonas testosteroni',
    'Providencia rettgeri',
    'Pantoea sp',
    'Proteus vulgaris',
    'Kluyvera cryocrescens',
    'Pseudomonas stutzeri',
    'Aeromonas hydrophila/punctata(caviae)',
    'Edwardsiella tarda',
    'Acinetobacter pittii',
    'Burkholderia gladioli',
    'Pantoea dispersa',
    'Pseudomonas mendocina',
    'Citrobacter amalonaticus'
  ]
  gram_positive = [
    'Streptococcus group B',
    'Streptococcus agalactiae',
    'Staphylococcus aureus',
    'Streptococcus pyogenes (Streptococcus group A)',
    'Staphylococcus capitis',
    'Staphylococcus hominis ssp hominis',
    'Staphylococcus hominis',
    'Staphylococcus epidermidis',
    'Streptococcus uberis',
    'Enterococcus gallinarum',
    'Enterococcus faecium',
    'Enterococcus faecalis',
    'Staphylococcus haemolyticus',
    'Streptococcus anginosus',
    'Streptococcus vestibularis',
    'Streptococcus parasanguinis',
    'Enterococcus mundtii',
    'Enterococcus hirae',
    'Streptococcus suis',
    'Streptococcus constellatus',
    'Enterococcus casseliflavus',
    'Staphylococcus sciuri',
    'Streptococcus dysgalactiae',
    'Corynebacterium striatum',
    'Bacillus clausii',
    'Streptococcus gallolyticus ssp gallolyticus',
    'Streptococcus mitis/Streptococcus oralis',
    'Staphylococcus ureilyticus',
    'Streptococcus intermedius',
    'Staphylococcus lugdunensis',
    'Streptococcus pneumoniae',
    'Lactobacillus salivarius',
    'Enterococcus avium',
    'Streptococcus pyogenes',
    'Coagulase negative Staphylococcus',
    'Bacillus cereus group',
    'Corynebacterium sp',
    'Streptococcus gallolyticus ssp pasteurianus'
]
  fungi = [
    'Candida albicans',
    'Candida tropicalis',
    'Candida glabrata',
    'Candida krusei',
    'Candida dubliniensis',
    'Candida parapsilosis'
]

  if organism in gram_negative:
      return 'Gram-negative'
  elif organism in gram_positive:
      return 'Gram-positive'
  elif organism in fungi:
      return 'Fungi'
  else:
      return 'Unknown'

def classify_organism(data):
  data['GRAM_GROUP'] = data['ISOLATE_ORGANISM'].apply(classify_organism_name)
  return data

def modify_antibiotic_column(data):
  data['SENSITIVITY_ANTIBIOTIC'] = data['SENSITIVITY_ANTIBIOTIC'].str.split('(').str[0].str.strip()
  data['SENSITIVITY_ANTIBIOTIC'] = data['SENSITIVITY_ANTIBIOTIC'].str.replace(r"\s*[+/\t]\s*", ";", regex=True)
  data['SENSITIVITY_ANTIBIOTIC'] = data['SENSITIVITY_ANTIBIOTIC'].str.lower()
  return data

def merge_atb_agent (data):
  dt_atb_grp = pd.read_excel("Antimicrobial Agents.xlsx")
  dt_atb_grp['FV_atb_modified'] = dt_atb_grp['FVH Antimicrobial Name'].str.replace(r"\s*[+/\-]\s*", ";", regex = True).str.lower()
  return data.merge(dt_atb_grp, left_on = 'SENSITIVITY_ANTIBIOTIC', right_on = 'FV_atb_modified', how='left')

def merge_unique_prb (data):
  unq_prb = pd.read_excel('unique_problems.xlsx')
  return data.merge(unq_prb, left_on='PROBLEM', right_on='unique_problem', how='left')

def age_columns(data):
  data['date_of_birth'] = pd.to_datetime(data['DATE_OF_BIRTH'], errors='coerce')
  today = datetime.today()
  data['AGE'] = data['date_of_birth'].apply(lambda x: today.year -x.year - ((today.month, today.day) <(x.month, x.day)))
  return data

def filter_sensitive_atb (data):
  data = data[data['SENSITIVITY_INTERPRETION'].isin (['S', 'R', 'I'])]
  data['SENSITIVITY_INTERPRETION'] = data['SENSITIVITY_INTERPRETION'].replace({
    'S': 1,
    'I': 1,
    'R': 0
  })
  return data

def select_columns_df1 (data):
  return data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem','SPECIMEN_SOURCE', 'GRAM_GROUP', 'Antimicrobial Class', 'SENSITIVITY_INTERPRETION', 'Month']]
def select_columns_df2 (data):
  return data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem','SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC','SENSITIVITY_INTERPRETION', 'Month']]

def remove_duplicates(data):
  return data.drop_duplicates()

def encode_columns_df1(data):
    """Mã hóa các cột bằng LabelEncoder."""
    os.makedirs('timeout_encoders.3', exist_ok=True)
    columns_to_encode = ['SEX_RCD', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'GRAM_GROUP', 'Antimicrobial Class']
    # Fit và lưu từng encoder
    for col in columns_to_encode:
        label_encoder = LabelEncoder()
        data[col] = data[col].astype(str)  # đảm bảo kiểu string
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])
        joblib.dump(label_encoder, f'timeout_encoders.3/{col}_encoder.pkl')
    return data


def encode_columns_df2(data):
    """Mã hóa các cột bằng LabelEncoder."""
    os.makedirs('timeout_encoders.3', exist_ok=True)
    columns_to_encode = ['SEX_RCD', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem',  'SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC','SENSITIVITY_INTERPRETION']
    # Fit và lưu từng encoder
    for col in columns_to_encode:
        label_encoder = LabelEncoder()
        data[col] = data[col].astype(str)  # đảm bảo kiểu string
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])
        joblib.dump(label_encoder, f'timeout_encoders.3/{col}_encoder.pkl')
    return data



def preprocess_pipeline_df(df):
    df = classify_organism(df)
    df = modify_antibiotic_column(df)
    df = merge_atb_agent(df)
    df = merge_unique_prb(df)
    df = age_columns(df)
    df = filter_sensitive_atb(df)
    return df


def preprocess_pipeline_df1(df):
    df = select_columns_df1(df)
    df = remove_duplicates(df)
    df = encode_columns_df1(df)
    return df

def xgb_model_1_1(data):
  X = data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'Month']]
  y = data["GRAM_GROUP"]
  # Tạo mô hình với tham số tốt nhất
  model = XGBClassifier(
    learning_rate=0.1,
    max_depth=7,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric='logloss'
    )
  # Huấn luyện model
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  model.fit(X_train, y_train)
  # lưu model
  joblib.dump(model,'xgb_model_gram_group.pkl')
  # dự đoán
  y_pred = model.predict(X_test)
  print("✅ Accuracy:", accuracy_score(y_test, y_pred))
  print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
  print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  # Gộp kết quả dự đoán với dữ liệu test
  results = X_test.copy()
  results = results.reset_index(drop=True)
  y_test = y_test.reset_index(drop=True)

  results['Actual'] = y_test
  results['Predicted'] = y_pred

  # plot_importance(model, max_num_features=10)
  # plt.show()

  return model, results

def voting_model_1_2(data):
  X1 = data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'GRAM_GROUP', 'Antimicrobial Class', 'Month']]
  y1 = data['SENSITIVITY_INTERPRETION']

  X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

  # Huấn luyện từng model
  xgb_model = XGBClassifier()
  xgb_model.fit(X_train1, y_train1)

  cat_model = CatBoostClassifier(verbose=0)
  cat_model.fit(X_train1, y_train1)

  lgbm_model = LGBMClassifier()
  lgbm_model.fit(X_train1, y_train1)
  # Huấn luyện model
  model1 = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'  # dùng xác suất để voting
)

  model1.fit(X_train1, y_train1)

  # lưu model
  joblib.dump(model1, 'voting_model_rs_1.pkl')
  # dự đoán
  y_pred1 = model1.predict(X_test1)
  print("✅ Accuracy:", accuracy_score(y_test1, y_pred1))
  print("\n📊 Classification Report:\n", classification_report(y_test1, y_pred1))
  print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test1, y_pred1))
  # Gộp kết quả dự đoán với dữ liệu test
  results = X_test1.copy()
  results = results.reset_index(drop=True)
  y_test1 = y_test1.reset_index(drop=True)

  results['Actual'] = y_test1
  results['Predicted'] = y_pred1


  # explainer = shap.Explainer(xgb_model)
  # shap_values = explainer(X_train1)

  # shap.plots.beeswarm(shap_values)
  # return model1, results

def preprocess_pipeline_df2(df):
    df = select_columns_df2(df)
    df = remove_duplicates(df)
    df = encode_columns_df2(df)
    return df

def xgb_model_2(data):
  X2 = data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC', 'Month']]
  y2 = data['SENSITIVITY_INTERPRETION']
  # Tạo mô hình với tham số tốt nhất
  model2 = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric='logloss'
)
  # Huấn luyện model
  X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
  model2.fit(X_train2, y_train2)
  # lưu model
  joblib.dump(model2, 'xgb_model_rs_2.pkl')
  # dự đoán
  y_pred2 = model2.predict(X_test2)
  print("✅ Accuracy:", accuracy_score(y_test2, y_pred2))
  print("\n📊 Classification Report:\n", classification_report(y_test2, y_pred2))
  print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test2, y_pred2))
  # Gộp kết quả dự đoán với dữ liệu test
  results = X_test2.copy()
  results = results.reset_index(drop=True)
  y_test = y_test2.reset_index(drop=True)

  results['Actual'] = y_test2
  results['Predicted'] = y_pred2

  model2.fit(X_train2, y_train2)

  # plot_importance(model2, max_num_features=10)
  # plt.show()
  return model2, results

def age_columns_new_pt (data):
    data['date_of_birth'] = pd.to_datetime(data['DATE_OF_BIRTH'], errors='coerce')
    today = datetime.today()
    dob = data.at[0, 'date_of_birth']
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    data['AGE'] = age
    return data

def month_order (data):
    data['ORDERED_DATE_TIME'] = pd.to_datetime(data['ORDERED_DATE_TIME'], errors='coerce')
    data['Month'] = data['ORDERED_DATE_TIME'].dt.month
    return data



def merge_unique_prb (data):
  unq_prb = pd.read_excel('unique_problems.xlsx')
  return data.merge(unq_prb, left_on='PROBLEM', right_on='unique_problem', how='left')

def select_columns_1_1(data):
    return data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'Month']]



def select_columns_2(data):
    return data[['SEX_RCD', 'AGE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED', 'general_problem', 'SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'Month']]



def add_antimicrobial_class(data, popular_by_gram):
    expanded_rows = []

    for idx, row in data.iterrows():
        gram_group = row['GRAM_GROUP']
        # Lọc các NHÓM kháng sinh phổ biến cho NHÓM organism
        antibiotics_group = popular_by_gram[popular_by_gram['GRAM_GROUP'] == gram_group]['Antimicrobial Class'].unique()

        for abx in antibiotics_group:
            new_row = row.copy()
            new_row['Antimicrobial Class'] = abx
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)






def add_sensitive_antibiotic(data, popular_by_organism):
    expanded_rows = []

    for idx, row in data.iterrows():
        organism = row['ISOLATE_ORGANISM']
        # Lọc các kháng sinh phổ biến cho organism
        antibiotics = popular_by_organism[popular_by_organism['ISOLATE_ORGANISM'] == organism]['SENSITIVITY_ANTIBIOTIC'].unique()

        for abx in antibiotics:
            new_row = row.copy()
            new_row['SENSITIVITY_ANTIBIOTIC'] = abx
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


def load_and_encode_1_1(data):
    columns_to_encode = ['SEX_RCD', 'general_problem', 'SPECIMEN_SOURCE', 'ORDER_OWNER.1', 'DEPARTMENT_ORDERED']

    for col in columns_to_encode:
        encoder_path = f'timeout_encoders.3/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.transform(data[col].astype(str))
        else:
            raise FileNotFoundError(f"Encoder for column {col} not found at {encoder_path}")

    return data



def load_and_encode_1_2(data):
    columns_to_encode = ['SEX_RCD', 'general_problem', 'SPECIMEN_SOURCE', 'GRAM_GROUP', 'Antimicrobial Class',
                         'ORDER_OWNER.1', 'DEPARTMENT_ORDERED']

    for col in columns_to_encode:
        encoder_path = f'timeout_encoders.3/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.transform(data[col].astype(str))
        else:
            raise FileNotFoundError(f"Encoder for column {col} not found at {encoder_path}")

    return data



def load_and_encode_2(data):
    columns_to_encode = ['SEX_RCD', 'general_problem', 'SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC',
                         'ORDER_OWNER.1', 'DEPARTMENT_ORDERED']

    for col in columns_to_encode:
        encoder_path = f'timeout_encoders.3/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.transform(data[col].astype(str))
        else:
            raise FileNotFoundError(f"Encoder for column {col} not found at {encoder_path}")

    return data


def decode_and_overwrite_1_1(data):
    columns_to_decode = ['SEX_RCD', 'general_problem', 'SPECIMEN_SOURCE',
                         'ORDER_OWNER.1', 'DEPARTMENT_ORDERED']

    for col in columns_to_decode:
        encoder_path = f'timeout_encoders.3/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.inverse_transform(data[col])
        else:
            print(f"⚠️ Không tìm thấy encoder cho {col}")

    return data


def decode_and_overwrite_1_2(data):
    columns_to_decode = ['SEX_RCD', 'general_problem', 'SPECIMEN_SOURCE', 'GRAM_GROUP', 'Antimicrobial Class',
                         'ORDER_OWNER.1', 'DEPARTMENT_ORDERED']

    for col in columns_to_decode:
        encoder_path = f'timeout_encoders.3/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.inverse_transform(data[col])
        else:
            print(f"⚠️ Không tìm thấy encoder cho {col}")

    return data



def decode_and_overwrite_2(data):
    columns_to_decode = ['SEX_RCD', 'general_problem', 'SPECIMEN_SOURCE', 'ISOLATE_ORGANISM', 'SENSITIVITY_ANTIBIOTIC',
                         'ORDER_OWNER.1', 'DEPARTMENT_ORDERED']

    for col in columns_to_decode:
        encoder_path = f'timeout_encoders.3/{col}_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            data[col] = label_encoder.inverse_transform(data[col])
        else:
            print(f"⚠️ Không tìm thấy encoder cho {col}")

    return data


# Hàm tô màu
def highlight_broad_spectrum(atb):
  broad_spectrum_antibiotics = [
    # Restricted Antibiotics
    'amikacin', 'amikacin sulfate',  'tobramycin', 'imipenem;cilastatin', 'imipenem;cilastatin;relebactam', 'meropenem', 'ertapenem',
    'cefepime', 'ceftazidime;avibactam', 'ceftolozane;tazobactam', 'colistin', 'fosfomycin', 'tigecycline', 'vancomycin',
    'teicoplanin', 'linezolid', 'levofloxacin', 'ciprofloxacin', 'moxifloxacin', 'aztreonam', 'ceftaroline'

    # Restricted Antifungals
    'caspofungin', 'micafungin', 'anidulafungin', 'amphotericin', 'voriconazole'
]
  if atb in broad_spectrum_antibiotics:
    return 'background-color: yellow; font-weight: bold'
  else:
    return ''
  
def preprocess_pipeline_new_pt_1_1(data):
    data = age_columns_new_pt(data)
    data = month_order(data)
    data = merge_unique_prb(data)
    data = select_columns_1_1(data)
    data = load_and_encode_1_1(data)
    return data

def preprocess_pipeline_new_pt_1_2(data, popular_by_gram):
    data = add_antimicrobial_class (data, popular_by_gram)
    data = load_and_encode_1_2(data)
    return data

def preprocess_pipeline_new_pt(data, popular_by_organism):
    data = age_columns_new_pt(data)
    data = month_order (data)
    data = merge_unique_prb(data)
    data = select_columns_2(data)
    data = add_sensitive_antibiotic (data, popular_by_organism)
    data = load_and_encode_2(data)
    return data

