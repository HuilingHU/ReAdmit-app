# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import torch
import joblib
import requests
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from transformers import AutoTokenizer, AutoModel
from paddleocr import PaddleOCR
import os
import requests

ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# ------------------ 页面设置 ------------------
st.set_page_config(page_title="再入ICU风险预测工具 - ReAdmit", layout="wide")
st.markdown("""
<style>
/* 页面整体字体大小和间距 */
body, .css-18e3th9, .stApp {
    font-size: 0.9rem;  /* 调整整体字体 */
    line-height: 1.2;
}

/* 标题字体 */
h1, h2, h3, h4, h5, h6 {
    font-size: 1.2rem;
}

/* 输入框和按钮字体 */
.stTextInput>div>input, .stNumberInput>div>input, .stButton>button, .stSelectbox>div>div {
    font-size: 0.9rem;
    padding: 0.25rem 0.4rem;
}

/* 表格字体 */
.stTable td, .stTable th {
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ 模型加载 ------------------
@st.cache_resource
def load_models():
    base_path = os.getcwd()
    model = joblib.load(os.path.join(base_path, "model_0922.joblib"))
    with open(os.path.join(base_path, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(base_path, "threshold_0922.txt"), "r") as f:
        threshold = float(f.read().strip())
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return model, pca, threshold, tokenizer, bert_model

model, pca, threshold, tokenizer, bert_model = load_models()
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# ------------------ LLM 在线调用 ------------------
def ask_deepseek_online(prompt):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",  # 在部署环境配置API Key
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",   # 或 deepseek-reasoner，根据你需要的模型
        "messages": [
            {"role": "system", "content": "你是一个医学助手，提供风险解读和临床建议。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM调用失败：{e}"

# ------------------ 文本预处理 ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s:=\.\(\)/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_embeddings(text):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings

# ------------------ OCR 提取文本并标准化 ------------------
def extract_text_from_image(image_file):
    with open("temp_report.png", "wb") as f:
        f.write(image_file.getvalue())
    result = ocr.ocr("temp_report.png", cls=True)
    full_text = " ".join([line[1][0] for line in result[0]])

    # ---- 文本标准化 ----
    full_text = full_text.upper()
    full_text = re.sub(r"\s+", " ", full_text).strip()
    full_text = full_text.replace("⁺", "+").replace("－", "-").replace("–", "-")
    full_text = full_text.replace("０","0").replace("１","1").replace("２","2").replace("３","3")
    full_text = full_text.replace("４","4").replace("５","5").replace("６","6").replace("７","7")
    full_text = full_text.replace("８","8").replace("９","9")
    full_text = full_text.replace("HC03", "HCO3")

    # ---- 去除参考值片段 ----
    full_text = re.sub(r"参考值?\s*[:：]?\s*-?\d+\.?\d*\s*-\s*\d+\.?\d*", "", full_text)
    full_text = re.sub(r"\(\s*-?\d+\.?\d*\s*-\s*\d+\.?\d*\s*\)", "", full_text)

    return full_text

# ------------------ 检验指标提取 ------------------
def extract_lab_values(text):
    alias_map = {
       "wbc": ["白细胞", "WBC"], "rbc": ["红细胞", "RBC"],
        "hemoglobin": ["血红蛋白", "HGB"], "hematocrit": ["红细胞压积", "HCT"],
        "mch": ["平均血红蛋白含量", "MCH"], "platelet": ["血小板", "PLT"],
        "rdw": ["红细胞分布宽度CV", "RDW_CV"], "alt": ["丙氨酸氨基转移酶", "ALT"],
        "ast": ["天冬氨酸氨基转移酶", "AST"], "albumin": ["白蛋白", "ALB"],
        "bilirubin_total": ["总胆红素", "TBIL", "T-BIL"], "creatinine": ["肌酐", "CR", "CREA"],
        "sodium": ["钠", "NA\\+", "钠离子", "Na\\+?"], "potassium": ["钾", "K\\+?", "钾离子"],
        "chloride": ["氯", "CL-?", "氯离子", "Cl\\-", "CL"], "calcium": ["钙", "CA\\+\\+?", "CA2\\+", "Ca"],
        "bicarbonate": ["HCO3-", "碳酸氢根", "碳酸氢盐", "HCO3-（std）"], "glucose": ["葡萄糖", "GLU", "Glu"],
        "lactate": ["乳酸", "LAC", "Lac"], "ph": ["PH"], "be": ["碱剩余", "SBE", "BE"],
        "pao2": ["氧分压", "PO2", "PAO2"], "paco2": ["二氧化碳分压", "PACO2", "PCO2"],
        "inr": ["国际标准比率", "INR"], "pt": ["PT", "凝血酶原时间"], "ptt": ["APTT", "活化部分凝血活酶"]
    }
    results = {}
    for standard_name, aliases in alias_map.items():
        for alias in aliases:
            pattern = rf"{alias}[^\d\-]{{0,6}}(-?\d+(?:\.\d+)?)(?!\s*-\s*\d)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[standard_name] = float(match.group(1))
                break
    return results

# ------------------ Charlson 指数 ------------------
def calculate_charlson_score(age, selections):
    score = 0
    weights = {"group1": 1, "group2": 2, "group3": 3, "group4": 6}
    for group, items in selections.items():
        score += weights[group] * len(items)
    if age >= 40:
        score += ((age - 40) // 10) + 1
    return score

# ------------------ SHAP 解释 ------------------
def predict_with_shap(model, structured_array, bert_pca_vec):
    X = np.hstack([structured_array, bert_pca_vec])
    prob = float(model.predict_proba(X)[0][1])

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)   # 这里传入单个样本也可以
        vals = shap_values.values[0]
        
        # 如果没有 feature_names，就自己生成
        feature_names = shap_values.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        idx = np.argsort(-np.abs(vals))[:10]
        top_features = [(feature_names[i], float(vals[i])) for i in idx]

        plt.figure(figsize=(6, 4))
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return prob, top_features, buf
    except Exception as e:
        print("SHAP 计算失败:", e)
        return prob, [], None

# ------------------ 页面 UI ------------------
st.title("再入ICU 风险预测工具 - ReAdmit")
st.warning("⚠️ 上传报告截图/照片前请务必隐去敏感信息。")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.form("icu_form"):
    col1, col2, col3, col4 = st.columns(4, gap="small")

    # 基本信息
    with col1:
        st.subheader("📝 基本信息")
        age = st.number_input("年龄（岁）", min_value=0, max_value=120)
        gender = st.radio("性别", options=["男", "女"])
        gender_score = 1 if gender == "男" else 0
        los_hospital = st.number_input("住院时长（天）")
        los_icu = st.number_input("ICU住院时长（天）")

    # 生命体征
    with col2:
        st.subheader("❤️ 生命体征")
        hr = st.number_input("心率（次/分）")
        sbp = st.number_input("收缩压（mmHg）")
        dbp = st.number_input("舒张压（mmHg）")
        mbp = (sbp + 2 * dbp) / 3 if sbp and dbp else 0
        st.number_input("平均动脉压（mmHg）自动计算", value=mbp, disabled=True)
        spo2 = st.number_input("血氧饱和度 (%)")
        temp = st.number_input("体温（℃）")

    # 其他体征
    with col3:
        st.subheader("🌡 其他体征")  
        urine = st.number_input("最后24h尿量（ml）")
        o2flow = st.number_input("吸氧流量（L/min）")
        intubated = st.radio("是否有气管插管或切开", options=["有", "无"])
        intubation_flag = 1 if intubated == "有" else 0
        mech_time = st.number_input("机械通气时长（小时）")

    # Charlson 合并症
    with col4:
        st.subheader("🧾 Charlson 合并症选择")
        group1 = st.multiselect("1 分类（1分）", ["心肌梗死","充血性心力衰竭","周围血管疾病","脑血管疾病","痴呆","慢性肺部疾病","结缔组织病","溃疡病","轻度肝脏疾病","糖尿病"])
        group2 = st.multiselect("2 分类（2分）", ["偏瘫","中度和重度肾脏疾病","糖尿病伴有器官损害","原发性肿瘤","白血病","淋巴瘤"])
        group3 = st.multiselect("3 分类（3分）", ["中度和重度肝脏疾病"])
        group4 = st.multiselect("6 分类（6分）", ["转移性肿瘤","获得性免疫缺陷综合征（艾滋病）"])
        selections = {"group1": group1, "group2": group2, "group3": group3, "group4": group4}
        charlson_score = calculate_charlson_score(age, selections)
        st.success(f"Charlson 合并症指数（含年龄加权）: {charlson_score}")

    # ---------- 检验报告上传（全宽） ----------
    st.subheader("🧪 上传检验报告")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4, gap="small")
    with col_u1:
        cbc_images = st.file_uploader("血常规", type=["png","jpg","jpeg"], accept_multiple_files=True)
    with col_u2:
        liver_kidney_images = st.file_uploader("肝肾功/生化", type=["png","jpg","jpeg"], accept_multiple_files=True)
    with col_u3:
        coag_images = st.file_uploader("凝血", type=["png","jpg","jpeg"], accept_multiple_files=True)
    with col_u4:
        abg_images = st.file_uploader("血气分析", type=["png","jpg","jpeg"], accept_multiple_files=True)

    # 放射学检查
    st.subheader("📷 放射学检查")
    no_radiology = st.checkbox("未进行放射学检查")
    report_image = st.file_uploader("最近一次放射学报告截图", type=["png","jpg","jpeg"], disabled=no_radiology)

    submitted = st.form_submit_button("🔍 进行风险预测")
    
    
# ------------------ 预测 ------------------
if submitted:
    try:
        embeddings_reduced = np.zeros((1, 768))
        if not no_radiology and report_image:
            extracted_text = extract_text_from_image(report_image)
            st.info(f"提取到的放射学报告文本：{extracted_text[:100]}...")
            embeddings = generate_embeddings(extracted_text)
            embeddings_reduced = pca.transform([embeddings])

        # 实验室指标提取
        lab_values = {}
        lab_groups = {
            "cbc": ["wbc","rbc","hemoglobin","hematocrit","mch","platelet","rdw"],
            "liver_kidney": ["creatinine","alt","ast","albumin","bilirubin_total"],
            "coag": ["inr","pt","ptt"],
            "abg": ["ph","be","pao2","paco2","lactate","bicarbonate","calcium",
                        "chloride","glucose","sodium","potassium"]
        }

        def extract_and_store(image_files, keys, label):
            combined_text = " "
            if image_files:
                for img in image_files:
                    combined_text += extract_text_from_image(img) + " "
                extracted = extract_lab_values(combined_text)
                for key in keys:
                    if key in extracted:
                        lab_values[key] = extracted[key]
                st.success(f"{label} 提取指标如下：")
                st.write({k: lab_values[k] for k in keys if k in lab_values})

        extract_and_store(cbc_images, lab_groups["cbc"], "血常规")
        extract_and_store(liver_kidney_images, lab_groups["liver_kidney"], "肝肾功/生化检验")
        extract_and_store(coag_images, lab_groups["coag"], "凝血检验")
        extract_and_store(abg_images, lab_groups["abg"], "血气分析")



        # 构造模型输入
        # ------------------ 单位换算 ------------------
        unit_conversion = {
            "hemoglobin": 0.1,
            "creatinine": 88.4,
            "albumin": 0.1,
            "bilirubin_total": 17.1,
            "glucose": 55.51,
            "calcium": 2
            }

# 不修改显示，只在模型输入时换算
        lab_values_for_model = lab_values.copy()
        for key, factor in unit_conversion.items():
            if key in lab_values_for_model:
                lab_values_for_model[key] = lab_values_for_model[key] * factor
        
        
        lab_features = lab_groups["cbc"] + lab_groups["liver_kidney"] + lab_groups["coag"] + lab_groups["abg"]
        lab_inputs = [lab_values_for_model.get(name, 0) for name in lab_features]
        input_values = np.array([
                age, gender_score, los_hospital, los_icu, hr, sbp, dbp, mbp, spo2, temp,
                urine, o2flow, intubation_flag, mech_time, charlson_score
        ] + lab_inputs).reshape(1, -1)
        final_input = np.hstack([input_values, embeddings_reduced])

        # 预测 + SHAP
        prob, top_features, shap_buf = predict_with_shap(model, input_values, embeddings_reduced)
        result = "自ICU转出到病房后72小时再入ICU的风险较高" if prob >= threshold else "自ICU转出到病房后72小时再入ICU的风险较低"
        st.subheader("📊 预测结果")
        st.write(f"风险分类结果：**{result}**")

        st.subheader("🔎 SHAP 解释")
        if top_features:
            st.table([{ "feature": f, "shap_value": v } for f, v in top_features])
        if shap_buf:
            st.image(shap_buf)
       
        # LLM 建议
        # -------- 整理患者信息 --------
        patient_summary = f"""
        基本信息:
        - 年龄: {age} 岁
        - 性别: {gender}
        - 住院时长: {los_hospital} 天
        - ICU住院时长: {los_icu} 天

        生命体征:
        - 心率: {hr} 次/分
        - 血压: {sbp}/{dbp} mmHg (平均动脉压 {mbp:.1f})
        - 血氧饱和度: {spo2}%
        - 体温: {temp} ℃
        - 24h尿量: {urine} ml
        - 吸氧流量: {o2flow} L/min
        - 气管插管/切开: {"是" if intubation_flag else "否"}
        - 机械通气时长: {mech_time} 小时

        Charlson 合并症指数（含年龄加权）: {charlson_score}
        具体合并症:
        - 1分类: {", ".join(group1) if group1 else "无"}
        - 2分类: {", ".join(group2) if group2 else "无"}
        - 3分类: {", ".join(group3) if group3 else "无"}
        - 6分类: {", ".join(group4) if group4 else "无"}

        实验室检验指标:
        """  

        # -------- 加入实验室指标 --------
        if lab_values:
            for k, v in lab_values.items():
                patient_summary += f"- {k}: {v}\n"
        else:
            patient_summary += "- 未提取到实验室指标\n"

        # -------- 模型预测 & SHAP --------
        patient_summary += f"\n模型预测结果: {result}\n"

        shap_text = "\n".join([f"{i+1}. {f}: {v:.3f}" for i, (f, v) in enumerate(top_features)])

        prompt = f"""
        患者情况如下:
        {patient_summary}

        模型预测结果: {result} 
        主要贡献特征 (SHAP 排名前列): 
        {shap_text}

        请根据以上信息提供:
        1. 简要解释风险结果（结合患者的临床特征、实验室检查和合并症情况）
        2. 三条可行的临床干预建议（每条附简短理由）
        3. 三篇相关文献（标题 + 期刊 + 年份）
        """
        advice = ask_deepseek_online(prompt)
        st.subheader("🤖 LLM 建议")
        st.markdown(advice)

        st.session_state["messages"].append({"role":"assistant","content":advice})

    except Exception as e:
        st.error(f"预测出错: {e}")

# ------------------ 多轮对话 ------------------
st.subheader("💬 与助手继续对话")
user_q = st.text_input("输入你的追问：")
if st.button("发送问题"):
    if user_q:
        st.session_state["messages"].append({"role":"user","content":user_q})
        history_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])
        resp = ask_deepseek_online(history_prompt)
        st.session_state["messages"].append({"role":"assistant","content":resp})
        st.write(resp)
