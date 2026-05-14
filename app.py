import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="纳米压痕 Tau_max 工具", layout="wide")
st.title("🧪 纳米压痕 Pop-in & τ_max 分析工具")

# 侧边栏参数
st.sidebar.header("参数设置")
r_tip = st.sidebar.number_input("压头半径 R (nm)", value=1000.0)
er_sample = st.sidebar.number_input("等效模量 Er (GPa)", value=82.7)
popin_threshold = st.sidebar.slider("Pop-in 阈值 (nm)", 0.1, 5.0, 1.0)
fit_range = st.sidebar.slider("弹性拟合范围 (nm)", 1.0, 10.0, 5.0)

# 赫兹模型
def hertzian_model(h, K):
    return K * np.power(h, 1.5)

# 单文件处理
def process_file(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content), sep='\t', skiprows=3)
        if df.shape[1] < 2:
            return None
        h = df.iloc[:,0].values
        p = df.iloc[:,1].values

        valid = (h>0.1) & (p>0.1)
        h,p = h[valid], p[valid]
        if len(h)<10:
            return None

        mask = h < fit_range
        popt, _ = curve_fit(hertzian_model, h[mask], p[mask])
        k_fit = popt[0]

        h_theory = np.power(p/k_fit, 1/1.5)
        dev = h - h_theory
        idx = np.where(dev > popin_threshold)[0]
        if len(idx)==0:
            return None

        P = p[idx[0]]
        Pn = P * 1e-6
        ErPa = er_sample * 1e9
        Rm = r_tip * 1e-9

        tau_pa = 0.31 * ((6 * Pn * ErPa**2) / (np.pi**3 * Rm**2)) ** (1/3)
        return tau_pa / 1e9
    except:
        return None

# 上传文件
uploaded = st.file_uploader("上传纳米压痕 txt 文件，可多选", type="txt", accept_multiple_files=True)

if uploaded:
    st.info(f"共上传 {len(uploaded)} 个文件")
    res = []
    for f in uploaded:
        t = process_file(f)
        if t:
            res.append(t)
    if res:
        res.sort()
        n = len(res)
        prob = np.arange(1,n+1)/(n+1)

        col1,col2 = st.columns([2,1])
        with col1:
            fig,ax = plt.subplots(figsize=(8,5))
            ax.plot(res, prob, 'o', mec='blue', mfc='none')
            ax.set_xlabel("τ_max (GPa)")
            ax.set_ylabel("Cumulative Probability")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        with col2:
            df_out = pd.DataFrame({"Tau_max(GPa)":res, "Prob":prob})
            st.dataframe(df_out.describe())
            csv = df_out.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载结果CSV", csv, "tau_result.csv")
    else:
        st.error("未识别到有效Pop-in，请调整阈值")
