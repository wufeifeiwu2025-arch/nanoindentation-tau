import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="纳米压痕 Tau_max 工具", layout="wide")
st.title("🧪 纳米压痕 Pop-in & τ_max 分析工具 (增强版)")

# --- 侧边栏：参数设置 ---
st.sidebar.header("参数设置")
r_tip = st.sidebar.number_input("压头半径 R (nm)", value=1000.0)
er_sample = st.sidebar.number_input("等效模量 Er (GPa)", value=82.7)
popin_threshold = st.sidebar.slider("Pop-in 阈值 (nm)", 0.1, 5.0, 1.0)
fit_range = st.sidebar.slider("弹性拟合范围 (nm)", 1.0, 10.0, 5.0)

# --- 核心计算逻辑 ---
def hertzian_model(h, K):
    return K * np.power(h, 1.5)

def process_file(file):
    try:
        content = file.getvalue().decode("utf-8", errors="ignore")
        # 跳过前3行，并且不把第4行当做列名(header=None)，避免列名乱码影响数据读取
        df = pd.read_csv(io.StringIO(content), sep='\t', skiprows=3, header=None)
        
        if df.shape[1] < 2:
            return None
            
        # 【关键修复1】强制将列转换为纯数字格式，遇到文本自动变为 NaN
        h = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        p = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        
        # 【关键修复2】去除 NaN 并保留大于 0.1 的有效接触点
        valid = (~np.isnan(h)) & (~np.isnan(p)) & (h > 0.1) & (p > 0.1)
        h, p = h[valid], p[valid]
        
        if len(h) < 10:
            return None
            
        mask = (h < fit_range)
        
        # 【关键修复3】拟合前必须确保数据点足够 (至少需要3个点才能稳定拟合)
        if len(h[mask]) < 3:
            return None
            
        popt, _ = curve_fit(hertzian_model, h[mask], p[mask])
        k_fit = popt[0]

        h_theory = np.power(p / k_fit, 1/1.5)
        dev = h - h_theory
        
        # 寻找偏离大于阈值的点
        idx = np.where(dev > popin_threshold)[0]
        if len(idx) == 0:
            return None

        # 提取 Pop-in 载荷并计算应力
        P = p[idx[0]]
        Pn = P * 1e-6
        ErPa = er_sample * 1e9
        Rm = r_tip * 1e-9

        tau_pa = 0.31 * ((6 * Pn * ErPa**2) / (np.pi**3 * Rm**2)) ** (1/3)
        return tau_pa / 1e9
    except Exception as e:
        # 出错时静默跳过，保证整个程序不崩溃
        return None

# --- 主界面：文件上传与处理 ---
uploaded_files = st.file_uploader("📂 上传纳米压痕 txt 文件，可多选", type="txt", accept_multiple_files=True)

if uploaded_files:
    st.info(f"共上传 {len(uploaded_files)} 个文件，正在处理中...")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        tau = process_file(file)
        if tau:
            results.append(tau)
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    if results:
        results.sort()
        n = len(results)
        f_prob = np.arange(1, n + 1) / (n + 1)
        
        st.success(f"处理完成！成功提取到 {n} 个有效 Pop-in 数据。")
        
        # 布局：左边画图，右边显示数据表格
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 统计分布图")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(results, f_prob, 'o', mfc='none', mec='blue', label='Experimental Data')
            ax.set_xlabel(r"Maximum shear stress $\tau_{max}$ (GPa)", fontsize=12)
            ax.set_ylabel(r"Cumulative probability $f$", fontsize=12)
            ax.set_title(r"Statistical distribution of $\tau_{max}$", fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            st.subheader("📝 数据摘要")
            res_df = pd.DataFrame({"Tau_max_GPa": results, "f_prob": f_prob})
            st.dataframe(res_df.describe(), use_container_width=True)
            
            # 下载按钮
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 下载分析结果 (CSV)", 
                data=csv, 
                file_name="popin_analysis_results.csv", 
                mime='text/csv'
            )
    else:
        st.error("未能在上传文件中识别到有效的 Pop-in 数据。\n\n可能原因：\n1. 文件带有无法解析的乱码\n2. 阈值 (Pop-in 阈值) 设置过大，错过了微小突跳\n3. 弹性拟合范围过大或过小。\n\n💡 建议在左侧边栏将【弹性拟合范围】缩小到 2.0，将【Pop-in 阈值】缩小到 0.5 再试一次。")
