"""
Streamlit Dashboard: Talent Success Formula — Professional, interactive, and production-ready
File: streamlit_dashboard_success_formula.py
Run: streamlit run streamlit_dashboard_success_formula.py
"""

# ============================================================
# 📦 IMPORT LIBRARIES
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import base64
import textwrap
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# ⚙️ PAGE CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title='Talent Success Formula — Interactive Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================
# 🌈 ELEGANT & LIVELY STREAMLIT THEME
# (Yellow Gradient Sidebar — High Contrast)
# ============================================================
st.markdown("""
<style>

    /* ====== BASE APP ====== */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        color: #0f172a;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ====== SIDEBAR ====== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff7cc 0%, #fde68a 50%, #fbbf24 100%) !important;
        color: #111827 !important;
        box-shadow: 4px 0 10px rgba(0,0,0,0.15);
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important; /* teks hitam pekat */
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #78350f !important; /* coklat gelap untuk heading */
        font-weight: 700;
    }

    [data-testid="stSidebar"] label {
        color: #27272a !important; /* abu gelap */
    }

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.9);
        border-radius: 6px;
        border: 1px solid #fcd34d;
    }

    [data-testid="stSidebar"] .stSelectbox:hover {
        background-color: rgba(255,255,255,1);
    }

    [data-testid="stSidebar"] .stCheckbox label {
        color: #111827 !important;
    }

    /* ====== CONTENT AREA ====== */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .stMarkdown, .stDataFrame, .stPlotlyChart {
        background-color: rgba(255,255,255,0.97);
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* ====== METRIC CARDS ====== */
    .stMetric {
        background: linear-gradient(135deg, #fff, #fef9c3);
        padding: 12px;
        border-radius: 10px;
        color: #1e293b !important;
        font-weight: 600;
    }

    /* ====== BUTTON ====== */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #facc15, #eab308);
        color: #111827;
        border-radius: 10px;
        border: none;
        padding: 0.7em 1.4em;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #eab308, #ca8a04);
        transform: scale(1.05);
    }

    /* ====== LINK ====== */
    a {
        color: #b45309 !important;
        font-weight: 600;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
        color: #92400e !important;
    }

    /* ====== FOOTER CAPTION ====== */
    .stCaption {
        color: #475569 !important;
        font-size: 0.9rem;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# 🎯 MODERN HEADER BANNER
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
    color: white;
    padding: 2.8rem 1rem;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(30,58,138,0.3);
">
    <h1 style="margin-bottom: 0.4rem; font-size: 2.6rem;">
        💼 Talent Success Formula Dashboard
    </h1>
    <p style="font-size: 1.1rem; opacity: 0.95;">
        Professional • Interactive • Insightful HR Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 🧰 HELPERS
# (fixed to cache DataFrames, not ExcelFile)
# ============================================================

@st.cache_data
def load_excel_bytes(bytes_data):
    """
    Return a dict of sheet_name -> DataFrame.
    This is pickle-serializable and safe for st.cache_data.
    """
    try:
        # sheet_name=None returns a dict of DataFrames
        sheets_dict = pd.read_excel(BytesIO(bytes_data), sheet_name=None)
        return sheets_dict
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        return None


@st.cache_data
def read_default_excel(path):
    """
    Read file from disk and return dict of sheet_name -> DataFrame.
    """
    try:
        sheets_dict = pd.read_excel(path, sheet_name=None)
        return sheets_dict
    except Exception as e:
        # we don't spam st.error here because this may be called at import-time
        return None


def safe_read_sheet(xls_dict, sheet_name):
    """
    xls_dict is expected to be dict returned by read_default_excel or load_excel_bytes.
    Returns DataFrame or None.
    """
    try:
        if xls_dict is None:
            return None
        return xls_dict.get(sheet_name)
    except Exception:
        return None


def download_link(df, filename, link_text='Download CSV'):
    csv = df.to_csv(index=True).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


# ============================================================
# 🧭 SIDEBAR CONTROLS
# ============================================================

# ---------------------------
# Load Data from GitHub (tanpa upload manual)
# ---------------------------
st.sidebar.title('Data Source')
st.sidebar.markdown('File otomatis dimuat dari GitHub repository.')

# URL file Excel di GitHub (gunakan raw link)
GITHUB_FILE_URL = (
    "https://github.com/azizmuzaki4/talent-success-formula-dashboard/"
    "raw/refs/heads/main/study_case_DA.xlsx"
)


@st.cache_data
def load_from_github(url):
    try:
        # dummy to keep caching consistent
        file_bytes = BytesIO(pd.util.hash_pandas_object(pd.DataFrame([url])).to_numpy().tobytes())
        response = pd.read_excel(url, sheet_name=None)
        return response
    except Exception as e:
        st.error(f"Gagal memuat file dari GitHub: {e}")
        return None


xls = load_from_github(GITHUB_FILE_URL)

if xls is None:
    st.error('❌ Tidak bisa memuat file dari GitHub. Pastikan URL benar dan file dapat diakses publik.')
    st.stop()
else:
    st.sidebar.success('✅ File berhasil dimuat dari GitHub.')

# ============================================================
# ⚙️ MODEL & PREPROCESSING OPTIONS
# ============================================================

impute_strategy = st.sidebar.selectbox(
    'Strategi Imputasi',
    options=['mean', 'median', 'most_frequent'],
    index=0
)

scaler_choice = st.sidebar.selectbox(
    'Skaler',
    options=['StandardScaler', 'None'],
    index=0
)

model_choice = st.sidebar.selectbox(
    'Model untuk Success Formula',
    options=['LinearRegression', 'RandomForestRegressor'],
    index=0
)

show_shap = st.sidebar.checkbox('Coba SHAP (jika tersedia)', value=False)

st.sidebar.markdown('---')
st.sidebar.markdown(
    'Tips: buka bagian *Data* untuk memeriksa sheet yang terdeteksi dan memilih/menyesuaikan kolom jika perlu.'
)
# ============================================================
# 📥 DATA INGESTION
# ============================================================

if xls is None:
    st.title('Talent Success Formula — Dashboard')
    st.info('Silakan upload file Excel (Study Case DA.xlsx) pada sidebar untuk memulai.')
    st.stop()

# ------------------------------------------------------------
# List available sheets (xls is dict of DataFrames)
# ------------------------------------------------------------
sheets = list(xls.keys())

st.sidebar.markdown('### Sheets terdeteksi')
for s in sheets:
    st.sidebar.text('• ' + s)

# ------------------------------------------------------------
# Recommended sheet names
# ------------------------------------------------------------
recommended = {
    'competencies_yearly': 'Kompetensi tahunan per employee',
    'dim_competency_pillars': 'Definisi pilar kompetensi',
    'papi_scores': 'PAPI scores',
    'profiles_psych': 'Profil psikometrik',
    'strengths': 'Strengths',
    'employees': 'Master karyawan',
    'performance_yearly': 'Performa tahunan (opsional)'
}

# ------------------------------------------------------------
# Auto read if present (safe_read_sheet returns DataFrame or None)
# ------------------------------------------------------------
with st.spinner('Membaca sheet...'):
    competencies_yearly = safe_read_sheet(xls, 'competencies_yearly')
    dim_competency_pillars = safe_read_sheet(xls, 'dim_competency_pillars')
    papi_scores = safe_read_sheet(xls, 'papi_scores')
    profiles_psych = safe_read_sheet(xls, 'profiles_psych')
    strengths = safe_read_sheet(xls, 'strengths')
    employees = safe_read_sheet(xls, 'employees')
    performance_yearly = safe_read_sheet(xls, 'performance_yearly')

# ------------------------------------------------------------
# Allow override if sheet names differ
# ------------------------------------------------------------
st.sidebar.markdown('---')
st.sidebar.markdown('Jika file Anda memakai sheet nama lain, Anda dapat memilihnya di sini:')

col1, col2 = st.sidebar.columns(2)
sheet_map = {}
keys_list = list(recommended.keys())

for i, key in enumerate(keys_list):
    with (col1 if i % 2 == 0 else col2):
        available = st.sidebar.selectbox(
            f"Pilih sheet untuk '{key}'",
            options=[None] + sheets,
            index=0,
            key=key
        )
        sheet_map[key] = available

# ------------------------------------------------------------
# If user selected a different sheet name, re-read from xls dict
# ------------------------------------------------------------
for logical_name, sheet_selected in sheet_map.items():
    if sheet_selected and sheet_selected in sheets:
        # set variable in locals() to DataFrame from xls
        locals()[logical_name] = safe_read_sheet(xls, sheet_selected)

# ------------------------------------------------------------
# Quick validation
# ------------------------------------------------------------
st.header('1) Ringkasan Data')

colA, colB, colC = st.columns([1, 1, 2])

with colA:
    st.metric('Sheets terdeteksi', len(sheets))
    st.write('Nama sheet (sample):')
    st.write(sheets[:10])

with colB:
    st.metric('Rows employees', int(employees.shape[0]) if employees is not None else 0)
    st.metric('Rows competencies', int(competencies_yearly.shape[0]) if competencies_yearly is not None else 0)

with colC:
    st.write('Keterangan singkat:')
    st.write(textwrap.dedent('''
        - Pastikan sheet 'employees' ada dan memuat kolom:
          employee_id, grade_id, education_id, years_of_service_months
        - Jika 'performance_yearly' tidak ada,
          dashboard akan membangun proxy target (rata-rata kompetensi)
    '''))

# ------------------------------------------------------------
# Show head of each important df and allow toggling
# ------------------------------------------------------------
st.subheader('Preview dataset penting')

if st.checkbox('Tampilkan preview semua dataset (komprehensif)'):
    for name in [
        'employees',
        'competencies_yearly',
        'dim_competency_pillars',
        'papi_scores',
        'profiles_psych',
        'strengths',
        'performance_yearly'
    ]:
        df = locals().get(name)
        st.write(f"### {name}")
        if df is None:
            st.write('Tidak tersedia')
        else:
            st.dataframe(df.head(200))
# ---------------------------
# Data preparation (as in user's script) — minimal changes
# ---------------------------

st.header('2) Persiapan Data & Eksplorasi Interaktif')

# === Merge Data ===
comp = None
if competencies_yearly is not None and dim_competency_pillars is not None:
    # Ensure pillar_code exists in competencies_yearly
    comp = competencies_yearly.merge(dim_competency_pillars, on='pillar_code', how='left')
    if employees is not None:
        comp = comp.merge(
            employees[['employee_id', 'grade_id', 'education_id', 'years_of_service_months']],
            on='employee_id', how='left'
        )

psych = None
if profiles_psych is not None and employees is not None:
    psych = profiles_psych.merge(
        employees[['employee_id', 'grade_id', 'education_id', 'years_of_service_months']],
        on='employee_id', how='left'
    )

papi = None
if papi_scores is not None and employees is not None:
    papi = papi_scores.merge(
        employees[['employee_id', 'grade_id', 'education_id', 'years_of_service_months']],
        on='employee_id', how='left'
    )

strength_top = None
if strengths is not None and employees is not None:
    if 'rank' in strengths.columns:
        strength_top = strengths[strengths['rank'] == 1].merge(
            employees[['employee_id', 'grade_id', 'education_id', 'years_of_service_months']],
            on='employee_id', how='left'
        )

# === EDA Controls ===
eda_options = list(comp['pillar_label'].unique()) if comp is not None and 'pillar_label' in comp.columns else []
eda_cols = st.multiselect(
    'Pilih pilar/kolom untuk EDA (jika tersedia)',
    options=eda_options,
    max_selections=6
)

# === Layout for Plots ===
col1, col2 = st.columns([2.5, 1])  # perbesar kolom kiri

# -------------------------
# BOX PLOT
# -------------------------
with col1:
    st.subheader('Distribusi Skor Kompetensi per Pilar')

    if comp is not None and not comp.empty:
        selected_pillars = eda_cols if eda_cols else comp['pillar_label'].unique()[:6]

        fig = px.box(
            comp[comp['pillar_label'].isin(selected_pillars)],
            x='pillar_label',
            y='score',
            color='grade_id',
            points='outliers',
            title='Distribusi Skor Kompetensi per Pilar & Grade',
            height=650
        )

        # ✨ Penyesuaian tampilan boxplot agar lebih seimbang dan jelas
        fig.update_traces(
            boxmean='sd',
            jitter=0.35,
            marker=dict(size=7, opacity=0.7),
            line=dict(width=1.4)
        )

        fig.update_layout(
            font=dict(size=14),
            margin=dict(l=40, r=40, t=60, b=80),
            boxmode='group',
            xaxis=dict(
                title='Pilar Kompetensi',
                tickangle=-30,
                tickfont=dict(size=13),
                categoryorder='array',
                categoryarray=selected_pillars
            ),
            yaxis=dict(
                title='Skor',
                range=[0, 10],           # 🔹 Batasi rentang skor 0–10
                dtick=1,                 # tampilkan grid tiap 1 poin
                gridcolor='rgba(200,200,200,0.3)',
            ),
            legend=dict(
                title='Grade ID',
                orientation='h',
                yanchor='bottom',
                y=-0.25,
                xanchor='center',
                x=0.5
            ),
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# STRENGTHS
# -------------------------
with col2:
    st.subheader('Top Strengths (Rank 1)')
    if strength_top is not None and not strength_top.empty:
        top_strengths = strength_top['theme'].value_counts().nlargest(10)
        fig2 = px.bar(
            x=top_strengths.values,
            y=top_strengths.index,
            orientation='h',
            title='Top Strengths (Rank 1)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('Data strengths tidak tersedia.')

# -------------------------
# HEATMAP KORELASI
# -------------------------
st.subheader('Korelasi antar Pilar Kompetensi')

if comp is not None:
    if 'pillar_label' in comp.columns and 'score' in comp.columns:
        comp_pivot = comp.pivot_table(
            index='employee_id',
            columns='pillar_label',
            values='score',
            aggfunc='mean'
        )
        corr = comp_pivot.corr()

        figc = px.imshow(
            corr,
            text_auto='.2f',
            title='Korelasi Pilar Kompetensi',
            height=800,  # 🔹 perbesar ukuran heatmap
            color_continuous_scale='RdYlBu'
        )
        figc.update_layout(
            font=dict(size=13),
            margin=dict(l=50, r=50, t=70, b=70)
        )
        st.plotly_chart(figc, use_container_width=True)

# -------------------------
# PSYCHOMETRIC PROFILE
# -------------------------
st.subheader('Profil Psikometrik')

if psych is not None:
    colA, colB = st.columns(2)

    with colA:
        if 'disc' in psych.columns:
            disc_fig = px.histogram(
                psych,
                x='disc',
                color='grade_id',
                barmode='group',
                title='Distribusi DISC per Grade'
            )
            st.plotly_chart(disc_fig, use_container_width=True)
        else:
            st.write('Kolom DISC tidak ditemukan')

    with colB:
        if 'iq' in psych.columns and 'grade_id' in psych.columns:
            iq_fig = px.box(
                psych,
                x='grade_id',
                y='iq',
                title='Perbandingan IQ antar Grade'
            )
            st.plotly_chart(iq_fig, use_container_width=True)
        else:
            st.write('Kolom IQ atau grade_id tidak ditemukan')
# ---------------------------
# Build aggregate dataset for modeling
# ---------------------------

st.header('3) Membangun Dataset untuk Success Formula')

# === Prepare component-wide tables ===
comp_wide = None
if comp is not None and not comp.empty and 'pillar_label' in comp.columns:
    comp_wide = comp.pivot_table(
        index='employee_id',
        columns='pillar_label',
        values='score',
        aggfunc='mean'
    )

papi_wide = None
if papi is not None and not papi.empty and 'scale_code' in papi.columns:
    papi_wide = papi.pivot_table(
        index='employee_id',
        columns='scale_code',
        values='score',
        aggfunc='mean'
    )

strength_main = None
if strength_top is not None and not strength_top.empty and 'theme' in strength_top.columns:
    strength_main = strength_top.pivot_table(
        index='employee_id',
        columns='theme',
        values='rank',
        aggfunc=lambda x: 1,
        fill_value=0
    )

# === Merge all sources ===
base = (
    employees[['employee_id', 'grade_id', 'education_id', 'years_of_service_months']]
    .set_index('employee_id')
    if employees is not None else pd.DataFrame()
)

dfs_to_join = [comp_wide, papi_wide, strength_main]

# optionally include psych-derived columns if present
if psych is not None and set(['iq', 'gtq', 'tiki']).issubset(psych.columns):
    dfs_to_join.append(psych.set_index('employee_id')[['iq', 'gtq', 'tiki']])

for df in dfs_to_join:
    if df is not None:
        base = base.join(df, how='left')

# === Performance ===
performance = None
if performance_yearly is not None:
    try:
        performance = (
            performance_yearly.groupby('employee_id')['score']
            .mean()
            .rename('performance_score')
        )
        base = base.join(performance, how='left')
    except Exception:
        performance = None

st.write(f'Dataset untuk modelling: {base.shape[0]} rows x {base.shape[1]} columns')

if st.checkbox('Tampilkan 10 baris dari dataset modelling'):
    st.dataframe(base.head(10))

# ---------------------------
# Preprocessing controls
# ---------------------------

imputer = SimpleImputer(strategy=impute_strategy)

# guard against empty columns
if base.shape[1] == 0:
    st.error('Dataset modelling kosong — pastikan sheet employees dan competencies tersedia dan di-join dengan benar.')
    st.stop()

base_filled = pd.DataFrame(
    imputer.fit_transform(base.fillna(0)),
    columns=base.columns,
    index=base.index
)

if scaler_choice == 'StandardScaler':
    scaler = StandardScaler()
    base_scaled = pd.DataFrame(
        scaler.fit_transform(base_filled),
        columns=base_filled.columns,
        index=base_filled.index
    )
else:
    base_scaled = base_filled.copy()

# === Choose target ===
use_actual_target = 'performance_score' in base_scaled.columns

if use_actual_target:
    st.success('Terdeteksi performance_yearly dan akan digunakan sebagai target utama.')
else:
    st.warning('Tidak ada performance_yearly — dashboard akan membuat proxy target: rata-rata kompetensi per employee (sebagai fallback).')

# make sure comp exists
proxy = (
    comp.groupby('employee_id')['score'].mean().reindex(base_scaled.index).fillna(0)
    if comp is not None else pd.Series(0, index=base_scaled.index)
)
base_scaled['performance_score_proxy'] = proxy

target_col = 'performance_score' if use_actual_target else 'performance_score_proxy'

# === Choose features subset ===
all_features = [c for c in base_scaled.columns if c != target_col]
selected_features = st.multiselect(
    'Pilih fitur untuk model (kosong = semua fitur)',
    options=all_features,
    default=all_features
)

X = base_scaled[selected_features]
y = base_scaled[target_col]

st.markdown('**Preview fitur/target shapes**')
st.write('X:', X.shape, ' y:', y.shape)

# ---------------------------
# Modeling — otomatis tanpa tombol
# ---------------------------

st.header('4) Membangun Model & Menampilkan Rumus Kesuksesan')

with st.spinner('Melatih model & menghitung Success Formula...'):
    # Pilih model otomatis
    if model_choice == 'LinearRegression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X.fillna(0), y.fillna(0))

    # Hitung bobot / feature importance
    if hasattr(model, 'coef_'):
        coefs = pd.Series(model.coef_, index=X.columns)
    else:
        coefs = pd.Series(model.feature_importances_, index=X.columns)

    coefs = coefs.sort_values(key=lambda s: np.abs(s), ascending=False)

    sf = pd.DataFrame({'Feature': coefs.index, 'Weight': coefs.values})
    sf['AbsWeight'] = sf['Weight'].abs()
    sf = sf.set_index('Feature')

    # === Tampilkan hasil ===
    st.subheader('Top 15 Faktor (berdasarkan absolute weight)')
    st.dataframe(sf.sort_values('AbsWeight', ascending=False).head(15))

    # === Plot bar kontribusi ===
    topN = min(15, sf.shape[0])
    figf = px.bar(
        sf.sort_values('AbsWeight', ascending=False).head(topN).reset_index(),
        x='AbsWeight',
        y='Feature',
        orientation='h',
        title='Top Fitur - Kontribusi (Abs)'
    )
    st.plotly_chart(figf, use_container_width=True)

    # === Pie chart distribusi bobot ===
    pie_df = sf.copy()
    if pie_df['AbsWeight'].sum() > 0:
        pie = px.pie(
            pie_df.reset_index().head(12),
            names='Feature',
            values='AbsWeight',
            title='Distribusi Bobot (Top 12)'
        )
        st.plotly_chart(pie, use_container_width=True)

# === Narrative generator ===
st.subheader('Narrative — Business Storytelling')

competency_terms = ['Drive', 'Curiosity', 'Insight', 'Quality', 'Synergy', 'Resilience', 'Discipline']
psych_terms = ['iq', 'gtq', 'tiki', 'papi', 'scale']
behavior_terms = list(strength_top['theme'].dropna().unique()) if strength_top is not None else []
context_terms = ['grade', 'education', 'years']


def categorize(name):
    n = name.lower()
    if any(t.lower() in n for t in competency_terms):
        return 'Kompetensi'
    if any(t.lower() in n for t in psych_terms):
        return 'Psikometri'
    if any(t.lower() in n for t in behavior_terms):
        return 'Perilaku'
    if any(t.lower() in n for t in context_terms):
        return 'Kontekstual'
    return 'Lainnya'


top_feats = sf.sort_values('AbsWeight', ascending=False).head(10).reset_index()
top_feats['Kategori'] = top_feats['Feature'].apply(categorize)

for cat in top_feats['Kategori'].unique():
    names = top_feats[top_feats['Kategori'] == cat]['Feature'].tolist()

    if cat == 'Kompetensi':
        st.markdown(
            f"**🏆 Kompetensi:** Faktor seperti **{', '.join(names[:5])}** terbukti berkontribusi kuat terhadap performa."
        )
    elif cat == 'Psikometri':
        st.markdown(
            f"**🧠 Psikometri:** Indikator seperti **{', '.join(names[:5])}** menunjukkan peran kognitif/kepribadian."
        )
    elif cat == 'Perilaku':
        st.markdown(
            f"**💡 Perilaku:** Tema kekuatan seperti **{', '.join(names[:5])}** sering muncul di individu berperforma tinggi."
        )
    elif cat == 'Kontekstual':
        st.markdown(
            f"**🌱 Kontekstual:** Faktor pengalaman/pendidikan seperti **{', '.join(names[:5])}** juga penting."
        )
    else:
        st.markdown(f"**🔎 Lainnya:** {', '.join(names[:5])}")

# === Model diagnostics ===
st.subheader('Model Diagnostics')

try:
    scores = cross_val_score(model, X.fillna(0), y.fillna(0), cv=5, scoring='r2')
    st.write('Cross-validated R² (5-fold):', np.round(scores, 3), 'mean:', np.round(scores.mean(), 3))
except Exception:
    st.write('Tidak bisa menghitung cross-val untuk model ini.')


# === Simpan hasil ke session_state ===
st.session_state['last_formula'] = sf

st.markdown(
    download_link(
        sf.reset_index().rename(columns={'index': 'Feature'})[['Feature', 'Weight']],
        'success_formula.csv',
        'Download Success Formula (CSV)'
    ),
    unsafe_allow_html=True
)


# ---------------------------
# Export / utility
# ---------------------------
st.header('5) Utility & Export')

if 'last_formula' in st.session_state:
    sf = st.session_state['last_formula']
    st.write('Preview last computed formula:')
    st.dataframe(sf.head(20))

    st.markdown(
        download_link(
            sf.reset_index().rename(columns={'index': 'Feature', 'Weight': 'Weight'}),
            'success_formula_latest.csv',
            'Download latest success formula CSV'
        ),
        unsafe_allow_html=True
    )
else:
    st.info('Belum ada formula yang dihitung — klik tombol "Hitung Success Formula" untuk memulai.')

st.markdown('---')
st.caption(
    'Dashboard ini dibuat untuk keperluan analisis HR dan talent management. '
    'Gunakan hasil model sebagai insight — bukan sebagai penentu tunggal keputusan SDM.'
)

# EOF
