"""
CleanData AI — Smart Guided Data Cleaning Tool
==============================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CleanData AI",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Industrial / Data-Lab Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0e1117;
    --surface:  #161b26;
    --border:   #2a3040;
    --accent:   #00e5a0;
    --accent2:  #3b82f6;
    --warn:     #f59e0b;
    --danger:   #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: var(--accent) !important;
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: var(--accent) !important; letter-spacing: -1px; }
h2 { color: var(--text)   !important; }
h3 { color: var(--accent2)!important; }

[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.75rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: var(--accent) !important;
    font-size: 1.5rem !important;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; }

.stButton > button {
    background: var(--accent) !important;
    color: #0e1117 !important;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none !important;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stDownloadButton"] > button {
    background: var(--accent2) !important;
    color: #fff !important;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none !important;
    border-radius: 6px;
}

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: 8px;
}

.stInfo    { background: rgba(59,130,246,.15) !important; border-left: 3px solid var(--accent2) !important; }
.stWarning { background: rgba(245,158,11,.15) !important; border-left: 3px solid var(--warn)    !important; }
.stSuccess { background: rgba(0,229,160,.15)  !important; border-left: 3px solid var(--accent)  !important; }
.stError   { background: rgba(239,68,68,.15)  !important; border-left: 3px solid var(--danger)  !important; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    margin-left: 6px;
}
.badge-blue { background: rgba(59,130,246,.18); color: var(--accent2); border: 1px solid var(--accent2); }

.hero {
    background: linear-gradient(135deg, #161b26 0%, #0e1117 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.4rem; }
.hero p  { color: var(--muted); margin: 0; font-size: 0.95rem; }

.step-header {
    border-top: 1px solid var(--border);
    padding-top: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        try:
            return pd.read_csv(uploaded_file, encoding='latin1')
        except:
            return pd.read_csv(uploaded_file, encoding='ISO-8859-1')
#def load_csv(uploaded_file) -> pd.DataFrame:
#    """Load a CSV file into a DataFrame (cached for performance)."""
#    return pd.read_csv(uploaded_file)



# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    result = pd.DataFrame({
        "Column":        missing_count.index,
        "Missing Count": missing_count.values,
        "Missing %":     missing_pct.values,
    })
    return result[result["Missing Count"] > 0].reset_index(drop=True)


def analyze_duplicates(df: pd.DataFrame) -> int:
    """Return number of fully-duplicate rows."""
    return int(df.duplicated().sum())


def analyze_type_issues(df: pd.DataFrame) -> list:
    """Detect object columns that might actually be numeric."""
    suspects = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str)
        converted = pd.to_numeric(sample.str.replace(",", "").str.strip(), errors="coerce")
        if len(sample) > 0 and converted.notna().sum() / len(sample) > 0.8:
            suspects.append(col)
    return suspects


def analyze_text_issues(df: pd.DataFrame) -> dict:
    """Find text columns with casing / whitespace issues."""
    issues = {"casing": [], "spaces": []}
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        if (sample != sample.str.lower()).any() and (sample != sample.str.upper()).any():
            issues["casing"].append(col)
        if (sample != sample.str.strip()).any():
            issues["spaces"].append(col)
    return issues


def analyze_outliers(df: pd.DataFrame) -> list:
    """Use IQR to flag numeric columns with potential outliers."""
    flagged = []
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out > 0:
            flagged.append((col, n_out))
    return flagged


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — SUGGESTION BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_suggestions(df: pd.DataFrame, template: str) -> list:
    """
    Dynamically generate cleaning suggestions based on analysis results
    and the selected template.

    Each suggestion is a dict:
      { "key": str, "label": str, "detail": str, "meta": any }
    """
    suggestions = []

    # ── Duplicates ──────────────────────────────────────────────────────
    n_dup = analyze_duplicates(df)
    if n_dup > 0:
        suggestions.append({
            "key":    "remove_duplicates",
            "label":  f"Remove {n_dup:,} duplicate rows",
            "detail": "Uses pandas drop_duplicates(). Safe to apply for most datasets.",
            "meta":   None,
        })

    # ── Missing values ───────────────────────────────────────────────────
    mv = analyze_missing(df)
    for _, row in mv.iterrows():
        col = row["Column"]
        pct = row["Missing %"]
        if pct > 50:
            suggestions.append({
                "key":    f"drop_col_{col}",
                "label":  f"Drop column '{col}' (>{pct:.0f}% missing)",
                "detail": "More than half the values are missing. Dropping may be wiser than filling.",
                "meta":   col,
            })
        else:
            strategy = "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
            suggestions.append({
                "key":    f"fill_{col}",
                "label":  f"Fill missing values in '{col}' with {strategy}",
                "detail": f"Column has {row['Missing Count']} missing values ({pct}%). Fill with {strategy}.",
                "meta":   {"col": col, "strategy": strategy},
            })

    # ── Type conversions ─────────────────────────────────────────────────
    for col in analyze_type_issues(df):
        suggestions.append({
            "key":    f"convert_numeric_{col}",
            "label":  f"Convert column '{col}' to numeric type",
            "detail": "Detected that most values in this text column are actually numbers.",
            "meta":   col,
        })

    # ── Text: spaces & casing ─────────────────────────────────────────
    text_issues = analyze_text_issues(df)
    for col in text_issues["spaces"]:
        suggestions.append({
            "key":    f"trim_{col}",
            "label":  f"Trim whitespace in '{col}'",
            "detail": "Leading/trailing spaces found — can cause grouping and join issues.",
            "meta":   col,
        })
    for col in text_issues["casing"]:
        suggestions.append({
            "key":    f"titlecase_{col}",
            "label":  f"Standardize casing in '{col}' (Title Case)",
            "detail": "Mixed casing detected. Title Case is typically best for names/labels.",
            "meta":   col,
        })

    # ── Outliers ─────────────────────────────────────────────────────────
    for col, n in analyze_outliers(df):
        suggestions.append({
            "key":    f"cap_outliers_{col}",
            "label":  f"Cap outliers in '{col}' ({n} detected)",
            "detail": "IQR method: values outside [Q1−1.5·IQR, Q3+1.5·IQR] will be clamped.",
            "meta":   col,
        })

    # ── Template-specific ────────────────────────────────────────────────
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if template == "Indian Data":
        for col in [c for c in obj_cols if any(k in c.lower() for k in ["phone","mobile","contact","tel"])]:
            suggestions.append({"key": f"format_phone_{col}", "label": f"Format phone numbers in '{col}' to +91XXXXXXXXXX", "detail": "Strips non-digits and prepends +91 if needed.", "meta": col})
        for col in [c for c in obj_cols if any(k in c.lower() for k in ["city","district","town"])]:
            suggestions.append({"key": f"standardize_city_{col}", "label": f"Standardize city names in '{col}' (Title Case)", "detail": "Capitalizes city names consistently.", "meta": col})
        for col in [c for c in obj_cols + [str(c) for c in num_cols] if any(k in str(c).lower() for k in ["pin","pincode","postal","zip"])]:
            suggestions.append({"key": f"validate_pincode_{col}", "label": f"Validate 6-digit pincodes in '{col}'", "detail": "Rows with invalid pincodes will have that cell set to NaN.", "meta": col})

    elif template == "Survey Data":
        for col in obj_cols:
            vals = df[col].dropna().astype(str).str.lower().str.strip()
            if len(vals) > 0 and vals.isin(["yes","no","y","n","true","false","1","0"]).mean() > 0.5:
                suggestions.append({"key": f"normalize_yn_{col}", "label": f"Normalize Yes/No values in '{col}'", "detail": "Maps y/yes/true/1 → 'Yes' and n/no/false/0 → 'No'.", "meta": col})
        suggestions.append({"key": "drop_high_missing_rows", "label": "Drop rows where >50% of fields are missing", "detail": "Removes incomplete survey responses that would skew analysis.", "meta": None})

    elif template == "Sales/Business Data":
        for col in [c for c in num_cols if any(k in c.lower() for k in ["price","amount","revenue","cost","sale","value"])]:
            suggestions.append({"key": f"remove_neg_{col}", "label": f"Remove negative values in price column '{col}'", "detail": "Negative prices are usually data errors. Sets them to NaN.", "meta": col})
            suggestions.append({"key": f"format_currency_{col}", "label": f"Add ₹ currency label to '{col}' (convert to string)", "detail": "Formats the column as '₹1,234.00' strings for display.", "meta": col})

    elif template == "Student Data":
        for col in [c for c in obj_cols if any(k in c.lower() for k in ["name","student","firstname","lastname"])]:
            suggestions.append({"key": f"fix_name_{col}", "label": f"Fix name capitalization in '{col}'", "detail": "Applies Title Case to student names.", "meta": col})
        for col in [c for c in obj_cols if any(k in c.lower() for k in ["grade","mark","result","score","gpa"])]:
            suggestions.append({"key": f"standardize_grade_{col}", "label": f"Standardize grade format in '{col}' (uppercase)", "detail": "Converts a/b/c grades to A/B/C.", "meta": col})

    return suggestions


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — CLEANING ENGINE
# ═══════════════════════════════════════════════════════════════════════

def apply_cleaning(df: pd.DataFrame, selected_keys: list, all_suggestions: list) -> pd.DataFrame:
    """Apply only the selected cleaning steps to a copy of the DataFrame."""
    cleaned = df.copy()
    key_to_meta = {s["key"]: s["meta"] for s in all_suggestions}

    for key in selected_keys:
        meta = key_to_meta.get(key)
        try:
            if key == "remove_duplicates":
                cleaned = cleaned.drop_duplicates()

            elif key == "drop_high_missing_rows":
                threshold = int(len(cleaned.columns) * 0.5)
                cleaned = cleaned.dropna(thresh=threshold)

            elif key.startswith("drop_col_"):
                col = meta
                if col in cleaned.columns:
                    cleaned = cleaned.drop(columns=[col])

            elif key.startswith("fill_"):
                col, strategy = meta["col"], meta["strategy"]
                if col in cleaned.columns:
                    if strategy == "median":
                        fill_val = cleaned[col].median()
                    elif strategy == "mean":
                        fill_val = cleaned[col].mean()
                    else:
                        mode_vals = cleaned[col].mode()
                        fill_val  = mode_vals.iloc[0] if not mode_vals.empty else None
                    if fill_val is not None:
                        cleaned[col] = cleaned[col].fillna(fill_val)

            elif key.startswith("convert_numeric_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = pd.to_numeric(
                        cleaned[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
                    )

            elif key.startswith("trim_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].astype(str).str.strip()

            elif key.startswith("titlecase_") or key.startswith("standardize_city_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.title()

            elif key.startswith("cap_outliers_"):
                col = meta
                if col in cleaned.columns:
                    q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
                    iqr = q3 - q1
                    cleaned[col] = cleaned[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

            elif key.startswith("format_phone_"):
                col = meta
                if col in cleaned.columns:
                    def fmt_phone(val):
                        if pd.isna(val): return val
                        digits = re.sub(r"\D", "", str(val))
                        if digits.startswith("91") and len(digits) == 12: return f"+{digits}"
                        elif len(digits) == 10: return f"+91{digits}"
                        return val
                    cleaned[col] = cleaned[col].apply(fmt_phone)

            elif key.startswith("validate_pincode_"):
                col = meta
                if col in cleaned.columns:
                    def validate_pin(val):
                        if pd.isna(val): return val
                        digits = re.sub(r"\D", "", str(val))
                        return digits if len(digits) == 6 else np.nan
                    cleaned[col] = cleaned[col].apply(validate_pin)

            elif key.startswith("normalize_yn_"):
                col = meta
                if col in cleaned.columns:
                    yn_map = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
                              "no": "No",  "n": "No",  "false": "No", "0": "No"}
                    cleaned[col] = (cleaned[col].astype(str).str.lower().str.strip()
                                                .map(yn_map).fillna(cleaned[col]))

            elif key.startswith("remove_neg_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].where(cleaned[col] >= 0, other=np.nan)

            elif key.startswith("format_currency_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].apply(
                        lambda x: f"₹{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                    )

            elif key.startswith("fix_name_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.title()

            elif key.startswith("standardize_grade_"):
                col = meta
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].astype(str).str.strip().str.upper()

        except Exception as e:
            st.warning(f"⚠️  Could not apply step '{key}': {e}")

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════

def render_summary(df: pd.DataFrame, label: str = "Dataset"):
    """Render a tabbed data summary section."""
    st.subheader(f"📋 {label} Summary")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab1, tab2, tab3 = st.tabs(["Column Info", "Numeric Stats", "Missing Values"])

    with tab1:
        info_df = pd.DataFrame({
            "Column":   df.columns,
            "Dtype":    [str(dt) for dt in df.dtypes],
            "Non-Null": df.count().values,
            "Unique":   [df[c].nunique() for c in df.columns],
            "Sample":   [str(df[c].iloc[0]) if len(df) > 0 else "—" for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with tab2:
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab3:
        mv = analyze_missing(df)
        if mv.empty:
            st.success("✅ No missing values detected!")
        else:
            st.dataframe(mv, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — CHARTS
# ═══════════════════════════════════════════════════════════════════════

def render_charts(df: pd.DataFrame):
    """Render mini histograms for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return

    st.subheader("📊 Quick Visual Inspection")
    cols_to_show = num_cols[:6]
    n_cols = min(3, len(cols_to_show))
    col_widgets = st.columns(n_cols)

    for i, col in enumerate(cols_to_show):
        with col_widgets[i % n_cols]:
            fig = px.histogram(df, x=col, nbins=30, title=col,
                               color_discrete_sequence=["#00e5a0"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", title_font_family="Space Mono", title_font_size=13,
                margin=dict(l=10, r=10, t=36, b=10), height=220, showlegend=False,
                xaxis=dict(gridcolor="#2a3040"), yaxis=dict(gridcolor="#2a3040"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — DOWNLOAD HELPER
# ═══════════════════════════════════════════════════════════════════════

def get_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to CSV bytes for download."""
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════

def main():

    # ── Hero Banner ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🧹 CleanData AI</h1>
        <p>Smart, guided, semi-automated data cleaning — upload a CSV and get instant suggestions.</p>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()
        st.markdown("### 📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"],
            help="Max 200 MB. Ensure the first row contains column headers.",
        )
        st.divider()
        st.markdown("### 🗂️ Dataset Template")
        template = st.selectbox(
            "Select a template",
            options=["General Dataset", "Survey Data", "Sales/Business Data", "Student Data", "Indian Data"],
            help="Templates add domain-specific cleaning rules.",
        )
        st.divider()
        st.markdown(
            "<small style='color:#64748b'>CleanData AI · v1.0<br>Built with Streamlit + pandas</small>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════
    # GUARD — no file uploaded yet
    # ═══════════════════════════════════════════════════════════════════
    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to get started.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Step 1 — Upload**\n\nDrop any CSV. The app handles analysis automatically.")
        with c2:
            st.markdown("**Step 2 — Review**\n\nInspect detected issues: missing values, duplicates, outliers, type mismatches.")
        with c3:
            st.markdown("**Step 3 — Clean & Download**\n\nPick your cleaning steps, apply them, and download the result.")
        return

    # ═══════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════════════════
    df_original = load_csv(uploaded_file)

    # ── Overview Metrics ─────────────────────────────────────────────
    st.markdown("## 📥 Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",           f"{df_original.shape[0]:,}")
    m2.metric("Columns",        f"{df_original.shape[1]:,}")
    m3.metric("Missing Cells",  f"{df_original.isnull().sum().sum():,}")
    m4.metric("Duplicate Rows", f"{analyze_duplicates(df_original):,}")

    with st.expander("🔍 Preview first 10 rows", expanded=True):
        st.dataframe(df_original.head(10), use_container_width=True)

    render_charts(df_original)
    render_summary(df_original, label="Original")

    # ═══════════════════════════════════════════════════════════════════
    # GENERATE SUGGESTIONS
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header"></div>', unsafe_allow_html=True)
    st.markdown(f"## 🤖 AI Suggestions  <span class='badge badge-blue'>{template}</span>", unsafe_allow_html=True)

    suggestions = build_suggestions(df_original, template)

    if not suggestions:
        st.success("🎉 No issues detected! Your dataset looks clean.")
        return

    st.markdown(f"Found **{len(suggestions)}** potential cleaning steps for your dataset:")

    # ── Categorize suggestions ────────────────────────────────────────
    def categorize(key: str) -> str:
        if "duplicate"      in key: return "🔁 Duplicates"
        if "drop_col"       in key: return "🗑️ Drop Columns"
        if "fill_"          in key: return "🩹 Fill Missing"
        if "convert"        in key: return "🔀 Type Fixes"
        if "trim"           in key: return "✂️ Whitespace"
        if any(x in key for x in ["titlecase", "fix_name", "standardize_city"]): return "🔡 Text Casing"
        if "cap_outlier"    in key: return "📐 Outliers"
        if any(x in key for x in ["phone", "pincode"]): return "🇮🇳 Indian Data"
        if any(x in key for x in ["yn", "high_missing"]): return "📋 Survey"
        if any(x in key for x in ["neg", "currency"]): return "💰 Sales/Business"
        if "grade"          in key: return "🎓 Student Data"
        return "🔧 Other"

    categories: dict = {}
    for s in suggestions:
        cat = categorize(s["key"])
        categories.setdefault(cat, []).append(s)

    selected_keys = []
    for cat_name, cat_suggestions in categories.items():
        with st.expander(f"{cat_name} ({len(cat_suggestions)} steps)", expanded=True):
            for s in cat_suggestions:
                checked = st.checkbox(s["label"], key=f"cb_{s['key']}", value=True, help=s["detail"])
                if checked:
                    selected_keys.append(s["key"])

    # ═══════════════════════════════════════════════════════════════════
    # APPLY CLEANING
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-header"></div>', unsafe_allow_html=True)
    st.markdown("## 🚀 Apply Cleaning")

    col_btn, col_info = st.columns([3, 9])
    with col_btn:
        apply_clicked = st.button("✨ Apply Selected Cleaning", use_container_width=True)
    with col_info:
        st.markdown(
            f"<small style='color:#64748b'>{len(selected_keys)} of {len(suggestions)} steps selected</small>",
            unsafe_allow_html=True,
        )

    if apply_clicked:
        if not selected_keys:
            st.warning("No steps selected. Please check at least one suggestion.")
            return

        with st.spinner("Cleaning your data..."):
            df_cleaned = apply_cleaning(df_original, selected_keys, suggestions)

        st.success(f"✅ Done! Applied {len(selected_keys)} cleaning step(s).")

        # ── Before / After Comparison ─────────────────────────────
        st.markdown("### 📊 Before vs After")
        b1, b2, b3, b4 = st.columns(4)
        row_diff   = df_original.shape[0] - df_cleaned.shape[0]
        col_diff   = df_original.shape[1] - df_cleaned.shape[1]
        miss_before = int(df_original.isnull().sum().sum())
        miss_after  = int(df_cleaned.isnull().sum().sum())
        miss_diff   = miss_before - miss_after

        b1.metric("Rows",          f"{df_cleaned.shape[0]:,}",  delta=f"-{row_diff}"  if row_diff  > 0 else "no change")
        b2.metric("Columns",       f"{df_cleaned.shape[1]:,}",  delta=f"-{col_diff}"  if col_diff  > 0 else "no change")
        b3.metric("Missing Cells", f"{miss_after:,}",           delta=f"-{miss_diff}" if miss_diff > 0 else "no change")
        b4.metric("Duplicates",    f"{analyze_duplicates(df_cleaned):,}")

        st.markdown("### 🔍 Cleaned Data Preview")
        st.dataframe(df_cleaned.head(10), use_container_width=True)
        render_summary(df_cleaned, label="Cleaned")

        # ── Download ──────────────────────────────────────────────
        st.markdown('<div class="step-header"></div>', unsafe_allow_html=True)
        st.markdown("## 💾 Download")
        csv_bytes = get_csv_bytes(df_cleaned)
        filename  = (uploaded_file.name or "data").replace(".csv", "") + "_cleaned.csv"
        st.download_button(
            label="⬇️  Download Cleaned CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )
        st.caption(f"File: `{filename}` · {len(csv_bytes)/1024:.1f} KB")


if __name__ == "__main__":
    main()
