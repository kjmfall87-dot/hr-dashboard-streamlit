# app.py
# ============================================================
# HR Dashboard (Game Company ~400 employees, founded 6 years ago)
# Streamlit + Plotly
# - Default: realistic synthetic data generator
# - Optional: CSV/XLSX upload
# - Dashboards 1~5 (per your plan)
# - Cascading filter: 상위조직 -> 팀
# - No hard dependency on reportlab (PDF is optional)
# ============================================================

import io
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Optional PDF (reportlab)
# -----------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="HR 대시보드 (게임회사 샘플)", layout="wide")
TODAY = pd.Timestamp(date.today())

# Month approximation (pandas Timedelta doesn't support "M")
DAYS_PER_MONTH = 30.4375  # 365.25 / 12


# -----------------------------
# Helpers
# -----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]
    return df


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def safe_div(a, b):
    if b in [0, 0.0, None] or pd.isna(b):
        return np.nan
    return a / b


def month_period(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").astype(str)


def build_pdf_bytes(title: str, lines: list[str]) -> bytes:
    """Return PDF bytes. If reportlab isn't installed, return empty bytes."""
    if not REPORTLAB_OK:
        return b""

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    x0, y = 18 * mm, h - 18 * mm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x0, y, title)
    y -= 10 * mm

    c.setFont("Helvetica", 10)
    for line in lines:
        if y < 18 * mm:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = h - 18 * mm
        c.drawString(x0, y, str(line)[:140])
        y -= 6 * mm

    c.showPage()
    c.save()
    return buf.getvalue()


# -----------------------------
# Synthetic Data Generator
# -----------------------------
@dataclass
class CompanySpec:
    n_employees_current: int = 400
    years_since_founded: int = 6


def generate_synthetic_hr_data(spec: CompanySpec, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = (TODAY - pd.DateOffset(years=spec.years_since_founded)).normalize()

    # Org / Team (game-company realistic)
    org_map = {
        "NX스튜디오": ["클라이언트", "서버", "게임플레이", "테크아트", "QA", "프로덕션"],
        "아트스튜디오": ["캐릭터아트", "배경아트", "UIUX", "VFX", "애니메이션"],
        "라이브옵스": ["운영", "커뮤니티", "CS", "데이터분석"],
        "퍼블리싱": ["UA마케팅", "브랜드", "사업개발", "로컬라이제이션"],
        "코퍼레이트": ["HR", "재무", "총무", "법무", "IT지원"],
    }

    roles_by_team = {
        "클라이언트": ["Client Engineer", "Unity Engineer", "Tools Engineer"],
        "서버": ["Backend Engineer", "SRE", "Data Engineer"],
        "게임플레이": ["Gameplay Engineer", "Combat Designer", "Level Designer"],
        "테크아트": ["Tech Artist", "Pipeline TD"],
        "QA": ["QA Engineer", "Test Analyst"],
        "프로덕션": ["Producer", "Project Manager"],

        "캐릭터아트": ["Character Artist"],
        "배경아트": ["Environment Artist"],
        "UIUX": ["UI Designer", "UX Designer"],
        "VFX": ["VFX Artist"],
        "애니메이션": ["Animator", "Motion Designer"],

        "운영": ["Live Ops Manager"],
        "커뮤니티": ["Community Manager"],
        "CS": ["CS Specialist"],
        "데이터분석": ["Data Analyst", "Product Analyst"],

        "UA마케팅": ["UA Marketer", "Growth Marketer"],
        "브랜드": ["Brand Marketer", "Content Marketer"],
        "사업개발": ["BizDev", "Partnership Manager"],
        "로컬라이제이션": ["Localization Manager"],

        "HR": ["HRBP", "Recruiter", "People Ops"],
        "재무": ["Accountant", "FP&A"],
        "총무": ["GA Specialist"],
        "법무": ["Legal Counsel"],
        "IT지원": ["IT Support", "Security Admin"],
    }

    grades = ["주니어", "미들", "시니어", "리드", "매니저"]
    grade_probs = np.array([0.18, 0.42, 0.26, 0.08, 0.06])

    emp_types = ["정규직", "계약직"]
    emp_probs = np.array([0.92, 0.08])

    genders = ["남", "여"]
    gender_probs = np.array([0.62, 0.38])

    def sample_age(n):
        mix = rng.choice([0, 1, 2], size=n, p=[0.45, 0.35, 0.20])
        ages = np.empty(n, dtype=int)
        for i, m in enumerate(mix):
            if m == 0:
                ages[i] = int(np.clip(rng.normal(28, 2.5), 22, 36))
            elif m == 1:
                ages[i] = int(np.clip(rng.normal(33, 3.0), 24, 45))
            else:
                ages[i] = int(np.clip(rng.normal(39, 4.0), 28, 55))
        return ages

    # Generate hires over 6 years then apply attrition to land near 400 active
    total_hires = int(520)

    months = pd.period_range(start=start_date.to_period("M"), end=TODAY.to_period("M"), freq="M")
    m_idx = np.arange(len(months))

    center = len(months) * 0.55
    spread = len(months) * 0.22
    growth = np.exp(-((m_idx - center) ** 2) / (2 * spread ** 2))
    season = 1.0 + 0.25 * np.sin(2 * np.pi * (m_idx / 12.0 - 0.15)) + 0.18 * np.sin(2 * np.pi * (m_idx / 6.0))
    w = np.clip(growth * season, 1e-6, None)
    w = w / w.sum()

    hire_months = rng.choice(months.astype(str), size=total_hires, p=w)
    hire_dates = pd.to_datetime(hire_months) + pd.to_timedelta(rng.integers(0, 27, size=total_hires), unit="D")
    hire_dates = pd.Series(hire_dates).clip(lower=start_date, upper=TODAY)

    orgs = list(org_map.keys())
    org_probs = np.array([0.44, 0.20, 0.14, 0.10, 0.12])
    chosen_org = rng.choice(orgs, size=total_hires, p=org_probs)

    teams = []
    jobs = []
    for o in chosen_org:
        t = rng.choice(org_map[o])
        teams.append(t)
        jobs.append(rng.choice(roles_by_team[t]))

    teams = np.array(teams)
    jobs = np.array(jobs)

    base_grade = rng.choice(grades, size=total_hires, p=grade_probs)
    recent_mask = hire_dates >= (TODAY - pd.DateOffset(months=18))
    if recent_mask.any():
        base_grade[recent_mask.values] = rng.choice(
            grades,
            size=int(recent_mask.sum()),
            p=np.array([0.26, 0.46, 0.20, 0.05, 0.03])
        )

    emp_type = rng.choice(emp_types, size=total_hires, p=emp_probs)
    gender = rng.choice(genders, size=total_hires, p=gender_probs)
    age = sample_age(total_hires)

    # ---- FIX: tenure months using day-based approximation (no "M" timedelta) ----
    tenure_months_now = (TODAY - hire_dates).dt.days / DAYS_PER_MONTH

    prob_exit_base = 0.22
    early_factor = np.clip(1.8 - (tenure_months_now / 18.0), 0.7, 1.8)
    grade_factor = np.where(base_grade == "주니어", 1.20,
                    np.where(base_grade == "미들", 1.00,
                    np.where(base_grade == "시니어", 0.92,
                    np.where(base_grade == "리드", 0.88, 0.90))))
    type_factor = np.where(emp_type == "계약직", 1.35, 1.0)

    raw_p = prob_exit_base * early_factor * grade_factor * type_factor
    raw_p = np.clip(raw_p, 0.02, 0.60)

    exited = rng.random(total_hires) < raw_p

    leave_dates = np.full(int(total_hires), np.datetime64("NaT"), dtype="datetime64[ns]")
    voluntary = np.array([np.nan] * total_hires, dtype=object)
    leave_reason = np.array([np.nan] * total_hires, dtype=object)

    reasons = ["급여/보상", "업무 강도", "성장 기회", "조직 문화", "개인 사유", "커리어 전환", "계약 만료", "직무 부적합"]

    for i in range(total_hires):
        if not exited[i]:
            continue

        max_m = max(1.0, float(tenure_months_now.iloc[i]))
        m = rng.gamma(shape=1.6, scale=6.0)  # more mass early
        m = float(np.clip(m, 1.0, max_m))

        d = hire_dates.iloc[i] + pd.DateOffset(days=int(m * DAYS_PER_MONTH))
        if d > TODAY:
            d = TODAY - pd.Timedelta(days=int(rng.integers(1, 20)))
        leave_dates[i] = d

        if emp_type[i] == "계약직" and rng.random() < 0.55:
            voluntary[i] = "비자발"
            leave_reason[i] = "계약 만료"
        else:
            voluntary[i] = "자발" if rng.random() < 0.82 else "비자발"
            if m < 6:
                leave_reason[i] = rng.choice(["조직 문화", "직무 부적합", "개인 사유", "업무 강도"])
            elif m < 12:
                leave_reason[i] = rng.choice(["성장 기회", "업무 강도", "조직 문화", "급여/보상"])
            elif m < 36:
                leave_reason[i] = rng.choice(["성장 기회", "급여/보상", "커리어 전환", "업무 강도"])
            else:
                leave_reason[i] = rng.choice(["커리어 전환", "급여/보상", "개인 사유", "성장 기회"])

    df = pd.DataFrame({
        "사번": [f"NX{100000+i}" for i in range(total_hires)],
        "성명": [f"직원{i+1}" for i in range(total_hires)],
        "상위조직": chosen_org,
        "팀": teams,
        "직무": jobs,
        "직급/직책": base_grade,
        "구분": emp_type,
        "성별": gender,
        "나이": age,
        "입사일": hire_dates.values,
        "퇴사일": leave_dates,
        "자발/비자발": voluntary,
        "퇴사사유": leave_reason,
    })

    df["입사일"] = to_dt(df["입사일"])
    df["퇴사일"] = to_dt(df["퇴사일"])
    df["재직여부"] = df["퇴사일"].isna()

    # Adjust to target headcount ~400
    current = int(df["재직여부"].sum())
    target = spec.n_employees_current

    if current > target:
        need = current - target
        actives = df[df["재직여부"]].copy()
        actives["tenure_m"] = (TODAY - actives["입사일"]).dt.days / DAYS_PER_MONTH
        actives = actives.sort_values("tenure_m")
        pick = actives.head(need).index
        for idx in pick:
            d = TODAY - pd.Timedelta(days=int(rng.integers(5, 90)))
            df.loc[idx, "퇴사일"] = d
            df.loc[idx, "재직여부"] = False
            df.loc[idx, "자발/비자발"] = "자발" if rng.random() < 0.85 else "비자발"
            if pd.isna(df.loc[idx, "퇴사사유"]):
                df.loc[idx, "퇴사사유"] = rng.choice(reasons)
    elif current < target:
        need = target - current
        leavers = df[~df["재직여부"]].copy().sort_values("퇴사일", ascending=False)
        pick = leavers.head(need).index
        df.loc[pick, "퇴사일"] = pd.NaT
        df.loc[pick, "재직여부"] = True
        df.loc[pick, "자발/비자발"] = np.nan
        df.loc[pick, "퇴사사유"] = np.nan

    # Age band
    bins = [0, 24, 29, 34, 39, 44, 49, 200]
    labels = ["~24", "25~29", "30~34", "35~39", "40~44", "45~49", "50+"]
    df["연령대"] = pd.cut(df["나이"], bins=bins, labels=labels, right=True)

    # ---- FIX: 근속기간(개월) also using day-based approximation ----
    end_dt = df["퇴사일"].fillna(TODAY)
    df["근속기간(개월)"] = ((end_dt - df["입사일"]).dt.days / DAYS_PER_MONTH).round(1)
    df["근속연수"] = (df["근속기간(개월)"] / 12.0).round(2)
    df["1년미만"] = df["근속기간(개월)"] < 12

    return df


# -----------------------------
# Metrics (month-based)
# -----------------------------
def month_range(df: pd.DataFrame) -> list[pd.Period]:
    dates = []
    if "입사일" in df.columns:
        dates.append(df["입사일"].dropna())
    if "퇴사일" in df.columns:
        dates.append(df["퇴사일"].dropna())
    if not dates:
        return []
    all_d = pd.concat(dates)
    if all_d.empty:
        return []
    start = all_d.min().to_period("M")
    end = TODAY.to_period("M")
    return list(pd.period_range(start=start, end=end, freq="M"))


def headcount_at_month_end(df: pd.DataFrame, m: pd.Period) -> int:
    if "입사일" not in df.columns:
        return int(len(df))
    month_end = (m.to_timestamp("M") + pd.offsets.MonthEnd(0))
    joined = df["입사일"].notna() & (df["입사일"] <= month_end)
    if "퇴사일" in df.columns:
        not_left = df["퇴사일"].isna() | (df["퇴사일"] > month_end)
    else:
        not_left = True
    return int((joined & not_left).sum())


def monthly_headcount(df: pd.DataFrame) -> pd.DataFrame:
    ms = month_range(df)
    if not ms:
        return pd.DataFrame(columns=["월", "월말인원"])
    rows = [{"월": str(m), "월말인원": headcount_at_month_end(df, m)} for m in ms]
    return pd.DataFrame(rows, columns=["월", "월말인원"])



def monthly_leavers(df: pd.DataFrame) -> pd.DataFrame:
    if "퇴사일" not in df.columns:
        return pd.DataFrame(columns=["월", "퇴사자수"])
    x = df["퇴사일"].dropna()
    if x.empty:
        return pd.DataFrame(columns=["월", "퇴사자수"])
    out = month_period(x).value_counts().sort_index().reset_index()
    out.columns = ["월", "퇴사자수"]
    return out


def monthly_turnover_rate(df: pd.DataFrame) -> pd.DataFrame:
    hc = monthly_headcount(df)
    lv = monthly_leavers(df)
    if hc.empty:
        return pd.DataFrame(columns=["월", "이직률(%)"])
    merged = hc.merge(lv, on="월", how="left").fillna({"퇴사자수": 0})
    merged["월말인원_prev"] = merged["월말인원"].shift(1)
    merged["평균인원"] = (merged["월말인원_prev"] + merged["월말인원"]) / 2
    merged["평균인원"] = merged["평균인원"].replace(0, np.nan)
    merged["이직률(%)"] = (merged["퇴사자수"] / merged["평균인원"]) * 100
    merged["이직률(%)"] = merged["이직률(%)"].replace([np.inf, -np.inf], np.nan)
    return merged[["월", "이직률(%)"]]


def last_two(series_df: pd.DataFrame, col: str):
    if series_df is None or series_df.empty or col not in series_df.columns:
        return (np.nan, np.nan, np.nan)

    s = series_df.dropna(subset=[col]).copy()
    if len(s) == 0:
        return (np.nan, np.nan, np.nan)
    if len(s) == 1:
        v = float(s[col].iloc[-1])
        return (v, np.nan, np.nan)

    now = float(s[col].iloc[-1])
    prev = float(s[col].iloc[-2])
    diff = now - prev
    pct_change = (diff / prev * 100) if prev != 0 else np.nan
    return (now, diff, pct_change)



def tail_months(series_df: pd.DataFrame, n=12):
    if series_df.empty or "월" not in series_df.columns:
        return series_df
    return series_df.tail(n)


def cohort_retention(df: pd.DataFrame, months_points=(3, 6, 12), by_dim=None, top_n=6):
    if "입사일" not in df.columns or "재직여부" not in df.columns:
        return pd.DataFrame()

    base = df.copy()
    base = base[base["입사일"].notna()].copy()
    base["입사월"] = base["입사일"].dt.to_period("M").astype(str)

    cutoff = (TODAY - pd.DateOffset(months=24)).to_period("M")
    base = base[base["입사일"].dt.to_period("M") >= cutoff]
    if base.empty:
        return pd.DataFrame()

    if by_dim and by_dim in base.columns:
        cat_counts = base[by_dim].dropna().astype(str).value_counts().head(top_n)
        cats = cat_counts.index.tolist()
        base = base[base[by_dim].astype(str).isin(cats)]
    else:
        by_dim = None

    rows = []
    for join_m, g in base.groupby("입사월"):
        for p in months_points:
            eligible = g[(TODAY - g["입사일"]).dt.days >= int(p * DAYS_PER_MONTH)]
            if eligible.empty:
                rate = np.nan
            else:
                rate = eligible["재직여부"].mean() * 100
            rows.append({"입사월": join_m, "개월": p, "정착률(%)": rate})
    return pd.DataFrame(rows)


# -----------------------------
# App Header
# -----------------------------
st.title("🎮 HR 대시보드 (400명 · 설립 6년차 게임회사 예시)")


# -----------------------------
# Sidebar: Data source
# -----------------------------
with st.sidebar:
    st.header("데이터 설정 ⚙️")
    mode = st.radio("데이터 소스", ["샘플 데이터(권장)", "파일 업로드"], index=0)
    seed = st.number_input("샘플 데이터 시드(바꾸면 데이터가 달라짐)", min_value=1, max_value=9999, value=42, step=1)

uploaded = None
df = None

if mode == "파일 업로드":
    uploaded = st.file_uploader("CSV 또는 XLSX 업로드", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("파일을 업로드하면 대시보드가 생성됩니다 🙂")
        st.stop()

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("시트 선택", xls.sheet_names, index=0)
            df = pd.read_excel(uploaded, sheet_name=sheet)
        df = clean_columns(df)
    except Exception:
        st.error("파일을 읽는 중 오류가 발생했습니다. CSV/XLSX 형식과 컬럼을 확인해 주세요.")
        st.stop()
else:
    df = generate_synthetic_hr_data(CompanySpec(), seed=int(seed))

# Normalize expected columns (upload-safe)
for col in ["입사일", "퇴사일"]:
    if col in df.columns:
        df[col] = to_dt(df[col])

if "재직여부" not in df.columns and "퇴사일" in df.columns:
    df["재직여부"] = df["퇴사일"].isna()

# Create tenure fields if missing
if "근속기간(개월)" not in df.columns and "입사일" in df.columns:
    end_dt = df["퇴사일"].fillna(TODAY) if "퇴사일" in df.columns else pd.Series([TODAY] * len(df))
    # ---- FIX: use day-based approx ----
    df["근속기간(개월)"] = ((end_dt - df["입사일"]).dt.days / DAYS_PER_MONTH).round(1)

if "근속연수" not in df.columns and "근속기간(개월)" in df.columns:
    df["근속연수"] = (pd.to_numeric(df["근속기간(개월)"], errors="coerce") / 12.0).round(2)

if "1년미만" not in df.columns and "근속기간(개월)" in df.columns:
    df["1년미만"] = pd.to_numeric(df["근속기간(개월)"], errors="coerce") < 12


# -----------------------------
# Sidebar Filters (상위조직 -> 팀)
# -----------------------------
with st.sidebar:
    st.header("필터 🔎")

    def multiselect_safe(label, col, base):
        if col not in base.columns:
            st.caption(f"⚠️ {col} 컬럼 없음")
            return []
        vals = base[col].dropna().astype(str).str.strip().unique().tolist()
        vals = sorted(vals)
        return st.multiselect(label, vals, default=[])

    org_sel = multiselect_safe("상위조직", "상위조직", df)
    df_org = df.copy()
    if org_sel and "상위조직" in df_org.columns:
        df_org = df_org[df_org["상위조직"].astype(str).isin(org_sel)]

    team_sel = multiselect_safe("팀(상위조직 선택 시 해당 팀만)", "팀", df_org)
    df_team = df_org.copy()
    if team_sel and "팀" in df_team.columns:
        df_team = df_team[df_team["팀"].astype(str).isin(team_sel)]

    grade_sel = multiselect_safe("직급/직책", "직급/직책", df_team)
    job_sel = multiselect_safe("직무", "직무", df_team)
    gender_sel = multiselect_safe("성별", "성별", df_team) if "성별" in df_team.columns else []
    type_sel = multiselect_safe("구분", "구분", df_team) if "구분" in df_team.columns else []

    filtered = df_team.copy()
    if grade_sel and "직급/직책" in filtered.columns:
        filtered = filtered[filtered["직급/직책"].astype(str).isin(grade_sel)]
    if job_sel and "직무" in filtered.columns:
        filtered = filtered[filtered["직무"].astype(str).isin(job_sel)]
    if gender_sel and "성별" in filtered.columns:
        filtered = filtered[filtered["성별"].astype(str).isin(gender_sel)]
    if type_sel and "구분" in filtered.columns:
        filtered = filtered[filtered["구분"].astype(str).isin(type_sel)]

    st.divider()
    chart_months = st.slider("트렌드 조회 기간(개월)", 6, 36, 12, step=6)

st.caption(f"✅ 데이터 로딩 완료 | 전체 행: {len(df):,} | 현재 필터 결과: {len(filtered):,}")

def monthly_hires(df: pd.DataFrame) -> pd.DataFrame:
    if "입사일" not in df.columns:
        return pd.DataFrame(columns=["월", "입사자수"])
    x = df["입사일"].dropna()
    if x.empty:
        return pd.DataFrame(columns=["월", "입사자수"])
    out = x.dt.to_period("M").astype(str).value_counts().sort_index().reset_index()
    out.columns = ["월", "입사자수"]
    return out

# -----------------------------
# Precompute series
# -----------------------------
hc = monthly_headcount(filtered)
hi = monthly_hires(filtered)
lv = monthly_leavers(filtered)
to = monthly_turnover_rate(filtered)

hc_tail = tail_months(hc, chart_months)
to_tail = tail_months(to, chart_months)

hc_now, hc_diff, hc_pct = last_two(hc, "월말인원")
to_now, to_diff, to_pct = last_two(to, "이직률(%)")

avg_tenure = float(filtered["근속연수"].mean()) if "근속연수" in filtered.columns and filtered["근속연수"].notna().any() else np.nan
u1 = float(filtered["1년미만"].mean() * 100) if "1년미만" in filtered.columns and filtered["1년미만"].notna().any() else np.nan

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard 1 · 경영진 요약",
    "Dashboard 2 · 인력 구조",
    "Dashboard 3 · 조직 리스크",
    "Dashboard 4 · 신규 입사자 품질",
    "Dashboard 5 · 이직 패턴 심층",
])

# ============================================================
# Dashboard 1
# ============================================================
with tab1:
    st.subheader("📌 KPI 카드 (4개)")
    c1, c2, c3, c4 = st.columns(4)

    if not pd.isna(hc_now):
        c1.metric("현재 인원(월말)", f"{int(hc_now):,}명", delta=(f"{hc_diff:+.0f}명 ({hc_pct:+.1f}%)" if not pd.isna(hc_diff) else None))
    else:
        c1.info("현재 인원: 데이터 없음")

    if not pd.isna(to_now):
        c2.metric("이직률(월별)", f"{to_now:.2f}%", delta=(f"{to_diff:+.2f}%p" if not pd.isna(to_diff) else None))
    else:
        c2.info("이직률: 데이터 없음")

    if not pd.isna(avg_tenure):
        c3.metric("평균 근속연수", f"{avg_tenure:.2f}년")
    else:
        c3.info("평균 근속연수: 데이터 없음")

    if not pd.isna(u1):
        c4.metric("1년 미만 근속자 비율", f"{u1:.1f}%")
    else:
        c4.info("1년 미만 비율: 데이터 없음")

    st.divider()
    st.subheader("📈 트렌드 차트 (2개)")
    left, right = st.columns(2)

    # hires vs leavers
    with left:
        if not hi.empty or not lv.empty:
            trend = hi.rename(columns={"입사자수": "입사"}).merge(
                lv.rename(columns={"퇴사자수": "퇴사"}), on="월", how="outer"
            ).fillna(0).sort_values("월")
            trend = trend.tail(chart_months)
            fig = px.line(trend, x="월", y=["입사", "퇴사"], markers=True, title="월별 인원 변동 (입사/퇴사)")
            fig.update_layout(legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("입사/퇴사 데이터가 없어 공란 처리")

    # turnover
    with right:
        if not to_tail.empty:
            fig = px.line(to_tail, x="월", y="이직률(%)", markers=True, title=f"이직률 추이 (최근 {chart_months}개월)")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("이직률 계산 데이터가 없어 공란 처리")

    st.divider()
    st.subheader("🧠 인사이트 섹션")

    # 요약 텍스트(짧게)
    summary_lines = []
    if ("trend" in locals()) and isinstance(trend, pd.DataFrame) and not trend.empty:
        last_m = trend["월"].iloc[-1]
        last_h = int(trend["입사"].iloc[-1])
        last_l = int(trend["퇴사"].iloc[-1])
        summary_lines.append(f"• {last_m} 인원 변동: 입사 {last_h}명 / 퇴사 {last_l}명")

    if "상위조직" in filtered.columns and "퇴사일" in filtered.columns and filtered["퇴사일"].notna().any():
        cut90 = TODAY - pd.Timedelta(days=90)
        recent_leavers = filtered[(filtered["퇴사일"].notna()) & (filtered["퇴사일"] >= cut90)]
        if not recent_leavers.empty:
            top_org = recent_leavers["상위조직"].dropna().astype(str).value_counts().head(3)
            org_txt = ", ".join([f"{k}({v})" for k, v in top_org.items()])
            summary_lines.append(f"• 최근 3개월 퇴사 상위 조직: {org_txt}")
        else:
            summary_lines.append("• 최근 3개월 퇴사: 유의미한 데이터 없음")
    else:
        summary_lines.append("• 이직률 변화 주요 조직: (데이터 없음)")

    a, b = st.columns([1.4, 1])
    with a:
        st.markdown("#### 변화 요약 텍스트")
        st.write("\n".join(summary_lines) if summary_lines else "데이터가 부족하여 요약이 비어 있습니다.")

    # 이슈 알림(룰 기반)
    issues = []
    with b:
        st.markdown("#### 주요 이슈 알림 🔔")
        if not pd.isna(u1) and u1 >= 30:
            issues.append(f"• 1년 미만 비율 {u1:.1f}% (30% 이상 경고)")

        if "상위조직" in filtered.columns and "퇴사일" in filtered.columns and filtered["퇴사일"].notna().any():
            cut30 = TODAY - pd.Timedelta(days=30)
            lv30 = filtered[(filtered["퇴사일"].notna()) & (filtered["퇴사일"] >= cut30)]
            if not lv30.empty:
                hot = lv30["상위조직"].dropna().astype(str).value_counts()
                hot = hot[hot >= 3].head(5)
                for org, cnt in hot.items():
                    issues.append(f"• {org}: 최근 30일 퇴사 {cnt}명 (주의)")

        if issues:
            st.warning("\n".join(issues))
        else:
            st.info("현재 표시할 이슈가 없거나 데이터가 부족합니다.")

    st.markdown('> **"지금 우리 조직은 건강한가? 어디를 봐야 하는가?"**')

    # PDF (optional)
    st.divider()
    st.markdown("#### 📄 요약 리포트 다운로드")
    if REPORTLAB_OK:
        issue_lines = issues if issues else ["• (없음/데이터 부족)"]
        pdf_lines = [
            f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"현재 인원(월말): {int(hc_now):,}명" if not pd.isna(hc_now) else "현재 인원(월말): (데이터 없음)",
            f"이직률(월별): {to_now:.2f}%" if not pd.isna(to_now) else "이직률(월별): (데이터 없음)",
            f"평균 근속연수: {avg_tenure:.2f}년" if not pd.isna(avg_tenure) else "평균 근속연수: (데이터 없음)",
            f"1년 미만 비율: {u1:.1f}%" if not pd.isna(u1) else "1년 미만 비율: (데이터 없음)",
            "",
            "[변화 요약]",
            *summary_lines,
            "",
            "[주요 이슈]",
            *issue_lines
        ]
        pdf_bytes = build_pdf_bytes("HR 대시보드 요약 리포트", pdf_lines)
        st.download_button("PDF 다운로드", data=pdf_bytes, file_name="hr_exec_summary.pdf", mime="application/pdf")
    else:
        st.info("PDF 다운로드는 reportlab 설치 후 사용 가능해요.  pip install reportlab")

# ============================================================
# Dashboard 2
# ============================================================
with tab2:
    st.subheader("👥 인력 구조 분석 (균형/다양성)")

    a, b = st.columns(2)

    with a:
        st.markdown("#### 상위조직별 인원 (재직 기준)")
        if "상위조직" in filtered.columns and "재직여부" in filtered.columns:
            base = filtered[filtered["재직여부"] == True]
            g = base["상위조직"].dropna().astype(str).value_counts().reset_index()
            g.columns = ["상위조직", "인원"]
            fig = px.bar(g, x="상위조직", y="인원", text="인원")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="인원", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("상위조직/재직여부 데이터가 부족합니다.")

        st.markdown("#### 팀별 인원 (Top 15)")
        if "팀" in filtered.columns and "재직여부" in filtered.columns:
            base = filtered[filtered["재직여부"] == True]
            g = base["팀"].dropna().astype(str).value_counts().head(15).reset_index()
            g.columns = ["팀", "인원"]
            fig = px.bar(g, x="팀", y="인원", text="인원")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="인원", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("팀/재직여부 데이터가 부족합니다.")

    with b:
        st.markdown("#### 직급별 분포 (피라미드 스타일)")
        if "직급/직책" in filtered.columns and "성별" in filtered.columns and "재직여부" in filtered.columns:
            base = filtered[filtered["재직여부"] == True].copy()
            pv = base.groupby(["직급/직책", "성별"]).size().reset_index(name="인원")

            order = ["주니어", "미들", "시니어", "리드", "매니저"]
            pv["직급/직책"] = pd.Categorical(pv["직급/직책"], categories=order, ordered=True)
            pv = pv.sort_values("직급/직책")

            pv["pyr"] = np.where(pv["성별"] == "남", -pv["인원"], pv["인원"])

            fig = go.Figure()
            for gend in ["남", "여"]:
                sub = pv[pv["성별"] == gend]
                fig.add_trace(go.Bar(
                    y=sub["직급/직책"],
                    x=sub["pyr"],
                    name=gend,
                    orientation="h",
                    customdata=sub["인원"],
                    hovertemplate="직급: %{y}<br>성별: " + gend + "<br>인원: %{customdata}<extra></extra>",
                ))
            fig.update_layout(
                barmode="relative",
                title="직급 피라미드(남/여)",
                xaxis_title="인원(남은 왼쪽, 여는 오른쪽)",
                yaxis_title="",
                hovermode="y unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("직급/성별/재직여부 데이터가 부족합니다.")

        st.markdown("#### 성별/연령대 분포 (적층)")
        if "연령대" in filtered.columns and "성별" in filtered.columns and "재직여부" in filtered.columns:
            base = filtered[filtered["재직여부"] == True].copy()
            tmp = base.dropna(subset=["연령대", "성별"])
            pv = tmp.groupby(["연령대", "성별"]).size().reset_index(name="인원")
            fig = px.bar(pv, x="연령대", y="인원", color="성별", barmode="stack")
            fig.update_layout(xaxis_title="연령대", yaxis_title="인원", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("연령대/성별/재직여부 데이터가 부족합니다.")

    st.divider()

    st.markdown("#### 근속연수 구간별 분포 (재직 기준)")
    if "근속기간(개월)" in filtered.columns and "재직여부" in filtered.columns:
        base = filtered[filtered["재직여부"] == True].copy()
        m = pd.to_numeric(base["근속기간(개월)"], errors="coerce")
        bins = [-1, 12, 36, 60, 10_000]
        labels = ["1년 미만", "1~3년", "3~5년", "5년 이상"]
        base["근속구간"] = pd.cut(m, bins=bins, labels=labels)
        dist = base["근속구간"].value_counts().reindex(labels).fillna(0).reset_index()
        dist.columns = ["근속구간", "인원"]
        dist["비율(%)"] = dist["인원"] / dist["인원"].sum() * 100

        fig = px.bar(dist, x="근속구간", y="인원", text=dist["비율(%)"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="인원")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("근속기간/재직여부 데이터가 부족합니다.")

    st.divider()
    st.markdown("#### 상세 인력 테이블")
    show_cols = [c for c in [
        "사번","성명","상위조직","팀","직무","직급/직책","구분","성별","나이","연령대","입사일","퇴사일","근속연수","재직여부"
    ] if c in filtered.columns]
    st.dataframe(filtered[show_cols] if show_cols else filtered, use_container_width=True, height=420)

    st.markdown("#### 다운로드 ⬇️")
    csv_buf = io.StringIO()
    (filtered[show_cols] if show_cols else filtered).to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button("CSV 다운로드(UTF-8 BOM)", data=csv_buf.getvalue().encode("utf-8-sig"), file_name="workforce_table.csv", mime="text/csv")

    if REPORTLAB_OK:
        pdf_lines = [
            f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"필터 결과 행 수: {len(filtered):,}",
            "표(상위 30행):"
        ]
        preview = (filtered[show_cols] if show_cols else filtered).head(30)
        for _, r in preview.iterrows():
            pdf_lines.append(" | ".join([str(r.get(c, "")) for c in preview.columns[:8]]))
        pdf_bytes = build_pdf_bytes("인력 구조 분석 - 테이블 요약", pdf_lines)
        st.download_button("PDF 다운로드(간단 보고용)", data=pdf_bytes, file_name="workforce_table_summary.pdf", mime="application/pdf")
    else:
        st.info("PDF 다운로드는 reportlab 설치 후 사용 가능해요.  pip install reportlab")

    st.markdown('> **"우리 조직은 어떻게 구성되어 있는가? 편중된 곳은 없는가?"**')

# ============================================================
# Dashboard 3
# ============================================================
with tab3:
    st.subheader("⚠️ 조직 리스크 조기경보")

    k1, k2, k3, k4 = st.columns(4)

    if not pd.isna(u1):
        k1.metric("1년 미만 비율", f"{u1:.1f}%", delta=("경고" if u1 >= 30 else "정상"))
    else:
        k1.info("1년 미만 비율: 데이터 없음")

    core_keywords = ["Engineer", "Designer", "Artist", "Tech", "Gameplay", "Backend", "Unity"]
    core_mask = filtered["직무"].astype(str).str.contains("|".join(core_keywords), case=False, na=False) if "직무" in filtered.columns else pd.Series([False]*len(filtered))
    core_df = filtered[core_mask].copy() if core_mask.any() else pd.DataFrame()
    core_to = monthly_turnover_rate(core_df) if not core_df.empty else pd.DataFrame()
    core_now, _, _ = last_two(core_to, "이직률(%)") if not core_to.empty else (np.nan, np.nan, np.nan)
    if not pd.isna(core_now):
        k2.metric("핵심 직무 이직률(월)", f"{core_now:.2f}%")
    else:
        k2.info("핵심 직무 이직률: 데이터 부족")

    dec_orgs = []
    if "상위조직" in filtered.columns and "입사일" in filtered.columns and len(month_range(filtered)) >= 4:
        ms = month_range(filtered)
        last4 = ms[-4:]
        for org in filtered["상위조직"].dropna().astype(str).unique().tolist():
            sub = filtered[filtered["상위조직"].astype(str) == org]
            hcs = [headcount_at_month_end(sub, m) for m in last4]
            if hcs[1] < hcs[0] and hcs[2] < hcs[1] and hcs[3] < hcs[2]:
                dec_orgs.append(org)
        k3.metric("3개월 연속 감소 조직 수", f"{len(dec_orgs)}개")
    else:
        k3.info("3개월 연속 감소: 데이터 부족")

    focus_pct = np.nan
    if "퇴사일" in filtered.columns and "근속기간(개월)" in filtered.columns and filtered["퇴사일"].notna().any():
        leavers = filtered[filtered["퇴사일"].notna()].copy()
        m = pd.to_numeric(leavers["근속기간(개월)"], errors="coerce")
        in_focus = (m >= 6) & (m < 12)
        focus_pct = in_focus.mean() * 100 if len(leavers) else np.nan
        if not pd.isna(focus_pct):
            k4.metric("6~12개월 이직 집중도", f"{focus_pct:.1f}%")
        else:
            k4.info("6~12개월 집중도: 데이터 부족")
    else:
        k4.info("6~12개월 집중도: 데이터 부족")

    st.divider()
    st.subheader("📊 트렌드 분석")
    c1, c2 = st.columns(2)

    # Stacked area
    with c1:
        st.markdown("#### 근속 구간별 인원 비중 변화 (스택 에어리어)")
        if "입사일" in filtered.columns and "근속기간(개월)" in filtered.columns:
            ms = month_range(filtered)
            if len(ms) >= 6:
                lastN = ms[-chart_months:]
                bands = [
                    ("0~3개월", 0, 3),
                    ("3~6개월", 3, 6),
                    ("6~12개월", 6, 12),
                    ("1~3년", 12, 36),
                    ("3년+", 36, 10_000),
                ]
                rows = []
                for mper in lastN:
                    month_end = (mper.to_timestamp("M") + pd.offsets.MonthEnd(0))
                    joined = filtered["입사일"].notna() & (filtered["입사일"] <= month_end)
                    not_left = (filtered["퇴사일"].isna() | (filtered["퇴사일"] > month_end)) if "퇴사일" in filtered.columns else True
                    snap = filtered[joined & not_left].copy()
                    if snap.empty:
                        continue
                    tenure_m = (month_end - snap["입사일"]).dt.days / DAYS_PER_MONTH  # FIX
                    total = len(snap)
                    for name, lo, hi_ in bands:
                        cnt = int(((tenure_m >= lo) & (tenure_m < hi_)).sum())
                        rows.append({"월": str(mper), "근속구간": name, "비중(%)": cnt / total * 100})
                area = pd.DataFrame(rows)
                if area.empty:
                    st.info("스냅샷 데이터가 부족합니다.")
                else:
                    fig = px.area(area, x="월", y="비중(%)", color="근속구간", groupnorm="percent",
                                  title=f"근속 구간 비중 변화 (최근 {chart_months}개월)")
                    fig.update_layout(hovermode="x unified", yaxis_title="비중(%)", legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("기간 데이터가 부족합니다.")
        else:
            st.info("입사일/근속 데이터가 부족합니다.")

    # Heatmap
    with c2:
        st.markdown("#### 상위조직별 인원 증감 히트맵 (3/6/12개월)")
        if "상위조직" in filtered.columns and "입사일" in filtered.columns and len(month_range(filtered)) >= 13:
            ms = month_range(filtered)
            last = ms[-1]
            horizons = [3, 6, 12]
            rows = []
            for org in filtered["상위조직"].dropna().astype(str).unique().tolist():
                sub = filtered[filtered["상위조직"].astype(str) == org]
                hc_last = headcount_at_month_end(sub, last)
                for h in horizons:
                    prev_m = ms[-(h + 1)]
                    hc_prev = headcount_at_month_end(sub, prev_m)
                    rate = (hc_last - hc_prev) / hc_prev * 100 if hc_prev > 0 else np.nan
                    rows.append({"상위조직": org, "기간": f"{h}개월", "증감률(%)": rate})
            heat = pd.DataFrame(rows).dropna()
            if heat.empty:
                st.info("히트맵 계산 데이터가 부족합니다.")
            else:
                piv = heat.pivot_table(index="상위조직", columns="기간", values="증감률(%)", aggfunc="mean")
                fig = px.imshow(
                    piv,
                    aspect="auto",
                    text_auto=".1f",
                    color_continuous_scale=["#b71c1c", "#f5f5f5", "#1b5e20"],
                    title="상위조직별 인원 증감률(%)"
                )
                fig.update_layout(coloraxis_colorbar_title="증감률(%)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("상위조직/기간 데이터가 부족합니다.")

    st.divider()
    st.subheader("🔔 리스크 알림 리스트 (드릴다운)")
    alerts = []
    if "상위조직" in filtered.columns:
        for org in filtered["상위조직"].dropna().astype(str).unique().tolist():
            sub = filtered[filtered["상위조직"].astype(str) == org]
            act = sub[sub["재직여부"] == True] if "재직여부" in sub.columns else sub
            u1_org = (act["1년미만"].mean() * 100) if ("1년미만" in act.columns and len(act) > 0) else np.nan

            lv_cnt = 0
            if "퇴사일" in sub.columns and sub["퇴사일"].notna().any():
                cut30 = TODAY - pd.Timedelta(days=30)
                lv_cnt = int(((sub["퇴사일"].notna()) & (sub["퇴사일"] >= cut30)).sum())

            level = None
            reasons = []
            if not pd.isna(u1_org):
                reasons.append(f"1년 미만 {u1_org:.1f}%")
            if lv_cnt > 0:
                reasons.append(f"30일 퇴사 {lv_cnt}명")

            if (not pd.isna(u1_org) and u1_org >= 35) or (lv_cnt >= 4):
                level = "높음"
            elif (not pd.isna(u1_org) and u1_org >= 30) or (lv_cnt == 3):
                level = "중간"
            elif (not pd.isna(u1_org) and u1_org >= 25):
                level = "낮음"

            if level:
                alerts.append({"상위조직": org, "경고수준": level, "근거": ", ".join(reasons)})

    alert_df = pd.DataFrame(alerts)
    if alert_df.empty:
        st.info("현재 표시할 리스크 알림이 없습니다 🙂")
    else:
        level_order = {"높음": 0, "중간": 1, "낮음": 2}
        alert_df["order"] = alert_df["경고수준"].map(level_order)
        alert_df = alert_df.sort_values(["order", "상위조직"]).drop(columns=["order"])
        st.dataframe(alert_df, use_container_width=True, height=240)

        pick_org = st.selectbox("상세로 볼 상위조직", alert_df["상위조직"].unique().tolist())
        drill = filtered[filtered["상위조직"].astype(str) == pick_org].copy()

        d1, d2 = st.columns(2)
        with d1:
            d_to = monthly_turnover_rate(drill)
            d_to = tail_months(d_to, chart_months)
            if not d_to.empty:
                fig = px.line(d_to, x="월", y="이직률(%)", markers=True, title=f"{pick_org} 이직률 추이")
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("조직 이직률 데이터가 부족합니다.")
        with d2:
            if "근속기간(개월)" in drill.columns and "재직여부" in drill.columns:
                act = drill[drill["재직여부"] == True].copy()
                m = pd.to_numeric(act["근속기간(개월)"], errors="coerce")
                bins = [-1, 12, 36, 60, 10_000]
                labels = ["1년 미만", "1~3년", "3~5년", "5년 이상"]
                act["근속구간"] = pd.cut(m, bins=bins, labels=labels)
                dist = act["근속구간"].value_counts().reindex(labels).fillna(0).reset_index()
                dist.columns = ["근속구간", "인원"]
                fig = px.bar(dist, x="근속구간", y="인원", title=f"{pick_org} 근속구간 분포(재직)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("근속 데이터가 부족합니다.")

    st.markdown('> **"문제가 터지기 전에 어디를 살펴봐야 하는가?"**')

# ============================================================
# Dashboard 4
# ============================================================
with tab4:
    st.subheader("🌱 신규 입사자 품질 분석")

    st.markdown("#### 1) 입사 현황")
    a, b = st.columns(2)

    with a:
        if not hi.empty:
            hi_t = hi.tail(chart_months)
            fig = px.line(hi_t, x="월", y="입사자수", markers=True, title=f"월별 신규 입사자 수 (최근 {chart_months}개월)")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("입사일 데이터가 부족합니다.")

    with b:
        if "직무" in filtered.columns and "입사일" in filtered.columns:
            cut = TODAY - pd.DateOffset(months=6)
            recent = filtered[(filtered["입사일"].notna()) & (filtered["입사일"] >= cut)].copy()
            if recent.empty:
                st.info("최근 6개월 입사 데이터가 없습니다.")
            else:
                g = recent["직무"].dropna().astype(str).value_counts().head(15).reset_index()
                g.columns = ["직무", "입사자수(6개월)"]
                fig = px.bar(g, x="직무", y="입사자수(6개월)", text="입사자수(6개월)", title="직무별 입사자 분포")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("직무/입사일 데이터가 부족합니다.")

    st.divider()
    st.markdown("#### 2) 정착률 분석 (입사월별 3/6/12개월)")

    ret = cohort_retention(filtered, months_points=(3, 6, 12))
    if ret.empty:
        st.info("정착률 계산을 위한 데이터가 부족합니다. (입사일/재직여부 필요)")
    else:
        piv = ret.pivot_table(index="입사월", columns="개월", values="정착률(%)", aggfunc="mean").reset_index()
        piv["입사월_p"] = pd.PeriodIndex(piv["입사월"], freq="M")
        piv = piv.sort_values("입사월_p").drop(columns=["입사월_p"])

        long = piv.melt(id_vars=["입사월"], var_name="개월", value_name="정착률(%)")
        fig = px.line(long, x="입사월", y="정착률(%)", color="개월", markers=True, title="입사월별 정착률 비교")
        fig.update_layout(hovermode="x unified", legend_title_text="개월")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 조직별 정착 곡선 비교 (선택 조회)")
        if "상위조직" in filtered.columns and "입사일" in filtered.columns and "재직여부" in filtered.columns:
            orgs = sorted(filtered["상위조직"].dropna().astype(str).unique().tolist())
            sel_org = st.selectbox("정착 곡선을 볼 상위조직 선택", orgs)
            sub = filtered[filtered["상위조직"].astype(str) == sel_org].copy()

            points = list(range(0, 37, 3))
            rows = []
            for p in points:
                eligible = sub[(TODAY - sub["입사일"]).dt.days >= int(p * DAYS_PER_MONTH)]
                rate = eligible["재직여부"].mean() * 100 if not eligible.empty else np.nan
                rows.append({"개월": p, "잔존율(%)": rate})
            curve = pd.DataFrame(rows).dropna()
            if curve.empty:
                st.info("정착 곡선 계산 데이터가 부족합니다.")
            else:
                fig = px.line(curve, x="개월", y="잔존율(%)", markers=True, title=f"{sel_org} 정착 곡선(0~36개월)")
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("상위조직/입사일/재직여부 데이터가 부족합니다.")

    st.markdown('> **"우리가 뽑은 사람들, 잘 정착하고 있는가? 어느 조직이 신입을 잘 키우는가?"**')

# ============================================================
# Dashboard 5
# ============================================================
with tab5:
    st.subheader("🚪 이직 패턴 심층 분석")

    st.markdown("#### 1) 기본 트렌드")
    a, b = st.columns(2)
    with a:
        if not to_tail.empty:
            fig = px.line(to_tail, x="월", y="이직률(%)", markers=True, title=f"월별 이직률 추이 (최근 {chart_months}개월)")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("이직률 데이터가 부족합니다.")

    with b:
        st.markdown("**상위조직별 이직(최근 12개월, 전사 평균 대비)**")
        if "상위조직" in filtered.columns and "퇴사일" in filtered.columns and "입사일" in filtered.columns:
            win_start = TODAY - pd.DateOffset(months=12)
            leavers = filtered[(filtered["퇴사일"].notna()) & (filtered["퇴사일"] >= win_start)].copy()
            hc_all = monthly_headcount(filtered).tail(12)
            avg_hc = hc_all["월말인원"].mean() if not hc_all.empty else np.nan
            overall_rate = (len(leavers) / avg_hc * 100) if (not pd.isna(avg_hc) and avg_hc > 0) else np.nan

            by_org = leavers["상위조직"].dropna().astype(str).value_counts().reset_index()
            by_org.columns = ["상위조직", "퇴사자수(12개월)"]

            act = filtered[filtered["재직여부"] == True] if "재직여부" in filtered.columns else filtered
            share = act["상위조직"].dropna().astype(str).value_counts()
            by_org["추정평균인원"] = by_org["상위조직"].map(lambda x: max(1, int(round(share.get(x, 1) * 0.9))))
            by_org["이직률(추정,%)"] = by_org["퇴사자수(12개월)"] / by_org["추정평균인원"] * 100
            by_org = by_org.sort_values("이직률(추정,%)", ascending=False)

            fig = px.bar(by_org, x="상위조직", y="이직률(추정,%)", title="상위조직별 이직률(추정)")
            if not pd.isna(overall_rate):
                fig.add_hline(y=overall_rate, line_dash="dash", annotation_text="전사 평균", annotation_position="top left")
            fig.update_layout(yaxis_title="이직률(%)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("상위조직/입사일/퇴사일 데이터가 부족합니다.")

    st.divider()
    st.markdown("#### 2) 이직 집중 구간 분석")
    if "퇴사일" in filtered.columns and "근속기간(개월)" in filtered.columns and filtered["퇴사일"].notna().any():
        leavers = filtered[filtered["퇴사일"].notna()].copy()
        m = pd.to_numeric(leavers["근속기간(개월)"], errors="coerce")
        bins = [-1, 12, 36, 10_000]
        labels = ["1년 미만", "1~3년", "3년 이상"]
        leavers["구간"] = pd.cut(m, bins=bins, labels=labels)
        dist = leavers["구간"].value_counts().reindex(labels).fillna(0).reset_index()
        dist.columns = ["구간", "퇴사자수"]
        dist["비율(%)"] = dist["퇴사자수"] / dist["퇴사자수"].sum() * 100

        fig = px.bar(dist, x="구간", y="퇴사자수", text=dist["비율(%)"].map(lambda x: f"{x:.1f}%"),
                     title="재직기간 구간별 이직 분포")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="퇴사자수")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("퇴사일/근속 데이터가 부족합니다.")

    st.divider()
    st.markdown("#### 3) 조직별 인력 유지 곡선 (전사 평균 + Top 3)")
    if "상위조직" in filtered.columns and "입사일" in filtered.columns and "재직여부" in filtered.columns:
        win_start = TODAY - pd.DateOffset(months=12)
        act = filtered[filtered["재직여부"] == True]
        org_list = act["상위조직"].dropna().astype(str).unique().tolist()

        scores = []
        for org in org_list:
            sub = filtered[filtered["상위조직"].astype(str) == org]
            le = sub[(sub["퇴사일"].notna()) & (sub["퇴사일"] >= win_start)]
            denom = max(1, int(act[act["상위조직"].astype(str) == org].shape[0]))
            score = le.shape[0] / denom
            scores.append((org, score))
        top3 = [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)[:3]]

        points = list(range(0, 37, 3))
        rows = []

        for p in points:
            eligible = filtered[(TODAY - filtered["입사일"]).dt.days >= int(p * DAYS_PER_MONTH)]
            rate = eligible["재직여부"].mean() * 100 if not eligible.empty else np.nan
            rows.append({"개월": p, "잔존율(%)": rate, "그룹": "전사 평균"})

        for org in top3:
            sub = filtered[filtered["상위조직"].astype(str) == org]
            for p in points:
                eligible = sub[(TODAY - sub["입사일"]).dt.days >= int(p * DAYS_PER_MONTH)]
                rate = eligible["재직여부"].mean() * 100 if not eligible.empty else np.nan
                rows.append({"개월": p, "잔존율(%)": rate, "그룹": org})

        curve = pd.DataFrame(rows).dropna()
        if curve.empty:
            st.info("유지 곡선 계산 데이터가 부족합니다.")
        else:
            fig = px.line(curve, x="개월", y="잔존율(%)", color="그룹", markers=True, title="재직 지속 곡선(0~36개월)")
            fig.update_layout(hovermode="x unified", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("상위조직/입사일/재직여부 데이터가 부족합니다.")

    st.divider()
    st.markdown("#### 4) 이직 사유 분석")
    a, b = st.columns(2)
    with a:
        if "자발/비자발" in filtered.columns and "퇴사일" in filtered.columns and filtered["퇴사일"].notna().any():
            x = filtered[filtered["퇴사일"].notna()].copy()
            pv = x["자발/비자발"].dropna().astype(str).value_counts().reset_index()
            pv.columns = ["구분", "퇴사자수"]
            fig = px.pie(pv, names="구분", values="퇴사자수", title="자발/비자발 비율")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("자발/비자발 데이터가 없어 공란 처리")

    with b:
        if "퇴사사유" in filtered.columns and "퇴사일" in filtered.columns and filtered["퇴사일"].notna().any():
            x = filtered[filtered["퇴사일"].notna()].copy()
            pv = x["퇴사사유"].dropna().astype(str).value_counts().reset_index()
            pv.columns = ["사유", "퇴사자수"]
            fig = px.bar(pv, x="사유", y="퇴사자수", text="퇴사자수", title="퇴사 사유 분포")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("퇴사 사유 데이터가 없어서 공란 처리")

    st.divider()
    st.markdown("#### 5) 고위험군 식별")
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("**최근 3개월 퇴사자 중 6~12개월 구간 비중(온보딩 리스크 신호)**")
        if "퇴사일" in filtered.columns and "근속기간(개월)" in filtered.columns and filtered["퇴사일"].notna().any():
            cut90 = TODAY - pd.Timedelta(days=90)
            lv90 = filtered[(filtered["퇴사일"].notna()) & (filtered["퇴사일"] >= cut90)].copy()
            if lv90.empty:
                st.info("최근 3개월 퇴사 데이터가 없습니다.")
            else:
                m = pd.to_numeric(lv90["근속기간(개월)"], errors="coerce")
                focus = ((m >= 6) & (m < 12)).mean() * 100 if m.notna().any() else np.nan
                st.metric("비중", f"{focus:.1f}%" if not pd.isna(focus) else "-")
        else:
            st.info("퇴사일/근속 데이터가 부족합니다.")

    with r2:
        st.markdown("**특정 연차 이탈 패턴**")
        if "퇴사일" in filtered.columns and "근속연수" in filtered.columns and filtered["퇴사일"].notna().any():
            x = filtered[filtered["퇴사일"].notna()].copy()
            y = pd.to_numeric(x["근속연수"], errors="coerce")
            bins = [-1, 1, 2, 3, 5, 10_000]
            labels = ["0~1년", "1~2년", "2~3년", "3~5년", "5년+"]
            x["연차구간"] = pd.cut(y, bins=bins, labels=labels)
            pv = x["연차구간"].value_counts().reindex(labels).fillna(0).reset_index()
            pv.columns = ["연차구간", "퇴사자수"]
            fig = px.bar(pv, x="연차구간", y="퇴사자수", title="연차 구간별 퇴사 분포")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("연차(근속연수)/퇴사 데이터가 부족합니다.")

    st.markdown('> **"왜 사람들이 떠나는가? 언제 떠나는가? 어떻게 막을 수 있는가?"**')

# -----------------------------
# Footer
# -----------------------------
st.caption("실행:  pip install streamlit pandas plotly openpyxl  |  streamlit run app.py")
# think about is step-by-step
