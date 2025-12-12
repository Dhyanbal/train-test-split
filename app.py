"""
Streamlit app: Train/Test Split & Data Leakage Demo

Run: streamlit run streamlit_train_test_leakage_demo.py

Features:
- Build a synthetic dataset or upload your own CSV
- Interactive controls for train/test split, random seed
- Choose scenario: Baseline, Target leakage, Preprocessing leakage,
  Feature-selection leakage, Correct pipeline (no leakage),
  Time-series: random vs chronological split
- Train a LogisticRegression, display Accuracy + ROC AUC, show ROC curve
- Summary table across scenarios and option to download results

Requirements: streamlit, scikit-learn, pandas, numpy, matplotlib

"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Train/Test Split & Leakage Demo", layout="wide")

# --- Helpers ---
@st.cache_data
def make_demo_data(n_samples=1000, n_features=10, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=max(2, n_features//2), n_redundant=2,
                               random_state=random_state)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df['target'] = y
    df['time'] = pd.date_range(start="2020-01-01", periods=len(df), freq='H')
    return df

@st.cache_data
def train_eval(X_train, X_test, y_train, y_test, use_scaler=True):
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500)
    if use_scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    return dict(model=clf, acc=acc, auc=auc, fpr=fpr, tpr=tpr, probs=probs)

# --- UI ---
st.title("Train/Test Split & Data-Leakage Demo")
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Controls")
    data_source = st.radio("Data source:", ("Synthetic demo data", "Upload CSV"))
    if data_source == "Synthetic demo data":
        n_samples = st.slider("# samples", 200, 5000, 1000, step=100)
        n_features = st.slider("# features", 4, 50, 10)
        seed = st.number_input("Random seed", value=42, step=1)
        df = make_demo_data(n_samples=n_samples, n_features=n_features, random_state=seed)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            # Try to detect target column
            if 'target' not in df.columns:
                st.warning("No 'target' column found. Please ensure your CSV has a binary 'target' column.")
        else:
            st.info("Upload a CSV or switch to Synthetic demo data.")
            st.stop()

    st.markdown("---")
    st.subheader("Split & model options")
    test_size = st.slider("Test set proportion", 0.05, 0.5, 0.25, step=0.05)
    random_state = st.number_input("Split random state", value=42, step=1)
    scenario = st.selectbox("Leakage scenario", [
        "Baseline (proper split)",
        "Target leakage (leak feature from target)",
        "Preprocessing leakage (fit scaler on whole data)",
        "Feature-selection leakage (SelectKBest before split)",
        "Correct pipeline (selection inside pipeline)",
        "Time: random split (possible leakage)",
        "Time: chronological split (no future leakage)"
    ])
    k_best = st.slider("SelectKBest: k", 1, max(1, min(20, n_features if data_source=='Synthetic demo data' else 20)), 5) if 'SelectKBest' in st.session_state or True else 5

    run_button = st.button("Run scenario")
    st.markdown("---")
    st.caption("This demo trains a LogisticRegression and reports Accuracy + ROC AUC. Use pipelines to avoid preprocessing/selection leakage.")

with col2:
    st.header("Dataset sample")
    st.dataframe(df.head(10))

# --- Core logic ---
if run_button:
    # Prepare X and y
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    if len(feature_cols) == 0:
        st.error("No feature columns starting with 'feat_' found in the data. For uploaded CSV, ensure features are named feat_0, feat_1, ... or edit the code.")
        st.stop()
    X = df[feature_cols].values
    y = df['target'].values

    results = {}

    if scenario == "Baseline (proper split)":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
        res = train_eval(X_train, X_test, y_train, y_test, use_scaler=True)
        results[scenario] = res

    elif scenario == "Target leakage (leak feature from target)":
        df2 = df.copy()
        np.random.seed(int(random_state))
        df2['leak_feat'] = df2['target'] + np.random.normal(0, 0.01, size=len(df2))
        X2 = df2[feature_cols + ['leak_feat']].values
        X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=test_size, random_state=int(random_state), stratify=y)
        res = train_eval(X_train, X_test, y_train, y_test, use_scaler=True)
        results[scenario] = res

    elif scenario == "Preprocessing leakage (fit scaler on whole data)":
        # Fit scaler on whole X (leakage), then split
        scaler_whole = StandardScaler()
        X_scaled_whole = scaler_whole.fit_transform(X)
        X_train_pw, X_test_pw, y_train_pw, y_test_pw = train_test_split(X_scaled_whole, y, test_size=test_size, random_state=int(random_state), stratify=y)
        # Train without further scaling
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train_pw, y_train_pw)
        preds = clf.predict(X_test_pw)
        probs = clf.predict_proba(X_test_pw)[:, 1]
        res = dict(model=clf, acc=accuracy_score(y_test_pw, preds), auc=roc_auc_score(y_test_pw, probs), fpr=roc_curve(y_test_pw, probs)[0], tpr=roc_curve(y_test_pw, probs)[1])
        results[scenario] = res

    elif scenario == "Feature-selection leakage (SelectKBest before split)":
        selector = SelectKBest(score_func=f_classif, k=min(k_best, X.shape[1]))
        X_selected_whole = selector.fit_transform(X, y)  # leakage
        X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_selected_whole, y, test_size=test_size, random_state=int(random_state), stratify=y)
        res = train_eval(X_train_fs, X_test_fs, y_train_fs, y_test_fs, use_scaler=True)
        results[scenario] = res

    elif scenario == "Correct pipeline (selection inside pipeline)":
        pipe = Pipeline([('selector', SelectKBest(score_func=f_classif, k=min(k_best, X.shape[1]))),
                         ('scaler', StandardScaler()),
                         ('clf', LogisticRegression(max_iter=500))])
        X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
        pipe.fit(X_train_corr, y_train_corr)
        probs = pipe.predict_proba(X_test_corr)[:, 1]
        preds = pipe.predict(X_test_corr)
        res = dict(model=pipe, acc=accuracy_score(y_test_corr, preds), auc=roc_auc_score(y_test_corr, probs), fpr=roc_curve(y_test_corr, probs)[0], tpr=roc_curve(y_test_corr, probs)[1])
        results[scenario] = res

    elif scenario == "Time: random split (possible leakage)":
        # Create time-dependent target for demo
        df_time = df.copy()
        time_numeric = (df_time['time'] - df_time['time'].min()).dt.total_seconds()
        np.random.seed(int(random_state))
        trend_score = (time_numeric - time_numeric.mean()) / time_numeric.std() + np.random.normal(0, 1, size=len(df_time))
        df_time['time_target'] = (trend_score > np.median(trend_score)).astype(int)
        X_time = df_time[feature_cols].values
        y_time = df_time['time_target'].values
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_time, y_time, test_size=test_size, random_state=int(random_state), shuffle=True, stratify=y_time)
        res = train_eval(X_tr_r, X_te_r, y_tr_r, y_te_r, use_scaler=True)
        results[scenario] = res

    elif scenario == "Time: chronological split (no future leakage)":
        df_time = df.copy()
        time_numeric = (df_time['time'] - df_time['time'].min()).dt.total_seconds()
        np.random.seed(int(random_state))
        trend_score = (time_numeric - time_numeric.mean()) / time_numeric.std() + np.random.normal(0, 1, size=len(df_time))
        df_time['time_target'] = (trend_score > np.median(trend_score)).astype(int)
        df_time_sorted = df_time.sort_values('time').reset_index(drop=True)
        cut = int(len(df_time_sorted) * (1 - test_size))
        X_tr_t = df_time_sorted[feature_cols].values[:cut]
        y_tr_t = df_time_sorted['time_target'].values[:cut]
        X_te_t = df_time_sorted[feature_cols].values[cut:]
        y_te_t = df_time_sorted['time_target'].values[cut:]
        res = train_eval(X_tr_t, X_te_t, y_tr_t, y_te_t, use_scaler=True)
        results[scenario] = res

    # --- Display results ---
    st.subheader("Results")
    for k, v in results.items():
        st.markdown(f"**{k}**")
        st.write(f"Accuracy: {v['acc']:.4f} — ROC AUC: {v['auc']:.4f}")
        # ROC plot
        fig, ax = plt.subplots()
        ax.plot(v['fpr'], v['tpr'])
        ax.set_title(f"ROC — {k}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True)
        st.pyplot(fig)

    # Summary download
    summary_rows = [[k, results[k]['acc'], results[k]['auc']] for k in results]
    summary_df = pd.DataFrame(summary_rows, columns=['Scenario', 'Accuracy', 'ROC_AUC'])
    st.subheader("Summary")
    st.dataframe(summary_df)
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download summary CSV", data=csv, file_name='leakage_summary.csv', mime='text/csv')

    # Optionally show model coefficients for interpretability (if logistic)
    st.markdown("---")
    if isinstance(list(results.values())[0]['model'], LogisticRegression):
        st.subheader("Model coefficients")
        m = list(results.values())[0]['model']
        coef = m.coef_.ravel()
        coef_df = pd.DataFrame({'feature': feature_cols[:len(coef)], 'coef': coef})
        st.dataframe(coef_df)

    st.success("Done. You can change controls and re-run to explore different leakage scenarios.")

# Footer / deployment tips
st.markdown("---")
st.markdown("**Deployment tips:**\n\n- Locally: `pip install streamlit scikit-learn pandas numpy matplotlib` then `streamlit run streamlit_train_test_leakage_demo.py`.\n- Streamlit Cloud: push to GitHub and deploy as a new Streamlit app.\n- Docker: create a small Dockerfile with python base image, install requirements, and run `streamlit run`.")
