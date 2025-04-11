import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, 
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from xgboost import XGBClassifier

st.set_page_config(layout="wide")
st.title("🤖 Underwater Bot ML Classifier & Regressor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your underwater dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df['range_rate'] = df['range'].diff() / df['time'].diff()
    df['bearing_rate'] = df['bearing'].diff() / df['time'].diff()
    df['elevation_rate'] = df['elevation'].diff() / df['time'].diff()

    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    le = LabelEncoder()
    df['depth_zone'] = le.fit_transform(df['depth_zone'])

    # Feature/Target
    features = ['bearing', 'elevation', 'range', 'range_rate', 'bearing_rate', 
                'elevation_rate', 'snr', 'measurement_quality', 'current_x', 
                'current_y', 'current_z', 'depth_zone', 'multipath']
    X = df[features]
    y = df['maneuver_mode']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ### Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    prob_rf = rf.predict_proba(X_test_scaled)

    ### Train XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)

    # Classification Reports
    st.subheader("📊 Classification Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Random Forest*")
        st.text(classification_report(y_test, y_pred_rf))
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_rf):.2f}")
    with col2:
        st.markdown("*XGBoost*")
        st.text(classification_report(y_test, y_pred_xgb))
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_xgb):.2f}")

    # Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)

    st.subheader("🔍 Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Random Forest Confusion Matrix*")
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Blues", ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.markdown("*XGBoost Confusion Matrix*")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap="Greens", ax=ax2)
        st.pyplot(fig2)

    # ROC Curve
    st.subheader("📈 ROC Curve (Random Forest)")
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(np.unique(y))
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], prob_rf[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax3.plot(fpr[i], tpr[i], label=f"Class {i} AUC = {roc_auc[i]:.2f}")
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve - Random Forest")
    ax3.legend(loc="lower right")
    st.pyplot(fig3)

    # Feature Importance
    st.subheader("🧠 Feature Importance")
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)
    fig4, ax4 = plt.subplots()
    ax4.barh(np.array(features)[sorted_idx], importance[sorted_idx])
    ax4.set_title("Feature Importance - Random Forest")
    st.pyplot(fig4)

    # Regressor
    st.subheader("🎯 Position Regressor (Next Target)")
    y_pos = df[['target_x', 'target_y', 'target_z']].shift(-1).iloc[:-1]
    X_pos = df[features].iloc[:-1]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)
    X_train_r_scaled = scaler.fit_transform(X_train_r)
    X_test_r_scaled = scaler.transform(X_test_r)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_r_scaled, y_train_r)
    y_pred_r = reg.predict(X_test_r_scaled)
    mse = np.mean((y_test_r.values - y_pred_r)**2, axis=0)

    st.markdown(f"*Mean Squared Error per Target:*\n- X: {mse[0]:.2f}\n- Y: {mse[1]:.2f}\n- Z: {mse[2]:.2f}")