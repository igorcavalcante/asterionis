import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide", page_title="Stellar Systems - Planet Classification Explorer")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

def get_model(name, params):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "auto"),
            class_weight=params.get("class_weight", None),
            random_state=params.get("random_state", 42),
        )
    elif name == "Logistic Regression":
        solver = "lbfgs" if params.get("penalty", "l2") in ["l2", "none"] else "liblinear"
        return LogisticRegression(
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            solver=solver,
            max_iter=1000,
            class_weight=params.get("class_weight", None),
            random_state=params.get("random_state", 42),
        )
    elif name == "SVM (RBF/Linear)":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            probability=True,
            class_weight=params.get("class_weight", None),
            random_state=params.get("random_state", 42),
        )
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=params.get("random_state", 42),
        )
    else:
        raise ValueError("Unknown model")

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=14)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    return fig, roc_auc

st.title("Stellar Systems — Planet vs. No-Planet Classification Explorer")
st.markdown("Upload or use the provided dataset. Target column: **tfopwg_disp** (planet or not).")

DATA_PATH = "df_toi_planet_or_not.csv"
df = load_data(DATA_PATH)

if "tfopwg_disp" not in df.columns:
    st.error("Target column 'tfopwg_disp' not found in dataset. Please upload a dataset with that column.")
else:
    st.sidebar.header("Data sampling & general settings")
    sample_mode = st.sidebar.radio("Subsample mode", ("Fraction", "Number", "Use all (no subsample)"))
    if sample_mode == "Fraction":
        frac = st.sidebar.slider("Fraction of rows to sample", 0.01, 1.0, 0.2, 0.01)
        df_sampled = df.sample(frac=frac, random_state=42)
    elif sample_mode == "Number":
        n = st.sidebar.number_input("Number of rows to sample", min_value=100, max_value=int(len(df)), value=min(2000, len(df)), step=100)
        df_sampled = df.sample(n=min(n, len(df)), random_state=42)
    else:
        df_sampled = df.copy()

    st.sidebar.write(f"Dataset rows after sampling: {len(df_sampled)}")

    st.header("Preview data")
    if st.checkbox("Show raw data (first 200 rows)", value=False):
        st.dataframe(df_sampled.head(200))

    all_columns = df_sampled.columns.tolist()
    default_features = [c for c in all_columns if c != "tfopwg_disp"]
    selected_features = st.sidebar.multiselect("Select features to use", options=all_columns, default=default_features)

    if not selected_features:
        st.error("Select at least one feature.")
    else:
        X = df_sampled[selected_features].copy()
        y = df_sampled["tfopwg_disp"].copy()

        # --- FIX: Encode target automatically if not numeric ---
        y_original = y.copy()
        if y.dtype == "O" or not np.issubdtype(y.dtype, np.number):
            y_encoded, uniques = pd.factorize(y)
            y = pd.Series(y_encoded, index=y.index)
            st.info(f"Target encoded: {dict(zip(uniques, range(len(uniques))))}")
        else:
            uniques = np.unique(y)

        if len(np.unique(y)) == 2:
            pos_label = 1
        else:
            pos_label = np.unique(y)[-1]

        # --- Preprocessing ---
        st.sidebar.header("Preprocessing")
        do_scaling = st.sidebar.checkbox("Scale numeric features (StandardScaler)", value=True)
        do_onehot = st.sidebar.checkbox("One-hot encode categorical features", value=True)

        if do_onehot:
            X = pd.get_dummies(X, dummy_na=False)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            imp = SimpleImputer(strategy="median")
            X[numeric_cols] = imp.fit_transform(X[numeric_cols])
            if do_scaling:
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # --- Split ---
        st.sidebar.header("Train/Test split")
        test_size = st.sidebar.slider("Test set proportion", 0.05, 0.5, 0.2, 0.01)
        random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y if len(np.unique(y)) > 1 else None
        )

        # --- Model selection ---
        st.sidebar.header("Model selection")
        model_name = st.sidebar.selectbox("Choose model", ["Random Forest", "Logistic Regression", "SVM (RBF/Linear)", "Gradient Boosting"])

        params = {}
        if model_name == "Random Forest":
            params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 1000, 200, step=10)
            max_depth = st.sidebar.slider("max_depth (None=0)", 0, 50, 0)
            params["max_depth"] = None if max_depth == 0 else int(max_depth)
            params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2)
            max_feat = st.sidebar.selectbox("max_features", ["auto", "sqrt", "log2", "None"], index=1)
            params["max_features"] = None if max_feat == "None" else max_feat
            cw = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
            params["class_weight"] = None if cw == "None" else "balanced"
            params["random_state"] = int(random_state)
        elif model_name == "Logistic Regression":
            params["C"] = st.sidebar.number_input("C (inverse regularization strength)", min_value=0.0001, value=1.0, format="%.4f")
            params["penalty"] = st.sidebar.selectbox("penalty", ["l2", "l1", "none"])
            cw = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
            params["class_weight"] = None if cw == "None" else "balanced"
            params["random_state"] = int(random_state)
        elif model_name == "SVM (RBF/Linear)":
            params["C"] = st.sidebar.number_input("C", min_value=0.01, value=1.0, format="%.2f")
            params["kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear"])
            params["gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"])
            cw = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
            params["class_weight"] = None if cw == "None" else "balanced"
            params["random_state"] = int(random_state)
        elif model_name == "Gradient Boosting":
            params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 1000, 100, step=10)
            params["learning_rate"] = st.sidebar.number_input("learning_rate", min_value=0.001, value=0.1, format="%.3f")
            params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 3)
            params["random_state"] = int(random_state)

        if st.sidebar.button("Train model"):
            model = get_model(model_name, params)
            with st.spinner("Training the model..."):
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                try:
                    y_proba = model.decision_function(X_test)
                    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)
                except Exception:
                    y_proba = np.zeros_like(y_pred)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            st.subheader("Evaluation metrics on test set")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Precision", f"{prec:.3f}")
            col3.metric("Recall", f"{rec:.3f}")
            col4.metric("F1-score", f"{f1:.3f}")

            cm = confusion_matrix(y_test, y_pred)
            fig_cm = plot_confusion_matrix(cm, labels=list(map(str, np.unique(y_original))))
            st.pyplot(fig_cm)

            fig_roc, roc_auc = plot_roc(y_test == pos_label, y_proba)
            st.pyplot(fig_roc)
            st.write(f"ROC AUC: **{roc_auc:.3f}**")

            st.subheader("Cross-validation (5-fold) on training set")
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
                st.write("F1 scores:", np.round(cv_scores, 3))
                st.write("Mean F1:", np.round(cv_scores.mean(), 3))
            except Exception as e:
                st.write("Cross-validation failed:", e)

            st.subheader("Model inspection")
            if model_name in ["Random Forest", "Gradient Boosting"]:
                try:
                    importances = model.feature_importances_
                    fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                    st.dataframe(fi.head(30).to_frame("importance"))
                    fig, ax = plt.subplots(figsize=(6, 6))
                    fi.head(20).plot.bar(ax=ax)
                    ax.set_ylabel("Importance")
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not compute feature importances:", e)
            elif model_name == "Logistic Regression":
                try:
                    coefs = pd.Series(model.coef_.ravel(), index=X.columns).sort_values(key=abs, ascending=False)
                    st.dataframe(coefs.head(30).to_frame("coef"))
                    fig, ax = plt.subplots(figsize=(6, 6))
                    coefs.head(20).plot.bar(ax=ax)
                    ax.set_ylabel("Coefficient")
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not show coefficients:", e)

            results_df = X_test.copy()
            results_df["y_true"] = y_test.values
            results_df["y_pred"] = y_pred
            results_df["y_proba"] = y_proba
            buf = BytesIO()
            results_df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download test-set predictions (CSV)", data=buf, file_name="predictions.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("App generated by ChatGPT — Streamlit. Adjust parameters and subsampling to explore model behavior.")
