import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap

logistic_model = load("models/logistic_model.pkl")
rf_model = load("models/rf_model.pkl")
xgb_model = load("models/xgb_model.pkl")

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])
preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

with tab1:

    st.title("Customer Churn Prediction App")

    st.write("Enter customer details to predict churn probability")


    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )


    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])

    tenure = st.slider("Tenure (months)",0,72)

    phoneservice = st.selectbox("Phone Service", ["Yes","No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    onlinesecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    streamingtv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    )

    monthly = st.number_input("Monthly Charges",0.0,200.0)
    total = st.number_input("Total Charges",0.0,10000.0)


    data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phoneservice],
    "MultipleLines":[multiplelines],
    "InternetService":[internet],
    "OnlineSecurity":[onlinesecurity],
    "OnlineBackup":[onlinebackup],
    "DeviceProtection":[deviceprotection],
    "TechSupport":[techsupport],
    "StreamingTV":[streamingtv],
    "StreamingMovies":[streamingmovies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
    })

    X_transformed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )
    explainer = shap.TreeExplainer(rf_classifier)


    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    if st.button("Predict Churn"):

        prob = model.predict_proba(data)[0][1]

        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

        if prob > 0.6:
            st.error("High Risk Customer")
        elif prob > 0.3:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")
        st.write(f"Model Used: **{model_choice}**")
        st.subheader("Prediction Explanation (SHAP)")

        X_transformed = preprocessor.transform(data)

        explainer = shap.Explainer(rf_classifier)
        shap_values = explainer(X_transformed_df)

        fig = plt.figure()

        shap.plots.waterfall(
            shap_values[0, :, 1],
            show=False
        )
        st.pyplot(fig)

with tab2:
    preprocessor = rf_model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    rf_classifier = rf_model.named_steps["model"]
    feature_importance = rf_classifier.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    top_features = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Drivers of Customer Churn")
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")

    sample_data = pd.concat([data]*50, ignore_index=True)

    sample_data["tenure"] = np.random.randint(0, 72, size=50)
    sample_data["MonthlyCharges"] = np.random.uniform(20, 120, size=50)
    sample_data["TotalCharges"] = np.random.uniform(100, 5000, size=50)

    X_sample_transformed = preprocessor.transform(sample_data)
    feature_names = preprocessor.get_feature_names_out()

    X_sample_df = pd.DataFrame(
        X_sample_transformed,
        columns=feature_names
    )
    explainer = shap.Explainer(rf_classifier)
    shap_values = explainer(X_sample_df)

    fig = plt.figure()
    shap.plots.beeswarm(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)

    st.subheader("SHAP Feature Importance")

    fig = plt.figure()
    shap.plots.bar(
        shap_values[:, :, 1],
        max_display=15,
        show=False
    )
    st.pyplot(fig)

    performance_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "ROC AUC": [0.86, 0.85, 0.85],
    "F1 Score": [0.64, 0.65, 0.63],
    "Precision": [0.52, 0.56, 0.55],
    "Recall": [0.84, 0.78, 0.75]
})

    st.subheader("Model Performance")
    st.dataframe(performance_df)

    st.subheader("Business Insights")
    st.markdown("""
### Key Drivers of Customer Churn

Based on the model analysis and feature importance results, several factors significantly influence customer churn:

**1️⃣ Customer Tenure**
- Customers with shorter tenure are much more likely to churn.
- New customers have a higher probability of leaving compared to long-term subscribers.

**2️⃣ Contract Type**
- Customers on **month-to-month contracts** show the highest churn risk.
- Long-term contracts such as **one-year or two-year agreements significantly reduce churn**.

**3️⃣ Monthly Charges**
- Higher monthly charges correlate with increased churn probability.
- Customers paying more are more likely to switch providers if they perceive better value elsewhere.

**4️⃣ Internet Service Type**
- Customers using **fiber optic internet services** show relatively higher churn rates compared to DSL users.

**5️⃣ Lack of Value-Added Services**
- Customers without services like **online security, tech support, or device protection** are more likely to churn.

---

### Business Recommendations

• Encourage **long-term contracts** through discounts or loyalty rewards.  
• Offer **bundled services (security, tech support)** to increase customer retention.  
• Provide **special retention offers for high-charge customers** to reduce churn risk.  
• Focus retention campaigns on **new customers with low tenure**.
""")
