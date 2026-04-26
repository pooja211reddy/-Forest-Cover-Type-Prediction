import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "tuned_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "target_encoder.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
# ===================== 🎨 STYLING =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0E1117, #111827);
    color: white;
}

h1, h2, h3 {
    color: #00C9FF;
}

.card {
    background: #1E293B;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-card {
    background: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.stButton>button {
    border-radius: 12px;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===================== 🧠 HEADER =====================
st.title("🌲 Forest Cover Type Dashboard")

st.markdown("""
<div class="card">
Interactive ML dashboard for predicting forest types and understanding model behavior.
</div>
""", unsafe_allow_html=True)


# ===================== 🔧 PREP INPUT =====================
import numpy as np
import pandas as pd


with st.form("input_form"):
    st.markdown("### 📝 Enter Input Features")

    col1, col2 = st.columns(2)

    with col1:
        Elevation = st.number_input("Elevation", value=2500)
        Aspect = st.number_input("Aspect", value=180)
        Slope = st.number_input("Slope", value=10)
        Horizontal_Distance_To_Hydrology = st.number_input("Distance to Hydrology", value=300)
        Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance", value=0)

    with col2:
        Horizontal_Distance_To_Roadways = st.number_input("Distance to Roads", value=1000)
        Horizontal_Distance_To_Fire_Points = st.number_input("Distance to Fire Points", value=1500)
        Hillshade_9am = st.number_input("Hillshade 9am", value=200)
        Hillshade_Noon = st.number_input("Hillshade Noon", value=220)
        Hillshade_3pm = st.number_input("Hillshade 3pm", value=150)

    Wilderness_Area = st.selectbox("Wilderness Area", [1, 2, 3, 4])
    Soil_Type = st.number_input("Soil Type", value=10)
    # all inputs here

    submit = st.form_submit_button("🚀 Predict")

if submit:
    Distance_To_Hydrology = np.sqrt(Horizontal_Distance_To_Hydrology**2 + Vertical_Distance_To_Hydrology**2)
    Hillshade_Mean = (Hillshade_9am + Hillshade_Noon + Hillshade_3pm) / 3
    Hillshade_Range = Hillshade_3pm - Hillshade_9am
    Near_Water = 1 if Horizontal_Distance_To_Hydrology < 100 else 0
    Fire_Risk = Horizontal_Distance_To_Fire_Points / (Horizontal_Distance_To_Roadways + 1)

    input_df = pd.DataFrame([{
        "Elevation": Elevation,
        "Aspect": Aspect,
        "Slope": Slope,
        "Horizontal_Distance_To_Hydrology": Horizontal_Distance_To_Hydrology,
        "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
        "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
        "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
        "Hillshade_9am": Hillshade_9am,
        "Hillshade_Noon": Hillshade_Noon,
        "Hillshade_3pm": Hillshade_3pm,
        "Distance_To_Hydrology": Distance_To_Hydrology,
        "Hillshade_Mean": Hillshade_Mean,
        "Hillshade_Range": Hillshade_Range,
        "Near_Water": Near_Water,
        "Fire_Risk": Fire_Risk,
        "Wilderness_Area": Wilderness_Area,
        "Soil_Type": Soil_Type
    }])
    # Encoding
    cat_cols = ["Wilderness_Area", "Soil_Type"]
    encoded = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    final_input = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)
    final_input = final_input.reindex(columns=features, fill_value=0)

    # Prediction
    prediction = model.predict(final_input)
    result = le.inverse_transform(prediction)[0]

    proba = model.predict_proba(final_input)[0]
    confidence = max(proba)

    # ===================== 🎯 KPI CARDS =====================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h4>🌳 Prediction</h4>
        <h2>{result}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h4>📊 Confidence</h4>
        <h2>{confidence:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h4>🔢 Classes</h4>
        <h2>{len(le.classes_)}</h2>
        </div>
        """, unsafe_allow_html=True)

    # ===================== 📊 PROBABILITY CHART =====================
    import plotly.express as px

    proba_df = pd.DataFrame({
        "Class": le.classes_,
        "Probability": proba
    }).sort_values(by="Probability", ascending=False)

    st.markdown("### 📊 Prediction Probabilities")

    fig = px.bar(
        proba_df.head(5),
        x="Probability",
        y="Class",
        orientation="h",
        color="Probability",
        color_continuous_scale="Tealgrn"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===================== 🌍 MAP =====================
    if result in ["Aspen", "Lodgepole Pine"]:
        st.markdown("### 🗺️ Likely Habitat Region")

        map_df = pd.DataFrame({
            "lat": [39.5],
            "lon": [-105.5]
        })

        st.map(map_df)

    # ===================== ⭐ FEATURE IMPORTANCE =====================
    if hasattr(model, "feature_importances_"):
        st.markdown("### ⭐ Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": final_input.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ===================== 🔍 SHAP =====================
 

    if result in ["Aspen", "Lodgepole Pine"]:

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_input)

        pred_class = prediction[0]

        # ✅ Handle multi-class correctly
        if isinstance(shap_values, list):
            values = shap_values[pred_class][0]
            base_value = explainer.expected_value[pred_class]

        else:
            # shap_values shape: (n_samples, n_features, n_classes)
            values = shap_values[0, :, pred_class]
            base_value = explainer.expected_value[pred_class]

        exp = shap.Explanation(
            values=values,
            base_values=base_value,
            data=final_input.iloc[0],
            feature_names=final_input.columns
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(exp, show=False)

        st.markdown("### 🔍 Prediction Explanation (SHAP)")
        st.pyplot(fig)