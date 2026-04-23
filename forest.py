import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(
    page_title="Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "tuned_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "target_encoder.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))


st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>🌲 Forest Cover Type Prediction</h1>
    <p style='text-align: center;'>Predict forest type based on environmental features</p>
""", unsafe_allow_html=True)

st.divider()


# -------- INPUTS --------
Elevation = st.number_input("Elevation", 0, 4000, 2000)
Aspect = st.number_input("Aspect", 0, 360)
Slope = st.number_input("Slope", 0, 90)

Horizontal_Distance_To_Hydrology = st.number_input("Distance to Hydrology")
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance to Hydrology")
Horizontal_Distance_To_Roadways = st.number_input("Distance to Roadways")
Horizontal_Distance_To_Fire_Points = st.number_input("Distance to Fire Points")

Hillshade_9am = st.number_input("Hillshade 9am", 0, 255)
Hillshade_Noon = st.number_input("Hillshade Noon", 0, 255)
Hillshade_3pm = st.number_input("Hillshade 3pm", 0, 255)

Wilderness_Area = st.selectbox("Wilderness Area", [1,2,3,4])
Soil_Type = st.number_input("Soil Type (1-40)", 1, 40)

# -------- DERIVED FEATURES --------
Distance_To_Hydrology = np.sqrt(Horizontal_Distance_To_Hydrology**2 + Vertical_Distance_To_Hydrology**2)
Hillshade_Mean = (Hillshade_9am + Hillshade_Noon + Hillshade_3pm) / 3
Hillshade_Range = Hillshade_3pm - Hillshade_9am
Fire_Risk = Horizontal_Distance_To_Fire_Points / (Horizontal_Distance_To_Roadways + 1)

# -------- DATAFRAME --------
input_df = pd.DataFrame([{
    "Elevation": Elevation,
    "Aspect": Aspect,
    "Slope": Slope,
    "Vertical_Distance_To_Hydrology": Vertical_Distance_To_Hydrology,
    "Horizontal_Distance_To_Roadways": Horizontal_Distance_To_Roadways,
    "Horizontal_Distance_To_Fire_Points": Horizontal_Distance_To_Fire_Points,
    "Hillshade_9am": Hillshade_9am,

    # ✅ derived features ONLY
    "Distance_To_Hydrology": Distance_To_Hydrology,
    "Hillshade_Mean": Hillshade_Mean,
    "Fire_Risk": Fire_Risk,

    # categorical
    "Wilderness_Area": Wilderness_Area,
    "Soil_Type": Soil_Type
}])

# -------- ENCODING --------
cat_cols = ["Wilderness_Area", "Soil_Type"]

encoded = encoder.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

final_input = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)

final_input = final_input.reindex(columns=features, fill_value=0)

# -------- PREDICTION --------
if st.button("Predict"):

    # 🔮 Prediction
    prediction = model.predict(final_input)
    result = le.inverse_transform(prediction)

    st.divider()
    st.subheader("🌳 Prediction Result")
    st.success(f"Predicted Forest Type: **{result[0]}**")

    # 📊 Create plot_df HERE
    plot_df = input_df.T.reset_index()
    plot_df.columns = ["Feature", "Value"]

    if hasattr(model, "feature_importances_"):
        st.subheader("🌟 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        # take top 15
        top_features = importance_df.head(15)

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            title="Top 15 Important Features"
        )

        # make highest importance on top
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)