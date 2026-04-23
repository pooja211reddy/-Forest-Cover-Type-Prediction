# 🌲 Forest Cover Type Prediction

A Machine Learning project to classify forest cover types using cartographic features, built with Random Forest and deployed via Streamlit.

---

## 📌 Problem Statement
Predict the forest cover type (e.g., Aspen, Lodgepole Pine, Spruce/Fir) using environmental and geographical features such as:

- Elevation  
- Slope & Aspect  
- Distance to Hydrology  
- Hillshade values  
- Soil Type  
- Wilderness Area  

---

## 🎯 Objective
- Build a high-performance classification model  
- Handle class imbalance effectively  
- Improve model interpretability  
- Deploy an interactive web application  

---

## 📊 Dataset
- **Source:** UCI / Kaggle (Covertype Dataset)  
- **Rows:** ~580,000  
- **Features:** 54 (after feature selection)  
- **Target:** Cover_Type (7 classes)  

---

## ⚙️ Data Preprocessing
- Removed highly correlated features  
- Encoded categorical variables using OneHotEncoder  
- Encoded target using LabelEncoder  
- Train-test split with stratification  

---

## 🧠 Feature Engineering

Created meaningful derived features:

- `Distance_To_Hydrology = sqrt(Horizontal² + Vertical²)`  
- `Hillshade_Mean = (9am + Noon + 3pm) / 3`  
- `Hillshade_Range = 3pm - 9am`  
- `Near_Water = 1 if distance < 100 else 0`  
- `Fire_Risk = Fire_Points / (Roadways + 1)`  

👉 These significantly improved model performance and interpretability.

---

## 🤖 Model
- Random Forest Classifier  
- Used `class_weight="balanced"` to handle imbalance  
- Hyperparameter tuning using RandomizedSearchCV  
- Final model optimized for efficiency  

---

## 📈 Model Performance
- **Accuracy:** ~95.5%  

### 🔍 Key Observations
- High performance across most classes  
- Aspen class has lower recall (~84%)  
- Confusion mainly between:
  - Aspen ↔ Lodgepole Pine  

---

## 🧪 Model Explainability (SHAP)
Used SHAP to:
- Understand feature importance  
- Explain predictions  
- Identify misclassification causes  

👉 Found that **Elevation heavily influences predictions**, causing overlap errors.

---

## 🌐 Streamlit App
Features:
- Interactive sliders for input  
- Real-time prediction  
- Feature importance chart  
- SHAP explanation plots  

---

## 🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
SHAP
Plotly
Streamlit

---

## ⚠️ Challenges
Class imbalance
Feature overlap between similar forest types
Ensuring feature consistency during deployment

---

## ✅ Solutions
Used class_weight="balanced" instead of heavy SMOTE
Added engineered features
Applied SHAP for debugging
Ensured feature alignment in Streamlit

---

## 💡 Key Learnings
High accuracy does NOT mean perfect classification
Feature engineering > hyperparameter tuning
Model explainability is critical in real-world ML
Deployment requires strict feature consistency

---

## 🔮 Future Improvements
Switch to XGBoost for better class separation
Add SHAP dashboard in Streamlit
Improve Aspen classification with targeted modeling
Deploy app to cloud (Streamlit Cloud / AWS)

---

## 👩‍💻 Author
Pooja Reddy Nedhunuri
Machine Learning Enthusiast 🚀
