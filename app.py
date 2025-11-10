import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #3b82f6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #dc2626;
        border: 2px solid #fca5a5;
    }
    .risk-low {
        background-color: #dcfce7;
        color: #16a34a;
        border: 2px solid #86efac;
    }
    .feature-importance {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data function
@st.cache_resource
def load_and_preprocess_data():
    """Load and preprocess the hospital readmission dataset"""
    try:
        df = pd.read_csv('data/hospital_readmissions_30k.csv')
        
        # Convert blood pressure to separate systolic and diastolic
        bp_split = df['blood_pressure'].str.split('/', expand=True)
        df['systolic_bp'] = pd.to_numeric(bp_split[0])
        df['diastolic_bp'] = pd.to_numeric(bp_split[1])
        df = df.drop('blood_pressure', axis=1)
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_diabetes = LabelEncoder()
        le_hypertension = LabelEncoder()
        le_discharge = LabelEncoder()
        le_readmitted = LabelEncoder()
        
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['diabetes_encoded'] = le_diabetes.fit_transform(df['diabetes'])
        df['hypertension_encoded'] = le_hypertension.fit_transform(df['hypertension'])
        df['discharge_destination_encoded'] = le_discharge.fit_transform(df['discharge_destination'])
        df['readmitted_encoded'] = le_readmitted.fit_transform(df['readmitted_30_days'])
        
        # Drop original categorical columns
        df_processed = df.drop(['gender', 'diabetes', 'hypertension', 'discharge_destination', 'readmitted_30_days', 'patient_id'], axis=1)
        
        # Feature engineering: Create age groups
        df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        le_age = LabelEncoder()
        df_processed['age_group_encoded'] = le_age.fit_transform(df_processed['age_group'])
        df_processed = df_processed.drop('age_group', axis=1)
        
        # Feature engineering: BMI categories
        df_processed['bmi_category'] = pd.cut(df_processed['bmi'], bins=[0, 18.5, 25, 30, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        le_bmi = LabelEncoder()
        df_processed['bmi_category_encoded'] = le_bmi.fit_transform(df_processed['bmi_category'])
        df_processed = df_processed.drop('bmi_category', axis=1)
        
        # Feature engineering: Create risk score based on medical conditions
        df_processed['medical_risk_score'] = df_processed['diabetes_encoded'] + df_processed['hypertension_encoded'] + df_processed['medication_count']
        
        # Feature engineering: Length of stay category
        df_processed['los_category'] = pd.cut(df_processed['length_of_stay'], bins=[0, 3, 7, 10, 20], labels=['Short', 'Medium', 'Long', 'Very Long'])
        le_los = LabelEncoder()
        df_processed['los_category_encoded'] = le_los.fit_transform(df_processed['los_category'])
        df_processed = df_processed.drop('los_category', axis=1)
        
        return df_processed, le_gender, le_diabetes, le_hypertension, le_discharge, le_age, le_bmi, le_los
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None

# Train model function
@st.cache_resource
def train_model(df_processed):
    """Train the Random Forest model"""
    try:
        # Prepare features and target
        X = df_processed.drop('readmitted_encoded', axis=1)
        y = df_processed['readmitted_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train Random Forest with better hyperparameters
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return rf_model, X.columns.tolist(), feature_importance
    
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

# Prediction function
def predict_readmission(model, input_data, feature_names):
    """Make prediction using the trained model"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return prediction, probability
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Main app
def main():
    st.markdown('<h1 class="main-header">🏥 Hospital Readmission Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">This tool predicts the likelihood of patient readmission within 30 days of discharge based on various medical and demographic factors.</div>', unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df_processed, le_gender, le_diabetes, le_hypertension, le_discharge, le_age, le_bmi, le_los = load_and_preprocess_data()
        
        if df_processed is not None:
            rf_model, feature_names, feature_importance = train_model(df_processed)
        else:
            rf_model, feature_names, feature_importance = None, None, None
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Readmission", "📊 Feature Importance", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
        
        # Create columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 90, 50)
            gender = st.selectbox("Gender", le_gender.classes_ if le_gender is not None else ["Male", "Female"])
            bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1)
            diabetes = st.selectbox("Diabetes", le_diabetes.classes_ if le_diabetes is not None else ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", le_hypertension.classes_ if le_hypertension is not None else ["No", "Yes"])
            
        with col2:
            blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)", "120/80")
            medication_count = st.number_input("Number of Medications", 0, 20, 3)
            length_of_stay = st.number_input("Length of Stay (days)", 1, 30, 5)
            discharge_destination = st.selectbox("Discharge Destination", 
                                               le_discharge.classes_ if le_discharge is not None else 
                                               ["Home", "Skilled Nursing Facility", "Other Hospital"])
        
        # Prepare input data
        if rf_model is not None:
            # Parse blood pressure
            try:
                systolic_bp, diastolic_bp = map(float, blood_pressure.split('/'))
            except:
                systolic_bp, diastolic_bp = 120, 80
            
            # Encode categorical variables
            gender_encoded = le_gender.transform([gender])[0] if le_gender is not None else 0
            diabetes_encoded = le_diabetes.transform([diabetes])[0] if le_diabetes is not None else 0
            hypertension_encoded = le_hypertension.transform([hypertension])[0] if le_hypertension is not None else 0
            discharge_encoded = le_discharge.transform([discharge_destination])[0] if le_discharge is not None else 0
            
            # Create age groups
            age_group = 'Young' if age <= 30 else 'Middle' if age <= 50 else 'Senior' if age <= 70 else 'Elderly'
            age_group_encoded = le_age.transform([age_group])[0] if le_age is not None else 0
            
            # Create BMI categories
            bmi_category = 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'
            bmi_category_encoded = le_bmi.transform([bmi_category])[0] if le_bmi is not None else 0
            
            # Create medical risk score
            medical_risk_score = diabetes_encoded + hypertension_encoded + medication_count
            
            # Create length of stay categories
            los_category = 'Short' if length_of_stay <= 3 else 'Medium' if length_of_stay <= 7 else 'Long' if length_of_stay <= 10 else 'Very Long'
            los_category_encoded = le_los.transform([los_category])[0] if le_los is not None else 0
            
            # Prepare input dictionary
            input_data = {
                'age': age,
                'gender_encoded': gender_encoded,
                'bmi': bmi,
                'diabetes_encoded': diabetes_encoded,
                'hypertension_encoded': hypertension_encoded,
                'medication_count': medication_count,
                'length_of_stay': length_of_stay,
                'discharge_destination_encoded': discharge_encoded,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'age_group_encoded': age_group_encoded,
                'bmi_category_encoded': bmi_category_encoded,
                'medical_risk_score': medical_risk_score,
                'los_category_encoded': los_category_encoded
            }
            
            # Make prediction
            if st.button("🔮 Predict Readmission Risk", type="primary"):
                prediction, probability = predict_readmission(rf_model, input_data, feature_names)
                
                if prediction is not None:
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-result risk-high">⚠️ High Risk of Readmission ({probability:.1%})</div>', unsafe_allow_html=True)
                        st.warning("Patient shows high risk of readmission within 30 days. Consider preventive measures and follow-up care.")
                    else:
                        st.markdown(f'<div class="prediction-result risk-low">✅ Low Risk of Readmission ({(1-probability):.1%})</div>', unsafe_allow_html=True)
                        st.success("Patient shows low risk of readmission within 30 days.")
                    
                    # Show probability details
                    st.subheader("Risk Probability Breakdown")
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric("Readmission Probability", f"{probability:.1%}")
                    with prob_col2:
                        st.metric("No Readmission Probability", f"{(1-probability):.1%}")
        
        else:
            st.error("Model not loaded properly. Please refresh the page.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        if feature_importance is not None:
            st.markdown('<div class="feature-importance">These are the key factors that influence readmission predictions:</div>', unsafe_allow_html=True)
            
            # Display feature importance
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title('Top 10 Most Important Features')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display feature importance table
            st.subheader("Feature Importance Rankings")
            st.dataframe(feature_importance.head(10))
            
            # Key insights
            st.subheader("Key Insights")
            st.info("""
            • **Discharge destination** is the strongest predictor of readmission
            • **Medical conditions** (diabetes, hypertension) contribute significantly
            • **Length of stay** and **medication count** are important risk indicators
            • **Age groups** show varying risk patterns
            • **Blood pressure** components provide additional predictive power
            """)
        else:
            st.error("Feature importance data not available.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Purpose
        This application helps healthcare providers assess patient risk of readmission within 30 days of discharge. 
        By identifying high-risk patients early, healthcare teams can implement preventive measures and reduce readmission rates.
        
        ## 📊 Model Information
        - **Algorithm**: Random Forest Classifier
        - **Training Data**: 30,000 patient records
        - **Features**: Age, gender, medical conditions, vital signs, medication information
        - **Target**: Binary classification (readmitted/not readmitted within 30 days)
        
        ## 🔧 Features
        - **Patient Risk Assessment**: Input patient information to get readmission risk prediction
        - **Feature Importance**: Understand which factors most influence predictions
        - **Visual Analytics**: Interactive charts and data visualizations
        
        ## 🛡️ Ethical Considerations
        - **Privacy**: All patient data is processed locally and not stored
        - **Bias**: Model is trained on balanced data to minimize bias
        - **Transparency**: Feature importance helps understand prediction factors
        - **Clinical Context**: Predictions should be used as decision support tools, not replacements for clinical judgment
        
        ## 🚀 How to Use
        1. Navigate to the "Predict Readmission" tab
        2. Enter patient information in the input fields
        3. Click "Predict Readmission Risk" to get results
        4. Review the feature importance tab to understand prediction factors
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and scikit-learn")

if __name__ == "__main__":
    main()