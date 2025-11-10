# Hospital Readmission Prediction Assignment Answers

## Assignment Overview
This document provides comprehensive answers to the Week 5 assignment questions based on the hospital readmission prediction project implemented in `hospital_readmission_improved.py`.

---

## 1. Problem Definition (6 points)

### Problem Definition
**Problem**: Predicting patient readmission within 30 days of discharge

**Objectives**:
1. Identify high-risk patients for early intervention
2. Reduce readmission rates through targeted care
3. Optimize resource allocation for at-risk patients

**Stakeholders**:
1. Hospital administrators
2. Healthcare providers
3. Insurance companies
4. Patients and families

**Key Performance Indicator (KPI)**:
- **Readmission Reduction Rate**: Percentage decrease in 30-day readmissions among predicted high-risk patients who receive targeted interventions

---

## 2. Data Collection & Preprocessing (8 points)

### Data Sources
Based on the project implementation, the following data sources would be used:

1. **Electronic Health Records (EHRs)**: Patient medical records, diagnoses, procedures, and treatment history
2. **Patient Demographics and Medical History**: Age, gender, chronic conditions, medication information, and vital signs
3. **Laboratory Results and Vital Signs**: Blood pressure, cholesterol levels, BMI, and other biomarkers
4. **Medication Records and Adherence Data**: Current medications, dosage, and patient compliance tracking
5. **Social Determinants of Health**: Information about living conditions, socioeconomic status, and support systems

### Potential Bias in Data
**Selection Bias**: The dataset may not represent the full diversity of the patient population. Specific demographic groups (e.g., elderly patients, those with multiple chronic conditions) may be overrepresented, while other groups (e.g., younger patients with rare conditions) may be underrepresented. This can lead to models that perform well for certain groups but poorly for others.

### Preprocessing Steps
Based on the implementation in [`hospital_readmission_improved.py`](PLP_AI_For_Software_Engineering_Week_5/hospital_readmission_improved.py:72-136):

1. **Data Cleaning and Feature Engineering**:
   - Blood pressure processing: Split into systolic and diastolic components
   - Age group categorization (Young, Middle, Senior, Elderly)
   - BMI classification (Underweight, Normal, Overweight, Obese)
   - Medical risk score calculation (diabetes + hypertension + medication count)
   - Length of stay categorization (Short, Medium, Long, Very Long)
   - Blood pressure risk indicator creation
   - Age-length of stay interaction features

2. **Categorical Variable Encoding**:
   - Label encoding for all categorical features
   - Conversion of text-based categories to numerical values

3. **Class Imbalance Handling**:
   - Upsampling of minority class (readmitted patients) to balance the dataset
   - Implementation of balanced class weights in models

4. **Feature Scaling and Normalization**:
   - Standard scaling of numerical features for model compatibility
   - Preparation of features for machine learning algorithms

---

## 3. Model Development (8 points)

### Selected Model and Justification
**Selected Model**: Random Forest Classifier with hyperparameter tuning

**Justification**:
1. **Handles Mixed Data Types**: Effectively processes both numerical and categorical features without requiring extensive preprocessing
2. **Robust to Outliers**: Less sensitive to extreme values compared to linear models
3. **Feature Importance Insights**: Provides transparent feature importance rankings, crucial for clinical interpretability
4. **Non-linear Relationship Handling**: Captures complex interactions between medical variables and readmission risk
5. **Performance with Imbalanced Data**: Performs well with class weighting and upsampling techniques
6. **Interpretability Balance**: Offers better interpretability than black-box models while maintaining high accuracy

### Data Splitting Strategy
From [`hospital_readmission_improved.py`](PLP_AI_For_Software_Engineering_Week_5/hospital_readmission_improved.py:169-170):

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

**Splitting Approach**:
- **Training Set**: 70% of the data for model training
- **Test Set**: 30% of the data for final evaluation
- **Stratified Sampling**: Ensures proportional representation of both readmitted and non-readmitted patients in both sets
- **Random State**: Fixed at 42 for reproducible results

### Hyperparameters for Tuning
From [`hospital_readmission_improved.py`](PLP_AI_For_Software_Engineering_Week_5/hospital_readmission_improved.py:191-199):

1. **n_estimators**: Number of decision trees in the forest
   - Values tested: [100, 200]
   - Importance: More trees generally improve performance but increase computational cost
   - Impact: Affects model stability and reduces overfitting

2. **max_depth**: Maximum depth of each decision tree
   - Values tested: [10, 20, None]
   - Importance: Controls model complexity and prevents overfitting
   - Impact: Deeper trees capture more patterns but risk memorizing noise

3. **min_samples_split**: Minimum samples required to split a node
   - Values tested: [2, 5, 10]
   - Importance: Prevents overfitting by requiring sufficient samples for splits
   - Impact: Higher values create more conservative trees

---

## 4. Evaluation & Deployment (8 points)

### Selected Evaluation Metrics

1. **F1-Score**:
   - **Relevance**: Balances precision and recall, crucial for imbalanced datasets
   - **Importance**: Prevents focusing solely on one metric (like accuracy) that can be misleading with imbalanced classes
   - **Application**: Measures the harmonic mean of precision and recall for the readmission class

2. **ROC-AUC Score**:
   - **Relevance**: Measures the model's ability to distinguish between classes across all classification thresholds
   - **Importance**: Provides threshold-independent evaluation of model performance
   - **Application**: Evaluates how well the model ranks readmission risk, essential for clinical decision-making

### Concept Drift and Monitoring

**Concept Drift Definition**: The gradual or sudden change in the relationship between input features and the target variable over time. In healthcare, this could occur due to:
- Changes in treatment protocols
- Population health shifts
- Seasonal variations in disease patterns
- Healthcare policy changes
- Evolution of medical practices

**Monitoring Strategy**:
1. **Performance Tracking**: Regular monitoring of model metrics (precision, recall, F1-score) on new patient data
2. **Drift Detection Algorithms**: Implement statistical methods to detect significant changes in:
   - Feature distributions
   - Target variable distribution
   - Prediction patterns
3. **Automated Alerts**: Set up alerts when performance drops below predefined thresholds
4. **Scheduled Retraining**: Implement quarterly or bi-annual model retraining pipelines
5. **Feedback Loops**: Collect clinician feedback on prediction accuracy and clinical relevance
6. **A/B Testing**: Continuously compare model predictions against clinical outcomes

### Technical Challenge During Deployment

**Challenge**: Scalability and Real-time Performance

**Description**: The model needs to handle predictions for potentially thousands of patients across multiple hospital departments while maintaining low latency for clinical decision-making.

**Mitigation Strategies**:
1. **Model Optimization**: Implement model pruning and feature selection to reduce computational complexity
2. **Caching**: Cache frequent predictions for common patient profiles
3. **Asynchronous Processing**: Use message queues for non-urgent predictions
4. **Load Balancing**: Distribute prediction requests across multiple servers
5. **Edge Computing**: Deploy lightweight models at departmental locations
6. **Performance Monitoring**: Continuously track prediction latency and throughput metrics
7. **Resource Scaling**: Implement auto-scaling based on demand patterns

---

## Summary

This hospital readmission prediction project addresses all the key components of the assignment:

1. **Problem Definition**: Clear objectives and stake identification for predicting 30-day readmission
2. **Data Strategy**: Comprehensive preprocessing pipeline addressing data quality and bias concerns
3. **Model Development**: Random Forest with robust hyperparameter tuning and class balancing
4. **Evaluation**: Multiple metrics focusing on clinical relevance and model reliability
5. **Deployment Planning**: Realistic considerations for scalability and concept drift monitoring

The implementation demonstrates practical AI engineering principles while maintaining focus on ethical considerations, clinical interpretability, and real-world applicability.