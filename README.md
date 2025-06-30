# 🧪 Food Classifier

**Binary Classification of Food Products Based on Nutritional Profile**  
*Final project for the Big Data Lab course – Master’s Degree in Data Science, Business Analytics & Innovation*

---

## 🎯 Objective

The goal of this project is to develop a robust binary classifier capable of distinguishing between **healthy** and **unhealthy** food products based on their nutritional information.  
The dataset is sourced from Open Food Facts and processed using Apache Spark to efficiently handle large-scale data.

---

## 🧠 Problem Type

- **Task**: Binary classification  
- **Target Variable**: `label`  
  - `0` = Healthy  
  - `1` = Unhealthy  
- **Domain**: Nutrition, FoodTech, Public Health  
- **Toolset**: PySpark MLlib

---

## 🧰 Technologies & Tools

- **Apache Spark (PySpark)**
- `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` (for comparison)
- Spark ML: `RandomForestClassifier`, `GBTClassifier`, `LogisticRegression`

---

## 📊 Methodology

The pipeline includes:

1. **Data Loading**  
   Efficient reading and schema inspection of the dataset (`data.csv`)

2. **Data Cleaning**  
   - Handling missing values  
   - Selecting numerical features  
   - Casting and filtering  
   - Normalization and imputation

3. **Target Engineering**  
   - Custom rule for binary label assignment based on nutritional thresholds  
   - Label statistics and distribution analysis

4. **Feature Transformation**  
   - Scaling and vector assembly  
   - Feature selection using feature importance from Random Forest

5. **Model Training & Evaluation**  
   - Split into training/test sets  
   - Training multiple classifiers (RF, GBT, LR)  
   - Evaluation using accuracy, precision, recall, F1-score, AUC  
   - Confusion matrix and ROC curves

6. **Model Optimization**  
   - Hyperparameter tuning with grid search and cross-validation  
   - Ensemble approach (optional): majority voting across trees trained on balanced subsets

---

## 📈 Results

- **Best Model**: Optimized Random Forest  
- **Accuracy**: 97.9%  
- **AUC**: 0.997  
- The model shows strong generalization with minimal overfitting.  
- The most important features were:  
  - Saturated fat  
  - Sugars  
  - Calories  
  - Proteins

---

## 📁 Repository Structure

FoodClassifier.py # Final pipeline script (PySpark)
data/ # Input CSV file
figures/ # Confusion matrix, ROC curves, feature importance plots
metrics/ # Saved evaluation results




---

## 📄 Deliverables

- ✅ Final, modular pipeline in PySpark  
- 📊 Performance metrics and visualizations  
- 📁 Clean, reproducible structure suitable for real-world applications  
- 🧼 All functions documented and fully compliant with OOP principles

---

## 👨‍💻 Author

**Cristian Cernicchiaro**  
Master’s in Data Science, Business Analytics & Innovation  
University of Cagliari – A.Y. 2024/2025

---

> ⚡ *This repository is designed to be clear, scalable, and production-ready. Feel free to fork, explore, or get in touch if you're interested in collaborations.*
