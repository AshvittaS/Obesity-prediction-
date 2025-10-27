# Obesity Level Prediction App

A sophisticated machine learning application that predicts obesity categories using a unique hybrid approach combining unsupervised clustering with supervised learning models.

## 🚀 Live Demo

**[Try the App on Hugging Face Spaces](https://ashvitta07-obesity-prediction.hf.space/)**

## 🎯 Project Overview

This project implements an innovative approach to obesity prediction by using K-Means clustering to discover natural obesity patterns in the data, then training supervised models on these discovered clusters. The final Random Forest model achieves **99.76% accuracy**.

## 🔬 Unique Technical Approach

### Hybrid Unsupervised-Supervised Learning
- **Clustering First**: Uses K-Means clustering (k=3) to discover natural obesity patterns
- **Cluster-to-Label Mapping**: Maps discovered clusters to meaningful obesity categories:
  - Cluster 0: "Obesity_Type_II"
  - Cluster 1: "Normal_Weight" 
  - Cluster 2: "Overweight"
- **Supervised Training**: Trains multiple ML models on the cluster-derived labels

### Advanced Preprocessing Pipeline
- **Outlier Treatment**: Custom IQR-based outlier capping (preserves data integrity)
- **Power Transformation**: Yeo-Johnson transformation for skewed numerical features
- **Categorical Encoding**: Label encoding for all categorical variables
- **Standardization**: Final scaling for optimal model performance

## 📊 Dataset Features

The model analyzes 16 comprehensive lifestyle and health attributes:

**Physical Attributes:**
- Age, Height, Weight, Gender

**Dietary Habits:**
- FAVC (Frequent High Caloric Food)
- FCVC (Vegetable Consumption)
- NCP (Number of Meals per Day)
- CAEC (Eating Between Meals)

**Health & Lifestyle:**
- Family History with Overweight
- Smoking Status
- Water Consumption (CH2O)
- Caloric Intake Monitoring (SCC)
- Physical Activity Frequency (FAF)
- Technology Usage Time (TUE)
- Alcohol Consumption (CALC)
- Transportation Method (MTRANS)

## 🏆 Model Performance

| Model | Accuracy |
|-------|----------|
| **Random Forest** | **99.76%** |
| Gradient Boosting | 99.53% |
| AdaBoost | 99.53% |
| XGBoost | 99.53% |
| Logistic Regression | 99.05% |
| Naive Bayes | 85.82% |

## 🛠️ Technical Stack

- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Interface**: Gradio
- **Deployment**: Hugging Face Spaces

## 📁 Project Structure

```
├── obesity.ipynb          # Complete analysis notebook
├── app.py                 # Gradio web application
├── best_model.pkl         # Trained Random Forest model
├── power_transformer.pkl  # PowerTransformer for numerical features
├── scaler.pkl            # StandardScaler for final scaling
├── label_encoders.pkl    # LabelEncoders for categorical features
├── ObesityDataSet.csv    # Original dataset
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Locally:**
   ```bash
   python app.py
   ```

3. **Access the App:**
   Open your browser to `http://localhost:7860`

## 🔧 Key Features

- **Real-time Prediction**: Instant obesity level classification
- **Comprehensive Input**: 16 lifestyle and health factors
- **User-friendly Interface**: Clean, intuitive Gradio interface
- **Production Ready**: Complete preprocessing pipeline
- **High Accuracy**: 99.76% prediction accuracy
- **Modular Design**: Easy to maintain and update

## 📈 Clustering Analysis

- **Optimal Clusters**: 3 clusters determined by Elbow method
- **Silhouette Score**: 0.5165 (good cluster separation)
- **Cluster Distribution**: Balanced representation across obesity categories

## 🎨 App Interface

The web application provides:
- Dropdown menus for categorical variables
- Number inputs for continuous variables
- Real-time prediction display
- Professional, responsive design
- Dark/light theme support

## 🔬 Methodology

1. **Data Exploration**: Comprehensive EDA and feature analysis
2. **Preprocessing**: Advanced outlier handling and transformation
3. **Clustering**: K-Means to discover natural obesity patterns
4. **Model Training**: Multiple algorithms tested and compared
5. **Model Selection**: Best performing model saved and deployed
6. **Web Deployment**: Production-ready Gradio interface

## 📝 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Python, scikit-learn, and Gradio**
