**Salary Prediction Application**
This repository contains a machine learning–based web application designed to estimate salaries using professional and demographic inputs. 
The project demonstrates the practical implementation of a complete machine learning pipeline along with deployment using Streamlit.

**Tech Stack**
Python, Streamlit, Pandas, Scikit-learn

**Machine Learning Implementation**

Model Used: Linear Regression
Prediction Inputs:Age, Years of Experience, Education Level, Job Title

Data Processing:
Numerical variables are standardized using StandardScaler
Categorical variables are transformed using OneHotEncoder
All transformations and model training are combined into a single Scikit-learn pipeline

**Optimization Strategy**

The application leverages Streamlit’s @st.cache_resource to cache the trained model pipeline. 
This prevents repeated training during application reruns and ensures fast prediction performance.

**Deployment**
The application is deployed on the Streamlit platform, allowing users to interact with the model through a browser-based interface and receive real-time salary predictions.

**Live Application**

The application is deployed using Streamlit and can be accessed here:

https://salary-predictor-app-a3o5ynw7qxtbjuh8pjgtz8.streamlit.app/

