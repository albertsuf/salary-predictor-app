import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# --- 1. Set Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# --- Define the path to the data file ---
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Salary Data.csv')

# --- Use Streamlit's caching for resource-heavy operations ---
@st.cache_resource
def load_and_train_model():
    """
    Loads data, trains the model, and returns the trained pipeline and app job titles.
    This function is cached by Streamlit to run only once.
    """
    # Load the dataset
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        st.error(f"ERROR: Data file not found at '{DATA_FILE_PATH}'. Please ensure 'Salary Data.csv' is in the 'data/' directory.")
        st.stop()
    
    # Data Cleaning and Type Conversion
    df_cleaned = df.dropna().copy()
    df_cleaned['Age'] = df_cleaned['Age'].astype(int)
    df_cleaned['Years of Experience'] = df_cleaned['Years of Experience'].astype(float)
    df_cleaned['Salary'] = df_cleaned['Salary'].astype(int)

    # Handle 'Job Title' High Cardinality
    JOB_TITLE_FREQ_THRESHOLD = 2
    job_title_counts = df_cleaned['Job Title'].value_counts()
    job_titles_to_keep = job_title_counts[job_title_counts >= JOB_TITLE_FREQ_THRESHOLD].index.tolist()
    df_cleaned['Job Title Grouped'] = df_cleaned['Job Title'].apply(
        lambda x: x if x in job_titles_to_keep else 'Other'
    )
    
    # Get sorted lists for dropdowns
    app_education_levels = sorted(df_cleaned['Education Level'].unique().tolist())
    app_job_titles = sorted(df_cleaned['Job Title Grouped'].unique().tolist())

    # Define Features (X) and Target (y) - 'Gender' has been removed
    X = df_cleaned[['Age', 'Education Level', 'Years of Experience', 'Job Title Grouped']]
    y = df_cleaned['Salary']

    # Create a Preprocessing Pipeline - 'Gender' has been removed
    numerical_cols = ['Age', 'Years of Experience']
    categorical_cols = ['Education Level', 'Job Title Grouped']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the Model
    model_pipeline.fit(X, y)

    return model_pipeline, app_education_levels, app_job_titles

# --- Load and Train Model (this call will be cached) ---
model, education_levels, job_titles = load_and_train_model()

# --- Streamlit App UI ---
col1, col2 = st.columns([1, 6]) 

with col1:
    st.image("assets/dashboard5.png", width=400)

with col2:
    st.title("Salary Prediction App") 

# st.markdown("This app predicts salary based on your details like age, year of experience, education level and job title.")
# st.markdown("Fill the below details to make a prediction.")
st.markdown("<h2 style='font-size: 20px;'>This app predicts salary based on your details like age, year of experience, education level and job title.</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px;'>Fill in the details below to make a prediction.</p>", unsafe_allow_html=True)
# How to Use Section
with st.expander("🤔 How to Use This App"):
    st.write("""
        1.  **Adjust Sliders** 🎚️: Drag the sliders for 'Age' and 'Years of Experience' to match your details.
        2.  **Make Selections** 🖱️: Choose your 'Education Level' and 'Job Title' from the dropdown menus.
        3.  **Predict!** 🚀: Click the 'Predict Salary' button to see your estimated salary.
    """)

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("📊 Enter Your Details")

    # Use columns for a better layout
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
    with col2:
        years_experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, 0.5)
    
    education_level = st.selectbox("Education Level", education_levels)
    job_title_grouped = st.selectbox("Job Title", job_titles)

    # Submit button for the form
    submitted = st.form_submit_button("🚀 Predict Salary")

if submitted:
    # Create a DataFrame from the inputs - 'Gender' has been removed
    input_data = {
        'Age': [age],
        'Education Level': [education_level],
        'Years of Experience': [years_experience],
        'Job Title Grouped': [job_title_grouped]
    }
    input_df = pd.DataFrame(input_data)

    # Make prediction
    predicted_salary = model.predict(input_df)[0]
    
    # Display result using st.metric
    st.metric(label="Predicted Salary", value=f"₹{predicted_salary:,.0f}")
    st.info("Note: This is an estimation. Actual salaries may vary based on location, company, and other factors.", icon="💡")


# --- About Section ---
st.markdown("---")
with st.expander("About This App"):
    st.write("""
        This application predicts salaries using a machine learning model built with Python. Here’s a breakdown of how it works:

        **Technology & Libraries Used:**
        - **Streamlit:** For creating the interactive web application interface.
        - **Pandas:** For loading and manipulating the data from the CSV file.
        - **Scikit-learn:** For building and training the machine learning model pipeline.

        **Model Details:**
        - **Algorithm:** The model uses **Linear Regression**, a standard algorithm for predicting a continuous value like salary.
        - **Features:** The prediction is based on the following inputs: **Age**, **Years of Experience**, **Education Level**, and **Job Title**.
        - **Preprocessing:** Before training, the model uses a `ColumnTransformer` to prepare the data. Numerical features are scaled with `StandardScaler`, and categorical features are encoded with `OneHotEncoder`.

        **Performance Note:**
        The app uses Streamlit's `@st.cache_resource` decorator. This means the model is trained only once when the app starts, making subsequent predictions very fast.

        **Disclaimer:**
        This tool provides an estimate for educational purposes only. Actual salaries can vary based on location, company, and individual skills.
    """)

st.markdown("---")
st.markdown("Application by [Akash Ranjan](https://github.com/albertsuf)")
