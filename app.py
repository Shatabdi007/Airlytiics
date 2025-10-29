import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

st.set_page_config(page_title=" AirLytics:✈Aviation Severity Predictor", layout="wide")

# Custom Styling
st.markdown(""" <style> 
                .main { 
                    background-color: #f0f2f6; 
                    padding: 2rem; 
                    border-radius: 10px; 
                } 
                h1, h2, h3 { 
                    color: #0066cc; 
                } 
                .stButton > button { 
                    background-color: #0066cc; 
                    color: white; 
                    font-weight: bold; 
                    border-radius: 0.5rem; 
                } 
                .stTextInput, .stNumberInput, .stSelectbox { 
                    border: 1px solid #0066cc !important; 
                } 
                </style> """, unsafe_allow_html=True)

# Navigation
nav = st.sidebar.selectbox("Navigation", ["Home", "Incident Analysis", "About"])

# Upload CSV
st.sidebar.header("📁 Upload Aviation Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='latin1')

if nav == "Home":
    # Title
    st.title("AirLytics: ✈ Aviation Damage Severity Prediction")
    st.markdown("Predict the severity of aircraft incidents using AI.")

    if data is not None:
        # Code for Home page
        st.subheader("📊 Data Preview")
        st.dataframe(data.head())

        if 'Injury.Severity' not in data.columns:
            st.error("❌ 'severity' column not found in the dataset.")
        else:
            st.success("✅ Data successfully loaded.")

            # Feature Selection
            st.subheader("🛠 Feature Engineering")
            with st.expander("Select Features to Train the Model"):
                columns_to_use = st.multiselect("Select input features", options=data.columns.tolist(), default=[col for col in data.columns if col != 'Injury.Severity'])
            df = data[columns_to_use + ['Injury.Severity']].copy()

            # Label Encoding
            label_encoders = {}
            for col in df.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

            # Split data
            X = df[columns_to_use]
            y = df['Injury.Severity']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Evaluate
            y_pred = rf.predict(X_test)
            score = rf.score(X_test, y_test)

            # Performance
            st.subheader("📊 Model Performance")
            st.success(f"Test Accuracy: {score*100:.2f}%")
            st.code(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Prediction Form
            st.subheader("🎯 Predict New Case")
            with st.form("prediction_form"):
                input_data = {}
                for col in columns_to_use:
                    if col in label_encoders:
                        options = label_encoders[col].classes_.tolist()
                        val = st.selectbox(f"{col}", options)
                        input_data[col] = label_encoders[col].transform([val])[0]
                    else:
                        input_data[col] = st.number_input(f"{col}", value=0.0)
                submitted = st.form_submit_button("Predict Severity")
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    prediction = rf.predict(input_df)[0]
                    st.success(f"✅ Predicted Severity: {prediction}")
    else:
        st.info("📥 Please upload a dataset to begin.")

elif nav == "Incident Analysis":
    if data is not None:
        st.subheader("Incident Analysis")

        # Incident severity distribution
        st.subheader("Incident Severity Distribution")
        col1, col2 = st.columns(2)
        with col1:
            if 'Injury.Severity' in data.columns:
                severity_counts = data['Injury.Severity'].value_counts(normalize=True) * 100
                fig = px.pie(severity_counts, 
                             values=severity_counts.values,
                             names=severity_counts.index,
                             color_discrete_sequence=px.colors.sequential.RdBu,
                             title="Incident Severity Distribution")
                fig.update_layout(
                    autosize=True,
                    margin=dict(l=20, r=20, t=100, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Injury.Severity column not found in the dataset")
        
        with col2:
            if 'Injury.Severity' in data.columns:
                damage_counts = data['Injury.Severity'].value_counts()
                fig = px.bar(damage_counts,
                             x=damage_counts.index,
                             y=damage_counts.values,
                             color=damage_counts.index,
                             color_discrete_sequence=px.colors.sequential.RdBu,
                             title="Aircraft Damage Level")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Injury.Severity column not found in the dataset")

        # Bar chart for crashes due to different factors
        st.subheader("Crashes due to Different Factors")
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # Flight damage
        flight_damage_data = [100, 200, 300]
        ax[0, 0].bar(['Minor', 'Major', 'Destroyed'], flight_damage_data, color='skyblue')
        ax[0, 0].set_title('Flight Damage')

        # Weather condition
        weather_condition_data = [150, 250, 100]
        ax[0, 1].bar(['Clear', 'Cloudy', 'Rainy'], weather_condition_data, color='skyblue')
        ax[0, 1].set_title('Weather Condition')

        # Plane category
        plane_category_data = [200, 150, 50]
        ax[1, 0].bar(['Commercial', 'Private', 'Military'], plane_category_data, color='skyblue')
        ax[1, 0].set_title('Plane Category')

        # Engine failure
        engine_failure_data = [300, 700]
        ax[1, 1].bar(['Yes', 'No'], engine_failure_data, color='skyblue')
        ax[1, 1].set_title('Engine Failure')

        # Layout so plots do not overlap
        fig.tight_layout()

        st.pyplot(fig)
    else:
        st.info("📥 Please upload a dataset to begin.")

elif nav == "About":
    st.title("About AirLytics")
    st.write("AirLytics is a web application designed to predict the severity of aircraft incidents using machine learning algorithms.")
    st.write("The application uses a Random Forest Classifier to predict the severity of incidents based on a set of input features.")
    st.write("The goal of AirLytics is to provide a useful tool for aviation professionals and researchers to analyze and predict the severity of aircraft incidents.")
    st.write("This application was built using Python, Scikit-learn, and Streamlit.")

# Footer
st.markdown("""---""")
st.caption(" Built with Python, Scikit-learn & Streamlit | By Aviation AI")
