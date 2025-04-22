import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import base64
import sys
import joblib


st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load model
model = pickle.load(open("random_forest_model (1).joblib", "rb"))

# Initialize session state variables
if 'reported_transactions' not in st.session_state:
    st.session_state.reported_transactions = []

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'expander_open' not in st.session_state:
    st.session_state.expander_open = False

# App Title
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #00698f; font-size: 48px;'>Online Payments Fraud Detection System</h1>
    </div>
    """, unsafe_allow_html=True)

# Navigation Menu
def streamlit_menu():
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Single-Predict", "File-Predict", "History", "About"],  # required
        icons=["house", "search", "file-earmark-spreadsheet", "clock-history", "info-circle"],  # optional
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal", 
        styles={
            "container": {
                "background-color": "#2C3E50",  
                "padding": "12px 30px", 
                "border-radius": "10px",  
                "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.2)",  
            },
            "icon": {
                "color": "#ECF0F1", 
                "font-size": "20px", 
            },
            "nav-link": {
                "font-size": "18px",  
                "font-family": "'Roboto', sans-serif", 
                "text-align": "center",
                "color": "#ECF0F1", 
                "margin": "0px",
                "padding-left": "20px",
                "padding-right": "20px",
                "text-transform": "uppercase", 
                "transition": "color 0.3s, background-color 0.3s", 
            },
            "nav-link-selected": {
                "background-color": "#3498DB", 
                "color": "white",
                "border-radius": "8px", 
                "font-weight": "bold",  
                "padding": "10px 25px", 
            },
            "menu-icon": {
                "font-size": "22px",  
                "padding-right": "20px", 
            }
        }
    )

    return selected

# Call the menu function
selected = streamlit_menu()

# Display the selected option
st.write(f"You selected: {selected}")

# Home Section
if selected == "Home":
    st.markdown("""
    <style>
        .home-page-container {
            background-image: url('https://blogs.mastechinfotrellis.com/hubfs/AI%20for%20Fraud%20Detection-Use%20Case.jpg'); /* Replace with your image URL */
            background-size: cover;
            background-position: center;
            height: 100vh;
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .blur-box {
            background: rgba(0, 0, 0, 0.5);  /* Black background with transparency for blur effect */
            backdrop-filter: blur(10px);  /* Apply blur effect */
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            width: 70%;
        }
                ul{
                font-family: 'Roboto', sans-serif;
                color: white;
                text-align: left;
                }

        .home-text {
                font-family: 'Roboto', sans-serif;
            color: white;
            font-size: 36px;
            margin-bottom: 20px;
        }

        .description-text {
            color: white;
            font-size: 18px;
        }

        .features-text {
            color: white;
            font-size: 18px;
            margin-top: 20px;
        }

        .cta-button {
            margin-top: 30px;
            padding: 12px 30px;
            background-color: #00698f;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        .cta-button:hover {
            background-color: #00577a;
        }
    </style>
    <div class="home-page-container">
        <div class="blur-box">
            <h2 class="home-text">Welcome to the Fraud Detection System</h2>
            <p class="description-text">
                This system uses advanced machine learning algorithms to detect fraudulent transactions. 
                Navigate through the options above to analyze and predict potential fraud cases.
                <b>The Key Features of the System:</b>
                <ul>
                    <li>Real-time fraud detection using state-of-the-art machine learning models.</li>
                    <li>Accurate classification of transactions as Fraud or Non-Fraud.</li>
                    <li>Batch prediction support for large datasets.</li>
                    <li>History tracking to monitor past transaction results.</li>
                </ul>
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)


# Single Prediction Section
if selected == "Single-Predict":
    st.markdown("<h2 style='color: #00698f; text-align: center;'>Single Transaction Prediction</h2>", unsafe_allow_html=True)

    def find_type(text):
        transaction_types = {
            "CASH_IN": 0,
            "CASH_OUT": 1,
            "DEBIT": 2,
            "PAYMENT": 3,
            "TRANSFER": 4
        }
        return transaction_types.get(text, -1)

    col1, col2 = st.columns(2)

    with col1:
        types = st.selectbox("Transaction Type", ("Select", "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"))
        oldbalanceOrg = st.number_input("Old Balance Original", min_value=0.0, format="%.2f")

    with col2:
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        newbalanceOrg = st.number_input("New Balance Original", min_value=0.0, format="%.2f")

    if st.button("Predict"):
        if types == "Select" or amount == 0.0:
            st.markdown("<p style='font-size:20px; color:red;'>Error: Please fill all required fields.</p>", unsafe_allow_html=True)
        else:
            types = find_type(types)
            test = np.array([[types, amount, oldbalanceOrg, newbalanceOrg]])
            res = model.predict(test)[0]
            prediction = "Fraudulent" if res == 1 else "Not Fraudulent"
            color = "red" if res == 1 else "green"
            st.markdown(f"<p style='font-size:24px; color:{color};'>Prediction: {prediction}</p>", unsafe_allow_html=True)

            # Add to history
            st.session_state.prediction_history.append({
                "Type": types,
                "Amount": amount,
                "Old Balance Origin": oldbalanceOrg,
                "New Balance Origin": newbalanceOrg,
                "Prediction": prediction
            })

# Define the 'find_type' function
def find_type(transaction_type):
    transaction_types = {
        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4
    }
    return transaction_types.get(transaction_type, -1)

# File Prediction Section
if selected == "File-Predict":
    st.markdown("<h2 style='color: #00698f; text-align: center;'>Batch Prediction</h2>", unsafe_allow_html=True)

    st.write("""
        Upload a CSV file with transaction details to predict fraud cases in bulk. 
        Ensure the file has the following columns: `type`, `amount`, `oldbalanceOrg`, `newbalanceOrg`.
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        # Check if required columns are present
        required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrg']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Error: The CSV file is missing required columns: {', '.join(required_columns)}")
        else:
            if st.button("Predict"):
                try:
                    # Map 'type' column to numeric values
                    df['type'] = df['type'].map(find_type)
                    
                    # Fill missing values
                    df['type'].fillna('UNKNOWN', inplace=True)
                    df[['amount', 'oldbalanceOrg', 'newbalanceOrg']].fillna(df.mean(), inplace=True)

                    # Make predictions
                    predictions = model.predict(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrg']].values)
                    df['isFraud'] = ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions]

                    # Display prediction results
                    st.write("Prediction Results:")
                    st.dataframe(df)

                    # Allow the user to download the predicted CSV file
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Predicted CSV", data=csv_data, file_name="predicted_transactions.csv")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.warning("Please check if the file has the correct format and required columns.")

# Prediction History Section
if selected == "History":
    st.markdown("<h2 style='color: #00698f; text-align: center;'>Prediction History</h2>", unsafe_allow_html=True)

    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)

        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download History CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No predictions made yet.")

# About Section
if selected == "About":
    st.markdown("""
        <div style='text-align: center; background-color: #e0f7fa; padding: 20px; border-radius: 8px;'>
            <h2 style='color: #00698f;'>About the Fraud Detection System</h2>
        </div>
        <div style='text-align: left; padding-left: 20px; background-color: #f1f8e9; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <h3 style='color: #004d40;'>About the Project</h3>
            <p>
                The goal of this project is to detect fraudulent online payment transactions using Machine Learning techniques. 
                By analyzing transaction patterns and identifying unusual behavior, the system helps to flag potentially fraudulent activities. 
                The project uses a dataset with over 1 million rows, with both fraudulent and non-fraudulent transactions, 
                and applies various models to classify each transaction.
            </p>
        </div>

        <div style='text-align: left; padding-left: 20px; background-color: #ffecb3; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <h3 style='color: #ff6f00;'>Model Performance</h3>
            <p>
                The performance of the Random Forest model was evaluated on a set of evaluation metrics including Accuracy, Precision, 
                Recall, and F1-Score. The model showed impressive results, achieving an accuracy of 99.7% in detecting fraudulent transactions.
            </p>
            <ul>
                <li><b>Accuracy:</b> 99.72%</li>
                <li><b>Precision:</b> 99.76%</li>
                <li><b>Recall:</b> 99.68%</li>
                <li><b>F1-Score:</b> 99.72%</li>
            </ul>
        </div>

        <div style='text-align: left; padding-left: 20px; background-color: #e1bee7; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <h3 style='color: #7b1fa2;'>About Random Forest Model</h3>
            <p>
                Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy 
                and reduce overfitting. The model works by constructing multiple decision trees during training and outputting 
                the majority vote or average prediction from these trees.
            </p>
            <p>
                It is a robust and widely used algorithm due to its ability to handle large datasets, manage missing values, 
                and provide feature importance. Random Forest works well for both classification and regression tasks, 
                making it ideal for fraud detection in online transactions.
            </p>
            <p>
                By using Random Forest, the system can effectively detect fraudulent transactions with high accuracy, 
                even in the presence of class imbalances or noisy data.
            </p>
        </div>
    """, unsafe_allow_html=True)


