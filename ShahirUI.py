import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import io
import matplotlib.pyplot as plt
import sys
import joblib
from fpdf import FPDF



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

# Navigation Menu
# Function to encode an image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Menu function with logo included
def streamlit_menu():
    # Replace with the actual path to your logo
    logo_path = "C:/Users/shahi/money bazaar logo.png"

    # Convert logo to base64
    try:
        logo_base64 = get_base64_image(logo_path)
    except FileNotFoundError:
        st.error("Logo not found. Please check the path!")
        logo_base64 = ""

    # Navbar with logo
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: black; padding: 10px 20px; height: 80px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <div style="margin-right: 20px;">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 60px;"/>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Navbar with logo
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Single Prediction", "File Prediction", "History", "About"],  # required
        icons=["house", "search", "file-earmark-spreadsheet", "clock-history", "info-circle"],  # optional
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "background-color": "black",
                "display": "flex",
                "height": "50px",
                "max-width":"2200px",
                "border-radius":"0px",
                "padding": "0px 0px 0px 0px",
                "font-family": "Arial, sans-serif",
                "font-weight":"bolder",
                "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                "font-size": "40px",
                "letter-spacing": "3px",
                
            },
            "icon": {
                "color": "#ECF0F1",
                "font-size": "20px",
            },
            
            "nav-link": {
                "font-size": "20px",
                "font-family": "'Roboto', sans-serif",
                "text-align": "center",
                "color": "#ECF0F1",
                "margin": "0px",
                "width":"400px",
                "height":"50px",
                "padding-left": "0px",
                "padding-top":"10px",
                "padding-right": "0px",
                "text-transform": "uppercase",
                "transition": "color 0.3s, background-color 0.3s",
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 255, 255, 0.306)",
                "font-weight":"bold",
                "color": "rgb(75, 220, 62)",
                "border-radius": "0px",
            },
            "menu-icon": {
                "font-size": "22px",
                "padding-right": "10px",
            },
            "nav-link-selected icon": {
                "color": "rgb(75, 220, 62) !important",
            }
            
        },
    )

    

    return selected


# Call the menu function
selected = streamlit_menu()

# Display the selected option
st.write(f"You selected: {selected}")

def show_home_page():
    # Add custom styles
    st.markdown(
        """
        <style>
            .green-heading {
                color: green;
                font-family: 'Arial', sans-serif;
                font-size: 36px;
                font-weight: bold;
            }
            .black-text {
                color: black;
                font-family: 'Arial', sans-serif;
                font-size: 20px;
            }
            .section-title {
                margin-top: 40px;
            }
            .heading-line {
                display: inline-block;
                width: 150px;
                height: 8px;
                border-radius: 10px;
                background-color: rgb(47, 70, 46);
                margin-top: -10px;
                margin-bottom: 20px;
            }
            .image-container {
                display: flex;
                justify-content: center;
                margin-top: 0px;
                margin-bottom: 20px;
            }
            .image-container img {
                border-radius: 20px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
                width: 100%;
                max-width: 2200px;
                height: auto;
                object-fit: cover;
            }
            ul {
                font-size: 20px;
                color: black;
                margin-left: 20px;
            }
            li {
                font-size: 30px;
                margin-bottom: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Welcome message and header
    st.markdown('<h1 class="green-heading">Welcome to Money Bazaar</h1>', unsafe_allow_html=True)
    st.markdown('<div class="heading-line"></div>', unsafe_allow_html=True)

    # Image inclusion
    img_path = "home2.png"  # Replace with your image path
    try:
        image = Image.open(img_path)
    except FileNotFoundError:
        st.error(f"Image file not found at {img_path}. Please check the path!")
        return

    # Convert image to base64 for HTML embedding
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Display the image using HTML
    st.markdown(
        f"""
        <div class="image-container">
            <img src="data:image/png;base64,{img_str}" alt="Money Bazaar Overview">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Money Bazaar Overview
    st.markdown(
        '<p class="black-text"><span style="color: green; font-weight: bold;">Money </span><span style="color: black; font-weight: bold;">Bazaar</span> is a state-of-the-art online platform designed to safeguard payment systems from fraudulent activities. Leveraging advanced machine learning techniques, particularly the highly reliable Random Forest Classifier, Money Bazaar ensures secure and seamless financial transactions for individuals, businesses, and financial institutions.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="green-heading section-title">Why Choose Money Bazaar?</h2>', unsafe_allow_html=True)
    st.markdown('<div class="heading-line"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><span style="color: green; font-weight: bold;"><b>Cutting-Edge Machine Learning</b></span>: Utilizes advanced algorithms to deliver real-time fraud detection.</li>
            <li><span style="color: green; font-weight: bold;"><b>High Accuracy</b></span>: Our models are fine-tuned to minimize false positives while maximizing fraud detection rates.</li>
            <li><span style="color: green; font-weight: bold;"><b>Customizable Solutions</b></span>: Tailored fraud detection models to suit your business needs.</li>
            <li><span style="color: green; font-weight: bold;"><b>User-Friendly Interface</b></span>: Designed for ease of use, ensuring that even non-technical users can benefit from our platform.</li>
            <li><span style="color: green; font-weight: bold;"><b>Continuous Updates</b></span>: Our algorithms are continuously updated to stay ahead of evolving fraud tactics.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    # Features of Money Bazaar Section
    st.markdown('<h2 class="green-heading section-title">Features of Money Bazaar</h2>', unsafe_allow_html=True)
    st.markdown('<div class="heading-line"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul>
            <li><span style="color: green; font-weight: bold;"><b>Real-Time Fraud Detection</b></span>: Monitors transactions in real time to flag potential fraudulent activities instantly.</li>
            <li><span style="color: green; font-weight: bold;"><b>Detailed Analytics</b></span>: Gain insights into transaction patterns and fraud trends through interactive dashboards.</li>
            <li><span style="color: green; font-weight: bold;"><b>Scalable for Businesses</b></span>: Whether you’re a small business or a large financial institution, Money Bazaar adapts to your transaction volumes.</li>
            <li><span style="color: green; font-weight: bold;"><b>Secure and Reliable</b></span>: Ensures the safety and privacy of user data while providing dependable fraud detection.</li>
            <li><span style="color: green; font-weight: bold;"><b>Batch Transaction Analysis</b></span>: Upload bulk transaction data to identify fraud at scale, saving time and effort.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    # About Random Forest Model Section
    st.markdown('<h2 class="green-heading section-title">About Random Forest Model</h2>', unsafe_allow_html=True)
    st.markdown('<div class="heading-line"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="black-text">The <b>Random Forest Classifier</b> is a robust machine learning model that excels in detecting fraudulent transactions. Key advantages of this model include:</p>
        <ul>
            <li><b>Accuracy</b>: Combines predictions from multiple decision trees to reduce errors.</li>
            <li><b>Versatility</b>: Handles complex datasets with varying features and distributions.</li>
            <li><b>Feature Importance Analysis</b>: Identifies the most influential factors contributing to fraud detection.</li>
            <li><b>Scalability</b>: Performs well with large-scale data, making it ideal for payment platforms with high transaction volumes.</li>
        </ul>
        <p class="black-text">In fraud detection, the Random Forest model processes inputs such as transaction types, amounts, balance information, and user behavior to classify each transaction as fraudulent or legitimate. Its reliability and adaptability make it a trusted tool for modern payment systems.</p>
        """,
        unsafe_allow_html=True,
    )

    # Performance Metrics Bar Chart
    st.markdown('<h2 class="green-heading section-title">Model Performance Metrics</h2>', unsafe_allow_html=True)
    st.markdown('<div class="heading-line"></div>', unsafe_allow_html=True)

    # Example metrics (replace with your actual model's metrics)
    metrics = {
        "Accuracy": 0.997,
        "Precision": 0.998,
        "Recall": 0.997,
        "F1-Score": 0.997,
    }

    # Plotting the bar chart
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    fig, ax = plt.subplots(figsize=(1, 0.5))
    ax.bar(metrics_df['Metric'], metrics_df['Value'], color='green')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Metrics', fontsize=3)
    ax.set_ylabel('Scores', fontsize=3)
    ax.set_title('Performance Metrics of Random Forest Model', fontsize=4, color='green')
    ax.tick_params(axis='both', labelsize=3)

    # Display the bar chart
    st.pyplot(fig)

# Menu logic
if selected == "Home":
    show_home_page()

###############################################################
###############################################################

# Single Prediction Section
if selected == "Single Prediction":
    import streamlit as st
    import numpy as np
    import pandas as pd

    # Initialize the prediction history list if it doesn't exist yet
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Function to reset input values
    def reset_inputs():
        st.session_state["transaction_type"] = "Select"
        st.session_state["old_balance"] = 0.0
        st.session_state["amount"] = 0.0
        st.session_state["new_balance"] = 0.0

    # Large, bold headings aligned to the left in green color
    st.markdown(
        "<h2 style='color: #28a745; text-align: left; font-size: 40px; font-weight: bold;'>Single Transaction Prediction</h2>",
        unsafe_allow_html=True,
    )

    # Layout with one input per line
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h3 style='font-size: 20px; color: #28a745; margin-bottom: -50px;'>Transaction Type</h3>",
            unsafe_allow_html=True,
        )
        types = st.selectbox(
            "",
            ("Select", "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"),
            key="transaction_type",
            help="Select the transaction type",
        )

        st.markdown(
            "<h3 style='font-size: 20px; margin-bottom: -50px; color: #28a745;'>Old Balance Original</h3>",
            unsafe_allow_html=True,
        )
        oldbalanceOrg = st.number_input(
            "",
            min_value=0.0,
            format="%.2f",
            key="old_balance",
            help="Enter the old balance amount",
            step=0.01,
        )

    with col2:
        st.markdown(
            "<h3 style='font-size: 20px; color: #28a745; margin-bottom: -50px;'>Amount</h3>",
            unsafe_allow_html=True,
        )
        amount = st.number_input(
            "",
            min_value=0.0,
            format="%.2f",
            key="amount",
            help="Enter the transaction amount",
            step=0.01,
        )

        st.markdown(
            "<h3 style='font-size: 20px; color: #28a745; margin-bottom: -50px;'>New Balance Original</h3>",
            unsafe_allow_html=True,
        )
        newbalanceOrg = st.number_input(
            "",
            min_value=0.0,
            format="%.2f",
            key="new_balance",
            help="Enter the new balance amount",
            step=0.01,
        )

    # Add custom styles for buttons and prediction box
    st.markdown(
        """
        <style>
            .stButton>button {
                background-color: #28a745; /* Green background */
                color: white; /* White text */
                font-size: 18px; /* Font size */
                border-radius: 10px; /* Rounded corners */
                padding: 8px 16px; /* Padding */
                border: none; /* Remove border */
                cursor: pointer; /* Pointer cursor on hover */
                transition: all 0.3s ease; /* Smooth transition */
            }
            .stButton>button:hover {
                background-color: white; /* White background on hover */
                color: #28a745; /* Green text on hover */
                border: 2px solid #28a745; /* Green border on hover */
            }
            .prediction-box {
                padding: 15px;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                border-radius: 8px;
                margin-top: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .prediction-fraud {
                background-color: red; /* Red background for Fraudulent */
                color: white; /* White text */
            }
            .prediction-not-fraud {
                background-color: green; /* Green background for Not Fraudulent */
                color: white; /* White text */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Buttons with columns
    col3, col4 = st.columns(2)

    # Predict button functionality
    with col3:
        if st.button("Predict", key="predict_button", help="Click to predict fraud or not"):
            if types == "Select" or amount == 0.0 or oldbalanceOrg == 0.0 or newbalanceOrg == 0.0:
                st.markdown(
                    "<div class='prediction-box prediction-fraud'>Please fill all required fields.</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Prepare input data for prediction
                test = np.array([[types, amount, oldbalanceOrg, newbalanceOrg]])
                # Simulated prediction
                res = np.random.choice([0, 1])  # Replace with `model.predict`
                prediction = "Fraudulent" if res == 1 else "Not Fraudulent"

                # Display prediction in a box with dynamic color
                prediction_class = "prediction-fraud" if prediction == "Fraudulent" else "prediction-not-fraud"
                st.markdown(
                    f"<div class='prediction-box {prediction_class}'>Prediction: {prediction}</div>",
                    unsafe_allow_html=True,
                )

                # Save to history
                st.session_state.prediction_history.append(
                    {
                        "Transaction Type": types,
                        "Amount": amount,
                        "Old Balance": oldbalanceOrg,
                        "New Balance": newbalanceOrg,
                        "Prediction": prediction,
                    }
                )

    # Reset inputs functionality
    with col4:
        if st.button("Reset Inputs", key="reset_button", on_click=reset_inputs, help="Click to reset all inputs"):
            st.info("Inputs have been reset!")

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


#####################################################################
#####################################################################

# File Prediction Section
if selected == "File Prediction":
    # Page title
    st.markdown(
        "<h2 style='color: #28a745; text-align: left; font-size: 40px; font-weight: bold;'>Batch Transaction Prediction</h2>",
        unsafe_allow_html=True,
    )

    # Display accepted file types and required columns
    st.markdown(
        """
        <p style="font-size: 18px;">Accepted File Type: <b>CSV</b><br>
        Required Columns: <b>type</b>, <b>amount</b>, <b>oldbalanceOrg</b>, <b>newbalanceOrg</b></p>
        """,
        unsafe_allow_html=True,
    )

    # Show a sample format for the required columns
    st.markdown("<h4>Sample Format:</h4>", unsafe_allow_html=True)
    sample_data = pd.DataFrame({
        "type": ["CASH_OUT", "TRANSFER", "PAYMENT"],
        "amount": [2000.0, 5000.0, 1000.0],
        "oldbalanceOrg": [10000.0, 8000.0, 1500.0],
        "newbalanceOrg": [8000.0, 3000.0, 500.0],
    })
    st.dataframe(sample_data)

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

    if uploaded_file:
        try:
            # Load the uploaded CSV file
            data = pd.read_csv(uploaded_file)

            # Validate required columns
            required_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrg"]
            if not all(col in data.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in data.columns]
                st.markdown(
                    f"""
                    <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
                        <b>Error:</b> Missing required columns: {', '.join(missing_cols)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Show the uploaded file's first few rows
                st.markdown("<h4>Preview of Uploaded Data:</h4>", unsafe_allow_html=True)
                st.dataframe(data.head())

                # Prepare data for prediction
                data["type_encoded"] = data["type"].apply(find_type)  # Convert transaction types to numeric values
                input_features = data[["type_encoded", "amount", "oldbalanceOrg", "newbalanceOrg"]].values

                # Simulate predictions (replace with model.predict for actual implementation)
                predictions = np.random.choice([0, 1], size=len(data))  # Replace with `model.predict`
                data["Prediction"] = ["Fraudulent" if pred == 1 else "Not Fraudulent" for pred in predictions]

                # Apply conditional styling for predictions
                def color_predictions(val):
                    if val == "Fraudulent":
                        return "background-color: red; color: white;"
                    elif val == "Not Fraudulent":
                        return "background-color: green; color: white;"
                    return ""

                styled_data = data.style.applymap(color_predictions, subset=["Prediction"])

                # Display Results Table
                st.markdown("<h4>Prediction Results:</h4>", unsafe_allow_html=True)
                st.dataframe(styled_data, use_container_width=True)

                # Download Button for CSV
                csv_buffer = io.StringIO()
                data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_buffer.getvalue().encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    key="download_csv",
                )

                # Plot and Display Pie Chart
                st.markdown("<h4>Fraudulent vs. Non-Fraudulent Transactions:</h4>", unsafe_allow_html=True)
                counts = data["Prediction"].value_counts()
                colors = ["#FF9999", "#99FF99"]  # Fraudulent = red, Non-Fraudulent = green
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.pie(
                    counts,
                    labels=counts.index,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors,
                    textprops={"fontsize": 10},
                )
                ax.axis("equal")  # Equal aspect ratio for a circular pie chart
                st.pyplot(fig)

                # Download Button for Pie Chart
                pie_chart_buffer = io.BytesIO()
                fig.savefig(pie_chart_buffer, format="png", bbox_inches="tight", dpi=150)
                pie_chart_buffer.seek(0)
                st.download_button(
                    label="Download Pie Chart as PNG",
                    data=pie_chart_buffer,
                    file_name="prediction_pie_chart.png",
                    mime="image/png",
                    key="download_pie_chart",
                )

                # Update prediction history
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                for _, row in data.iterrows():
                    st.session_state.prediction_history.append(row.to_dict())

        except Exception as e:
            st.markdown(
                f"""
                <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
                    <b>Error:</b> {e}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Please upload a CSV file to proceed with batch prediction.")
##################################################################
##################################################################

# Display the History Page
if selected == "History":
    # Title
    st.markdown("<h2 style='color: #28a745;'>Prediction History</h2>", unsafe_allow_html=True)

    # Check if history exists
    if "prediction_history" not in st.session_state or len(st.session_state.prediction_history) == 0:
        st.warning("No prediction history available.")
    else:
        # Create a DataFrame from the prediction history
        history_df = pd.DataFrame(st.session_state.prediction_history)

        # Apply conditional styling for Fraudulent and Not Fraudulent
        def color_predictions(val):
            if val == "Fraudulent":
                return "background-color: red; color: white;"
            elif val == "Not Fraudulent":
                return "background-color: green; color: white;"
            return ""

        # Apply styles
        styled_df = history_df.style.applymap(
            color_predictions, subset=["Prediction"]
        )

        # Display the styled DataFrame
        st.dataframe(styled_df, use_container_width=True)

        # Add a Pie Chart for Fraudulent vs. Not Fraudulent Predictions
        st.markdown("<h3 style='color: #28a745;'>Prediction Summary Pie Chart</h3>", unsafe_allow_html=True)

        prediction_counts = history_df["Prediction"].value_counts()

        # Plotting the pie chart with reduced size and text font
        fig, ax = plt.subplots(figsize=(2, 1))  # Set smaller figure size
        ax.pie(
            prediction_counts,
            labels=prediction_counts.index,
            autopct=lambda p: f'{p:.1f}%',  # Format for percentage
            startangle=90,
            colors=["red", "green"],  # Colors for Fraudulent and Not Fraudulent
            textprops={'fontsize': 5},  # Reduce label font size
        )
        ax.axis("equal")  # Equal aspect ratio for a circular pie chart
        st.pyplot(fig)

        # Function to generate CSV for table
        # Function to generate CSV for table
        def generate_csv(dataframe):
            # Convert DataFrame to CSV and encode as binary
            csv_buffer = io.StringIO()
            dataframe.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")  # Encode as binary
            csv_buffer.close()
            return csv_bytes
        
        # Save the Pie Chart as an image
        def generate_pie_chart_image():
            chart_buffer = io.BytesIO()
            fig.savefig(chart_buffer, format="png", bbox_inches="tight", dpi=150)  # Adjust DPI for quality
            chart_buffer.seek(0)
            return chart_buffer

        # Download buttons
        col1, col2 = st.columns(2)  # Correctly unpack columns into two variables

        # Button to download the table as a CSV file
        with col1:
            csv_file = generate_csv(history_df)
            st.download_button(
                label="Download Table as CSV",
                data=csv_file,
                file_name="Prediction_History_Table.csv",
                mime="text/csv",
                key="csv_download",  # Unique key
            )

        # Button to download the pie chart as an image
        with col2:
            pie_chart_image = generate_pie_chart_image()
            st.download_button(
                label="Download Pie Chart as PNG",
                data=pie_chart_image,
                file_name="Prediction_History_Pie_Chart.png",
                mime="image/png",
                key="image_download",  # Unique key
            )

        # Add Clear History Button
        if st.button("Clear History", key="clear_history"):
            st.session_state.prediction_history.clear()
            st.success("Prediction history cleared!")

##################################################################
##################################################################

# About Section with Multiple Boxes and Feedback Form
if selected == "About":
    import base64

    # Helper function to encode local images
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # Paths to your local images
    project_image_path = "C:/Users/shahi/money bazaar logo.png"
    features_image_path = "C:/Users/shahi/feature1.png"
    tips_image_path = "C:/Users/shahi/tips.png"
    fraud_types_image_path = "C:/Users/shahi/types.png"
    awareness_image_path = "C:/Users/shahi/awareness.png"

    # Encode images
    project_image = get_base64_image(project_image_path)
    features_image = get_base64_image(features_image_path)
    tips_image = get_base64_image(tips_image_path)
    fraud_types_image = get_base64_image(fraud_types_image_path)
    awareness_image = get_base64_image(awareness_image_path)

    # Function to render a box
    def render_box(title, description, image_encoded):
        st.markdown(
            f"""
            <div style="display: flex; border: 2px solid #28a745; border-radius: 8px; background-color: #ffffff; margin-top: 10px; padding: 15px; align-items: center; font-size: 20px;">
                <!-- Left Content: Text Section -->
                <div style="flex: 70%; padding-right: 15px;">
                    <h3 style="color: #28a745; margin-top: 0;">{title}</h3>
                    <p style="font-size: 14px; line-height: 1.6; color: #000000;">{description}</p>
                </div>
                <!-- Right Content: Image Section -->
                <div style="flex: 30%; text-align: center; padding: 0px 0px 0px 0px;">
                    <img src="data:image/png;base64,{image_encoded}" alt="{title}" style="border-radius: 8px; max-width: 100%; height: auto; border: 1px solid #28a745;">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. About the Project
    render_box(
        "About the Project",
        "The Fraud Detection System is designed to safeguard online payment systems by leveraging the power of advanced Machine Learning techniques. With the exponential growth of digital transactions, protecting users from fraudulent activities has become a critical necessity. This project provides an innovative and reliable solution to detect and prevent such fraudulent transactions in real time."
        "By analyzing transaction patterns and identifying unusual behavior, the system helps to flag potentially fraudulent activities.",
        project_image,
    )

    # 2. Features of Fraud Detection
    render_box(
    "Features of Fraud Detection",
    """
    <p>
        <span style="color: green; font-weight: bold;">Money </span>
        <span style="color: black; font-weight: bold;">Bazaar</span>
        is a fraud detection system designed to provide robust security for online transactions, incorporating advanced capabilities that ensure both efficiency and reliability. Below are the key features:
    </p>

    1. **Real-Time Monitoring:** Continuously tracks transactions as they occur, enabling instant detection and prevention of fraudulent activities.
    2. **Customizable Alerts:** Configurable alert systems tailored to user-specific thresholds, ensuring relevant and timely notifications.
    3. **Detailed Analytics:** Comprehensive dashboards that provide insights into transaction patterns, enabling better decision-making.
    4. **Machine Learning Models:** Utilizes sophisticated algorithms such as Random Forest and Logistic Regression for precise classification of transactions.
    5. **Scalability:** Seamlessly handles both small-scale and large-scale datasets, making it suitable for businesses of all sizes.
    6. **Accuracy and Reliability:** High precision in detecting fraud while minimizing false positives, building trust with users.
    7. **Integration Flexibility:** Easily integrates with existing payment platforms and financial systems without disrupting operations.
    8. **Adaptive Learning:** Continuously updates and refines models based on new transaction data to stay ahead of emerging fraud tactics.


    This system not only ensures secure transactions but also builds confidence in digital platforms by significantly reducing financial risks.
    """,
    features_image,
)

    # 3. Tips for Staying Protected During Online Transactions
    render_box(
        "Tips for Staying Protected During Online Transactions",
        "Always use secure websites, enable two-factor authentication, avoid sharing personal information online, "
        "and regularly monitor your accounts for suspicious activity.",
        tips_image,
    )

    # 4. Common Types of Online Payment Fraud
    render_box(
        "Common Types of Online Payment Fraud",
        "Learn about common online frauds such as phishing, identity theft, and fake merchant websites to stay vigilant and protect yourself.",
        fraud_types_image,
    )

    # 5. Raising Awareness
    render_box(
        "Raising Awareness",
        "Educating individuals and businesses about online fraud and promoting secure practices can significantly reduce risks. "
        "This project aims to create awareness and provide tools for protection.",
        awareness_image,
    )

    # Feedback Form Section
    st.markdown(
        """
        <div style='text-align: center; background-color: #28a745; padding: 20px; border-radius: 8px; margin-top: 20px;'>
            <h3 style='color: white;'>We Value Your Feedback</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feedback Form
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your Name", placeholder="Enter your name here")
        email = st.text_input("Your Email", placeholder="Enter your email here")
        comments = st.text_area("Your Comments", placeholder="Share your feedback here")

        submitted = st.form_submit_button("Submit Feedback")

        # Feedback validation
        if submitted:
            if not name.strip():
                st.markdown(
                    """
                    <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
                        <b>Error:</b> Name is required.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif not email.strip() or "@" not in email or "." not in email:
                st.markdown(
                    """
                    <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
                        <b>Error:</b> Please enter a valid email address.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif not comments.strip():
                st.markdown(
                    """
                    <div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">
                        <b>Error:</b> Comments are required.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success("Thank you for your feedback!")

