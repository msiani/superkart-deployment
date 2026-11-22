import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="icon", layout="wide")

st.title("SuperKart Sales Prediction")
st.markdown("Predict sales revenue for products across SuperKart stores")
st.markdown("---")

# Backend URL Configuration
st.sidebar.header("Configuration")
BACKEND_URL = st.sidebar.text_input(
    "Backend API URL",
    value="http://backend:7860",
    help="Enter the Flask API URL"
)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Single Product Sales Prediction")
    st.markdown("Enter product and store details to predict sales")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Product Details")
        product_weight = st.number_input("Product Weight (kg)", min_value=0.0, max_value=50.0, value=12.66)
        product_sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_area = st.number_input("Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
        product_mrp = st.number_input("Product MRP (₹)", min_value=0.0, max_value=500.0, value=117.08)

    with col2:
        st.subheader("Store Details")
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", [
            "Supermarket Type1", "Supermarket Type2",
            "Departmental Store", "Food Mart"
        ])
        store_age = st.number_input("Store Age (Years)", min_value=1, max_value=50, value=16)

    with col3:
        st.subheader("Category Details")
        product_id_char = st.selectbox("Product ID Prefix", ["FD", "NC", "DR"])
        product_type_cat = st.selectbox("Product Type", ["Perishables", "Non Perishables"])

    st.markdown("---")

    if st.button("Predict Sales", type="primary", use_container_width=True):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_type_cat
        }

        with st.spinner("Predicting..."):
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"### Predicted Sales: ₹{result['predicted_sales']:,.2f}")
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Please check if the API is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with multiple products to predict sales in bulk")

    # Show expected format
    with st.expander("📋 View Expected CSV Format"):
        sample_df = pd.DataFrame({
            'Product_Weight': [12.66, 8.5],
            'Product_Sugar_Content': ['Low Sugar', 'Regular'],
            'Product_Allocated_Area': [0.027, 0.045],
            'Product_MRP': [117.08, 85.50],
            'Store_Size': ['Medium', 'High'],
            'Store_Location_City_Type': ['Tier 2', 'Tier 1'],
            'Store_Type': ['Supermarket Type2', 'Supermarket Type1'],
            'Product_Id_char': ['FD', 'NC'],
            'Store_Age_Years': [16, 22],
            'Product_Type_Category': ['Non Perishables', 'Perishables']
        })
        st.dataframe(sample_df)

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head(10))
        st.write(f"Total rows: {len(df)}")

        if st.button("Predict Batch", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(df)} predictions..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {'file': ('batch.csv', uploaded_file.getvalue(), 'text/csv')}
                    response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=120)

                    if response.status_code == 200:
                        predictions = response.json()
                        df['Predicted_Sales'] = [predictions[str(i)] for i in range(len(df))]

                        st.success(f"Successfully predicted sales for {len(df)} products!")
                        st.write("### Results:")
                        st.dataframe(df)

                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=" Download Results as CSV",
                            data=csv,
                            file_name="superkart_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Please check if the API is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*SuperKart Sales Prediction System | Powered by Machine Learning*")
