import requests
import streamlit as st

# 1. --- Configuration --- 
API_URL = "https://immo-eliza-model-deployment.onrender.com"
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOIT = f"{API_URL}/"

st.set_page_config(page_title= "Immo Elisa - Price Predictor", page_icon="🏠",
                   layout="centered")

# 2. --- Header --- 
st.title("🏠 Immo Elisa - Price Predictor")
st.write(
    "Fill the property details below and get an instant price estimate " 
         "from our machine learning model"
         )

## Optional: ping the API health route so the user knows if the backend is awake
# (Render free-tier services spin down when idle and take ~ 30-60s to wake up)
with st.expander("API status"):
    if st.button("check API status"):
        try:
            r = requests.get(HEALTH_ENDPOIT, timeout=10)
            if r.ok:
                st.success(f"API is alive: {r.text}")
            else:
                st.warning(f"API responded with status {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")
    st.caption(
        "Render free-tier apps sleep when idle. The first request after a while" 
        "can take 30-60 seconds to wake it up"
    )
st.divider()

# 3. --- Input form ---  
PROPERTY_TYPES=["House", "Apartment"]

PROPERTY_SUBTYPES = ['Duplex', 'Flat', 'FlatStudio', 'House',
    'MixedBuilding', 'GroundFloor', 'Villa', 'Penthouse',
    'Other', 'MasterHouse', 'Chalet', 'Cottage',
    'Bungalow']

PROPERTY_CONDITIONS= ['Excellent', 'Fully renovated',
    'To be renovated', 'Normal', 'To renovate', 'Unknown', 
    'New', 'To restore', 'Under construction', 'To demolish']

PROVINCES = ['Brussels', 'Liège', 'Vlaams brabant', 'Namur', 
    'Luxembourg', 'Hainaut', 'Brabant wallon','Limburg', 
    'Oost vlaanderen','West vlaanderen', 'Antwerp']


AVAILABILITY = ['On contract', 'Immediately', 'Unknown', 'Negotiable', 'Future date']

with st.form("prediction_form"):
    st.subheader("Property details")
 
    col1, col2 = st.columns(2)
    with col1:
        property_type = st.selectbox("Property type", PROPERTY_TYPES)
        number_of_bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2, step=1)
        livable_surface = st.number_input("Livable surface (m²)", min_value=1, value=100, step=1)
        date_of_construction = st.number_input(
            "Year of construction", min_value=1800, max_value=2026, value=2000, step=1
        )
        province = st.selectbox("Province", PROVINCES)
    with col2:
        property_subtype = st.selectbox("Property subtype", PROPERTY_SUBTYPES)
        number_of_bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1, step=1)
        land_surface = st.number_input("Land surface (m²)", min_value=0, value=0, step=1)
        property_condition = st.selectbox("Condition", PROPERTY_CONDITIONS)
        availability = st.selectbox("Availability", AVAILABILITY)
 
    st.subheader("Location")
    col3, col4 = st.columns(2)
    with col3:
        latitude = st.number_input("Latitude", value=0.0, step=0.000001, format="%.6f")
    with col4:
        longitude = st.number_input("Longitude", value=0.0, step=0.000001, format="%.6f")
 
    st.subheader("Extra features")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        balcony = st.checkbox("Balcony")
    with col6:
        swimming_pool = st.checkbox("Swimming pool")
    with col7:
        elevator = st.checkbox("Elevator")
    with col8:
        furnished = st.checkbox("Furnished")
 
    submitted = st.form_submit_button("Predict price", use_container_width=True)
 
# 4. --- Build payload & call the API ---

# This dict mirrors the raw column names your model was trained on
# (see NUMERIC_COLS / BINARY_COLS / CATEGORICAL_COLS in train_model.py).
# If your FastAPI app.py Pydantic model uses different key names or wraps
# this in a "data": {...} envelope, adjust the `payload` shape here only.
 
if submitted:
    # Keys and types must match `propertyData` in app/main.py exactly.
    # Every field is required (no Optional[...] in that model), and the
    # numeric fields are typed as float there, so we cast explicitly.
    payload = {
        "longitude": float(longitude),
        "latitude": float(latitude),
        "property_type": property_type,
        "property_subtype": property_subtype,
        "date_of_construction": float(date_of_construction),
        "property_condition": property_condition,
        "livable_surface": float(livable_surface),
        "number_of_bedrooms": float(number_of_bedrooms),
        "number_of_bathrooms": float(number_of_bathrooms),
        "elevator": elevator,
        "furnished": furnished,
        "availability": availability,
        "province": province,
        "land_surface": float(land_surface),
        "balcony": balcony,
        "swimming_pool": swimming_pool,
    }
 
    with st.spinner("Contacting the model... (the free-tier API may take a moment to wake up)"):
        try:
            response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API: {e}")
            response = None
 
    if response is not None:
        if response.ok:
            try:
                result = response.json()
                prediction = result.get("predicted_price")
                if prediction is not None:
                    st.success(f"### Estimated price: € {prediction:,.0f}")
                else:
                    st.warning("API responded but no prediction was found in the payload:")
                    st.json(result)
            except ValueError:
                st.error("API returned a non-JSON response:")
                st.code(response.text)
        else:
            st.error(f"API returned an error (status {response.status_code}):")
            st.code(response.text)
 
    with st.expander("Request payload sent to the API"):
        st.json(payload)
 
st.divider()
st.caption("Immo Eliza · Streamlit frontend · calls the FastAPI backend deployed on Render")
