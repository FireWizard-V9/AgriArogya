import requests
from PIL import Image
import io
import streamlit as st

def app():
    # FastAPI endpoint URL
    fastapi_url = "http://ec2-65-2-137-251.ap-south-1.compute.amazonaws.com:8000/predict"

    # Function to send the image to FastAPI and get the prediction
    def predict_image_class(image, url):
        # Convert the image to a byte stream
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')  # Save image as JPEG
        img_byte_arr.seek(0)  # Move to the beginning of the byte stream

        # Prepare the file payload
        files = {
            "file": ("image.jpg", img_byte_arr, "image/jpeg")
        }

        # Send the POST request to FastAPI
        try:
            response = requests.post(url, files=files)

            # Check if the request was successful
            if response.status_code == 200:
                prediction = response.json().get('prediction')
                return prediction
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
            return None

    # Streamlit App Layout
    st.title('Crop Disease Classifier')

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                with st.spinner('Classifying the image...'):
                    prediction = predict_image_class(image, fastapi_url)
                if prediction:
                    st.success(f'Prediction: {str(prediction)}')
                else:
                    st.error("Failed to get a valid prediction. Please try again.")

# Call the app function
if __name__ == "__main__":
    app()
