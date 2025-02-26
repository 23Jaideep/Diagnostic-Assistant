import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient
from PIL import Image

# Configure API Keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

# Streamlit UI
st.title("🩺 AI-Powered Medical Consultation")
st.warning("⚠️ **Disclaimer:** This AI-based consultation tool is for informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any medical concerns.")

st.write("Get medical advice based on your **symptoms (text)** and/or **medical images**.")

# User input for text-based consultation
symptoms = st.text_area("Describe your symptoms :")
medical_history = st.text_area("Provide your medical history :")

# File uploader for medical images
uploaded_file = st.file_uploader("Upload a medical image (X-ray, MRI, etc.) ", type=["jpg", "png", "jpeg"])

# Function for Image-Based Consultation (Gemini 2.0 Flash)
def analyze_medical_image(image):
    try:
        # Open and process image
        img = Image.open(image)

        # Generate response
        response = gemini_model.generate_content(
            [
                "You are an advanced medical AI. Analyze the provided medical image and deliver a **concise diagnosis** "
                "based on visible abnormalities.. Give a warning if you get any non medical image as input. Give the consultation with proper insights in 1 or 2 sentences, not any probabilities.",
                img
            ],
            generation_config={"temperature": 0.1}
        )

        return response.text

    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Submit button
if st.button("Get AI Consultation"):
    response_text = ""

    #  **1. Image-Based Consultation (Gemini)**
    image_diagnosis = ""
    if uploaded_file:
        image_diagnosis = analyze_medical_image(uploaded_file)
        response_text += f"### 🖼️ Image-Based Consultation:\n{image_diagnosis}\n\n"

    #  **2. Text-Based Consultation (Hugging Face)**
    if symptoms or medical_history:  # Now it can run even if symptoms are not given
        combined_input = ""

        if symptoms:
            combined_input += f"Symptoms: {symptoms}\n"
        if medical_history:
            combined_input += f"Medical history: {medical_history}\n"
        if image_diagnosis:
            combined_input += f"Image Diagnosis: {image_diagnosis}\n"

        hf_messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced medical AI assistant. Your task is to provide **brief, concise, and to-the-point** medical consultations.\n\n"
                    "- **Answer in 2-3 short sentences maximum**.\n"
                    "- **Avoid unnecessary details** and keep responses direct.\n"
                    "- **Use simple medical terms** that are understandable to general users.\n"
                    "- If the input is **not related to medical concerns**, **give a warning instead of an answer**.\n"
                    "- **Do not generate probabilities or uncertainties**—provide a **final, confident answer**.\n\n"
                    "If a non-medical query is detected, respond: \n"
                    '"I am designed for medical consultations only. Please provide symptoms or a health-related question."'
                ),
            },
            {
                "role": "user",
                "content": combined_input,
            }
        ]

        hf_response = hf_client.chat.completions.create(
            model="ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025",
            messages=hf_messages,
            max_tokens=150,
            temperature=0.1
        )

        response_text += f"### 📄 Final AI Consultation:\n{hf_response.choices[0].message.content}\n\n"

    # Display final consultation result
    if response_text:
        st.subheader("🩺 AI Consultation Report")
        st.write(response_text)
    else:
        st.warning("Please enter symptoms, medical history, or upload an image for consultation.")
