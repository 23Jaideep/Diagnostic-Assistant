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
                    "You are an expert AI medical assistant trained to analyze patient symptoms, medical history, and diagnostic imaging to provide **a confident and informative consultation**. Your task is to identify the **most probable diagnosis** based on the input and provide a **detailed, medically sound explanation**."
                    "- Your response should be **detailed yet concise**, around **150-200 words** or more if necessary."                    "- **Assess all provided information**, including symptoms, medical history, and diagnostic image insights (if available)."
                    "- **Give a confident diagnosis** based on the provided details and explain why this condition is the most probable."
                    "- If multiple conditions could be possible, state the **most likely one first**, followed by other potential considerations."
                    "- **Provide actionable insights**—such as recommended tests, treatment options, or next steps a patient should consider."
                    "- **Ensure your response is clear, professional, and medically accurate**, using terminology a general user can understand." 
                    "- **Avoid generic statements** like 'I cannot diagnose' or 'consult a doctor immediately' unless symptoms are highly urgent."
                    "- If symptoms are **life-threatening (e.g., severe chest pain, difficulty breathing, stroke symptoms)**, **urgently recommend** seeking immediate medical attention."
                    "- If the input is **not related to a medical condition**, respond with:"
                      "I specialize in medical consultations. Please provide symptoms or a health-related question."

                    "### Example Response Structure:" 
                    " **Diagnosis:** Likely condition with a confident explanation."
                    " **Reasoning:** Why this diagnosis fits the symptoms & medical history."  
                    " **Next Steps:** Recommended tests, medications, or when to seek a doctor." 

                    "Be clear, accurate, and direct in your answers."
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
            max_tokens=00,
            temperature=0.1
        )

        response_text += f"### 📄 Final AI Consultation:\n{hf_response.choices[0].message.content}\n\n"

    # Display final consultation result
    if response_text:
        st.subheader("🩺 AI Consultation Report")
        st.write(response_text)
    else:
        st.warning("Please enter symptoms, medical history, or upload an image for consultation.")
