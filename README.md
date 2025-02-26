#  MedAssistant – AI-Powered Medical Consultation  

**MedAssistant** is an AI-driven medical consultation tool that provides **diagnostic insights** based on **textual symptoms, medical history, and medical images**. By integrating **large language models (LLMs) for text-based medical analysis** and **vision models for image-based diagnosis**, MedAssistant aims to assist users in obtaining preliminary medical advice efficiently.  

---

##  Project Overview  

### 🔹 Problem Statement  
Training domain-specific medical AI models requires **significant computational resources** and **time**, making it impractical under constrained conditions. Additionally, most available **medical AI models lack API access**, making them difficult to integrate into real-time applications.  

### 🔹 Our Approach 
1️ **For Text-Based Medical Consultation**  
   - Explored models like **BioBERT, MedLLaMA-2, MedAlpaca, MedLLaVa-2, BioMistral etc**, but they lacked API support.  
   - Identified **[ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025)** from Hugging Face, which provides API-based inference.  
   - Used **zero-shot inference** with **optimized prompts and temperature settings** to generate concise medical consultations.  

2️ **For Image-Based Medical Consultation**  
   - No suitable **public medical image-to-text models** were available with API support.  
   - Utilized **Google’s Gemini 2.0 Flash** API for **image-based medical diagnosis**.  
   - Applied **zero-shot prompting** with controlled temperature settings to extract meaningful insights from medical images.  

3️ **Deployment**  
   - Implemented a **Streamlit-based user interface** for real-time AI consultation.  
   - Allowed users to **input text-based symptoms and/or upload medical images** for analysis.  

---

## Known Issues
Due to the use of **free API tiers**, the application may experience **occasional server errors** or **rate limits** while processing requests. If the app fails to generate a response, try:  
- **Refreshing the page** and re-submitting the request.  
- **Waiting a few minutes** before retrying (especially for Hugging Face API).  
- **Reducing request frequency** to avoid hitting rate limits.  

For **more stable performance**, consider using **paid API plans** for Hugging Face and Google AI services.  

---

##  Features 
 **AI-Powered Symptom-Based Consultation** – Enter symptoms and medical history to receive an AI-generated diagnosis.  
 **Medical Image Analysis** – Upload medical images (X-rays, MRIs, etc.) for AI-driven insights.  
 **Multi-Input Functionality** – Users can enter **only symptoms, only an image, or both** for enhanced results.  
 **API-Integrated Inference** – Uses **LLMs and vision models via API providers** for real-time processing.  
 **Fast & Lightweight Deployment** – Streamlit-based UI ensures easy accessibility and fast performance.  

---

##  Installation & Setup 

### **1️ Clone the Repository**  
```bash
git clone https://github.com/23Jaideep/Diagnostic-Assistant
cd MedAssistant
