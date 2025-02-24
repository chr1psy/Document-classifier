import streamlit as st
from langchain_anthropic import ChatAnthropic
import pandas as pd
import PyPDF2
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Load API keys securely
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("❌ Anthropic API Key not found! Please add it to `secrets.toml` or set it as an environment variable.")
    st.stop()

# Paths & Configuration
DB_FILE = "vector_index.faiss"
CORRECTIONS_FILE = "corrections.json"
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS Vector Index Setup
vector_dim = 384  # MiniLM embedding dimension
if os.path.exists(DB_FILE):
    index = faiss.read_index(DB_FILE)
else:
    index = faiss.IndexFlatL2(vector_dim)

correction_data = {}
if os.path.exists(CORRECTIONS_FILE):
    with open(CORRECTIONS_FILE, "r") as f:
        correction_data = json.load(f)

# Categories
CATEGORIES = {
    "Finance & Accounting": "Invoices, tax returns, payroll, audit reports, financial statements, procurement, accounts payable, balance sheets",
    "Legal & Compliance": "Contracts, NDAs, compliance reports, regulatory filings, litigation, corporate governance, intellectual property",
    "HR": "Job applications, employment contracts, termination letters, salary structures, performance reviews, training materials",
    "Marketing & Sales": "Ad campaigns, market research, sponsorships, digital marketing reports, sales contracts, pricing strategies",
    "Operations & Manufacturing": "Production schedules, supply chain logistics, quality assurance reports, maintenance logs, warehouse inventory, process optimization, sustainability strategies",
    "Procurement & Supply Chain": "Purchase agreements, RFQ (request for quote), shipping invoices, supplier evaluations, customs documentation, vendor contracts",
    "IT & Cybersecurity": "Software licenses, security policies, DevOps strategies, network infrastructure, vulnerability assessments, API documentation, cloud computing",
    "Executive Office / Strategy": "Board meeting minutes, business continuity plans, investor reports, strategic planning, CEO communications",
    "Customer Service": "Support tickets, refund policies, customer complaints, service agreements, troubleshooting guides",
    "Facility Management": "Maintenance logs, lease agreements, safety regulations, security reports, emergency protocols, work orders",
    "CSR": "Sustainability reports, environmental impact assessments, carbon footprint reduction, ethical sourcing policies",
    "R&D": "Patent applications, feasibility studies, prototype testing, innovation roadmaps, new product development",
    "Spam / Fraud / Phishing": "Emails, fraudulent business proposals, fake job offers, scams, phishing attempts, malicious links",
    "General / Miscellaneous": "Company-wide announcements, newsletters, training manuals, travel policies, event invitations"
}

# Extract text from PDFs
def extract_text_from_pdf(file):
    """Extracts text from a PDF document"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text[:4000] if text.strip() else None
    except Exception as e:
        st.error(f"❌ PDF Processing Error: {str(e)}")
        return None

# Get similar past corrections using FAISS
def get_similar_past_correction(text):
    """Retrieves similar past corrections using FAISS vector search"""
    text_embedding = EMBEDDING_MODEL.encode([text])[0].astype(np.float32).reshape(1, -1)

    if index.ntotal > 0:
        _, I = index.search(text_embedding, 1)  # Retrieve 1 closest match
        matched_text = list(correction_data.keys())[I[0][0]]
        return correction_data.get(matched_text, None)
    return None

# AI Classification using Claude with Learning
def classify_document(text):
    """Uses Claude AI to classify a document, ensuring JSON response format with a formal summary."""
    try:
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=ANTHROPIC_API_KEY
        )

        past_correction = get_similar_past_correction(text)
        correction_context = f"Previous correction applied: {past_correction}" if past_correction else "No past corrections available."

        prompt = f"""
        You are an AI-powered document classifier for enterprise use. Your task is to analyze and classify documents into the most relevant business category.

        **Available Categories:**
        {json.dumps(CATEGORIES, indent=2)}

        **Classification Logic:**
        - If confidence is **below 85%**, return `"Unclear"` and request human review.
        - Extract **3-5 key phrases** that indicate the category.
        - If multiple categories apply, choose the **best fit** and list **alternative categories**.
        - If the document appears to be **fraudulent, spam, phishing, or a scam**, classify it under `"Spam / Fraud / Phishing"`.
        - Detect whether the document contains **Personally Identifiable Information (PII)**.
        - Perform **sentiment analysis** based on document tone.
        - Provide an **archival recommendation** with a retention duration.

        ## **📌 Learning from User Corrections**
        {correction_context}

        ---
        🔹 **Return a structured AI classification response in this strict JSON format:** 🔹

        {{
            "category": "Best Matching Category",
            "confidence": 0.92,
            "key_phrases": ["Phrase 1", "Phrase 2", "Phrase 3"],
            "alternative_categories": ["Alternative 1", "Alternative 2"],
            "explanation": "Detailed formal business explanation of why this category was chosen.",
            "contains_pii": "yes/no",
            "sentiment_analysis": "Positive/Neutral/Negative",
            "archival_recommendation": "Retention duration and reason"
        }}

        🔹 **Ensure the response is strictly valid JSON. Do NOT return human-readable text.** 🔹

        ---
        **Now classify this document:**
        {text}
        """

        response = llm.invoke(prompt)

        # Ensure response is valid before attempting JSON parsing
        if response is None or not response.content.strip():
            st.error("❌ Claude API returned an empty response. Check API Key or input formatting.")
            return None

        try:
            classification_data = json.loads(response.content)
            return classification_data
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {e}")
            return None

    except Exception as e:
        st.error(f"❌ Claude API Error: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="📂 AI Document Classifier", layout="wide")
    st.title("🤖 ClassifAI ")

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file:
        st.write("📂 **File uploaded successfully! Processing...**")

        with st.spinner("Extracting text from the document..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("❌ Failed to extract text. The document might be empty or unreadable.")
                return

        st.write("✅ **Text extracted successfully! Running classification...**")

        classification = classify_document(text)
        if not classification:
            st.error("❌ Classification failed! Check API response or model error.")
            return

        # Extract classification data
        category = classification.get("category", "Unknown")
        confidence = classification.get("confidence", 0) * 100  # Convert to percentage
        key_phrases = classification.get("key_phrases", [])
        explanation = classification.get("explanation", "No explanation provided.")
        alternative_categories = classification.get("alternative_categories", [])
        contains_pii = classification.get("contains_pii", "No")
        sentiment_analysis = classification.get("sentiment_analysis", "Neutral")
        archival_recommendation = classification.get("archival_recommendation", "No recommendation.")

        # Display classification results in a clean format
        st.write(f"### 📌 AI Classification: **{category}**")
        st.write(f"🎯 **Confidence:** {confidence:.2f}%")
        st.write(f"💡 **Key Phrases Identified:** {', '.join(key_phrases)}")
        st.write(f"📖 **AI Explanation:** {explanation}")
        
        if alternative_categories:
            st.write("🔄 **Alternative Categories Considered:**")
            for alt in alternative_categories:
                st.write(f"- {alt}")

        st.write(f"🔍 **Contains PII:** {contains_pii}")
        st.write(f"🧠 **Sentiment Analysis:** {sentiment_analysis}")
        st.write(f"📂 **Archival Recommendation:** {archival_recommendation}")

        # Confidence threshold check for human review
        if confidence < 85:
            st.warning("⚠️ **AI is uncertain! Human review recommended.**")

        # Category Correction Feature
        category_list = list(CATEGORIES.keys())  
        corrected_category = st.selectbox("🔧 **Select the correct category:**", category_list, index=category_list.index(category) if category in category_list else 0)

        if st.button("✅ Confirm & Train AI"):
            store_correction(text, corrected_category)
            st.success("📚 AI will now use this correction for future classifications.")

if __name__ == "__main__":
    main()
