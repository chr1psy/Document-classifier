import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
import json
import os
import base64
import PyPDF2

# Initialize Claude
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
)

CATEGORIES = {
    "Finance & Accounting & Conrolling": "Invoices, credit note, tax notices, audit requests, payment reminders...",
    "Legal & Compliance": "Legal notices, compliance inquiries, regulatory updates...",
    "HR": "Job applications, employment verification requests...",
    "Marketing & Sales": "Advertising proposals, sponsorship requests...",
    "Operations/Manufacturing": "Production issue reports, equipment maintenance...",
    "Procurement/Supply Chain": "Vendor proposals, purchase orders...",
    "Research & Development": "Collaboration proposals, research funding...",
    "IT": "Technical support requests, software updates...",
    "CEO Office": "Executive-level correspondence, partnership proposals...",
    "Corporate Communication": "Media inquiries, press releases...",
    "Facility Management": "Repair requests, cleaning inquiries...",
    "CSR": "Sustainability initiatives, NGO partnership proposals...",
    "SPAM": "potential spam or fraud",
    "Unclear": "Any document that does not meet the above categories"
}

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return pdf_reader.pages[0].extract_text()[:2000]

def get_blinking_dot_html():
    return """
        <style>
        .dot {
            width: 12px;
            height: 12px;
            background-color: red;
            border-radius: 50%;
            display: inline-block;
            margin-left: 5px;
            position: relative;
            top: 2px;
            animation: blink 1s ease-in-out infinite;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }
        .metric-label {
            font-size: 1rem;
            color: #808495;
            font-weight: 500;
        }
        .metric-value {
            font-size: 1.1rem;
            font-weight: 600;
            margin-left: 5px;
        }
        </style>
    """

def display_metric_with_dot(label, value):
    html = f"""
        <div style="margin-bottom: 1rem;">
            <span class="metric-label">{label}:</span>
            <span class="metric-value">{value}</span>
            {' <span class="dot"></span>' if value.lower() == 'yes' else ''}
        </div>
    """
    st.markdown(get_blinking_dot_html() + html, unsafe_allow_html=True)

def classify_document(text):
    prompt = f"""
    Analyze this document excerpt and classify it into one of these categories:
    {json.dumps(CATEGORIES, indent=2)}
    
    Document text:
    {text}
    
    Respond in JSON format:
    {{
        "category": "category_name",
        "key_indicators": ["3-5 specific phrases indicating category"],
        "category_analysis": "explanation of categorization",
        "confidence": 0.95,
        "PII": "yes/no",
        "sentiment": "single word",
        "human": "yes/no",
        "archive": "archival recommendation with duration and reason",
        "archive_duration": "duration in years",
        "deletion_date": "calculated deletion date",
        "alternative_categories":"Alternative categories and why not selected"
    }}
    """
    response = llm.invoke(prompt)
    try:
        json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
        return json.loads(json_str)
    except:
        return None

def display_results(classification):
    col1, col2 = st.columns(2)
    
    with col1:
        display_metric_with_dot("Category", classification["category"])
        display_metric_with_dot("Confidence of Identification", f"{classification['confidence']*100:.1f}%")
        display_metric_with_dot("Contains Personally Identifiable Information", classification["PII"])
        display_metric_with_dot("Sentiment of the letter", classification["sentiment"])
        display_metric_with_dot("Human Review required", classification["human"])
        display_metric_with_dot("Archive Duration", f"{classification['archive_duration']} years")
        display_metric_with_dot("Deletion Date",  classification["deletion_date"])
    
    with col2:
        st.write("**Category Analysis:**", classification["category_analysis"])
        st.write("**Key Indicators:**", ", ".join(classification["key_indicators"]))
        st.write("**Archival Info:**", classification["archive"])
        st.write("**Alternative Categories:**", classification["alternative_categories"])
        

def main():
    st.set_page_config(layout="wide", page_title="Document Classification")
    
    st.title("Document Classification")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        st.subheader("Classification Results")
        with st.spinner("Analyzing document..."):
            text = process_pdf(uploaded_file)
            classification = classify_document(text)
            if classification:
                display_results(classification)
            else:
                st.error("Error processing document")

if __name__ == "__main__":
    main()