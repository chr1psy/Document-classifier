import streamlit as st
from langchain_anthropic import ChatAnthropic
import pandas as pd
import PyPDF2
import os
from datetime import datetime

# Constants
LEARNING_DB = "learning_data.csv"
CATEGORIES = {
    "Finance": "Financial documents, invoices, budgets",
    "Legal": "Contracts, agreements, legal notices",
    "HR": "Employment, recruiting, personnel",
    "Marketing": "Promotional materials, marketing plans",
    "Operations": "Process documents, procedures",
    "IT": "Technical documentation, system specs",
    "General": "Miscellaneous business documents"
}

def process_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def classify_document(text):
    """Classify document using Claude"""
    try:
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
        
        prompt = f"""You are a document classification expert. Based on the following text, classify it into one of these categories:
        {', '.join(CATEGORIES.keys())}
        
        Provide your response as a Python dictionary with two keys:
        - 'category': The most appropriate category
        - 'confidence': A float between 0 and 1 indicating your confidence
        
        Text to classify:
        {text}
        """
        
        response = llm.invoke(prompt)
        result = eval(response.content)
        return result
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return None

def save_learning_data(text, predicted_category, confidence, actual_category):
    """Save classification data for learning"""
    df = pd.DataFrame({
        'text': [text],
        'predicted_category': [predicted_category],
        'confidence': [confidence],
        'user_feedback': [actual_category],
        'timestamp': [pd.Timestamp.now()]
    })
    
    if os.path.exists(LEARNING_DB):
        df.to_csv(LEARNING_DB, mode='a', header=False, index=False)
    else:
        df.to_csv(LEARNING_DB, index=False)

def store_document(file, category):
    """Store document in category folder"""
    try:
        folder = f"classified_docs/{category}"
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.name}"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            f.write(file.getvalue())
        return True
    except Exception as e:
        st.error(f"Error storing document: {str(e)}")
        return False

def main():
    st.set_page_config(layout="wide", page_title="Document Classification")
    st.write("# 📂 Document Classification with Learning Feature")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            try:
                text = process_pdf(uploaded_file)
                if text:
                    classification = classify_document(text)
                    
                    if classification:
                        category = classification["category"]
                        confidence = classification["confidence"]
                        
                        st.write("### Classification Results")
                        st.write(f"📑 Category: **{category}**")
                        st.write(f"🎯 Confidence: **{confidence:.2%}**")
                        
                        if confidence >= 0.85:
                            st.success("High confidence classification!")
                            store_document(uploaded_file, category)
                            save_learning_data(text, category, confidence, category)
                        else:
                            st.warning("Please verify the classification:")
                            correct_category = st.selectbox(
                                "Select correct category:",
                                list(CATEGORIES.keys()),
                                index=list(CATEGORIES.keys()).index(category)
                            )
                            
                            if st.button("Confirm Classification"):
                                store_document(uploaded_file, correct_category)
                                save_learning_data(text, category, confidence, correct_category)
                                st.success("Thank you! This feedback will help improve future classifications.")
                    else:
                        st.error("Error in classification")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
