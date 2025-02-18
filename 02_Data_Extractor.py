import streamlit as st
from langchain_anthropic import ChatAnthropic
import json
import os
import base64
import PyPDF2
import re

# Initialize Claude
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
)

def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def extract_document_info(text):
    prompt = f"""
    You are a document analysis expert. Analyze this document and extract structured information.
    The document is in German or English - handle both languages.
    
    IMPORTANT: Your response must be ONLY valid JSON without any additional text or explanation.
    
    Extract these fields if present:
    - Invoice number (Rechnungsnummer)
    - Invoice date (Rechnungsdatum)
    - Order number (Bestellnummer)
    - Vendor name and address
    - Customer name and address
    - VAT ID (USt-IDNr)
    - Payment reference (Zahlungsreferenz)
    - Line items with prices
    - Tax amounts (USt/MwSt)
    - Total amount
    
    Respond with this exact JSON structure:
    {{
        "document_type": "Invoice",
        "extracted_fields": [
            {{
                "field_name": "Invoice Number",
                "original_label": "Rechnungsnummer",
                "value": "extracted value",
                "confidence": "high/medium/low"
            }}
        ],
        "amounts": {{
            "net_amount": "amount without tax",
            "tax_amount": "tax amount",
            "total_amount": "total amount",
            "currency": "EUR"
        }},
        "line_items": [
            {{
                "description": "item description",
                "quantity": "quantity",
                "unit_price_net": "price before tax",
                "total_price": "total price"
            }}
        ],
        "validation_warnings": []
    }}

    Analyze this text: {text}
    """
    
    try:
        response = llm.invoke(prompt)
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            st.error("No valid JSON found in response")
            return None
            
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        st.write("Raw response:", response.content)
        return None
    except Exception as e:
        st.error(f"Extraction error: {str(e)}")
        return None

def display_field_with_confidence(label, original_label, value, confidence):
    confidence_colors = {
        "high": "#00ff00",
        "medium": "#ffff00",
        "low": "#ff0000"
    }
    
    html = f"""
        <div style="margin-bottom: 1rem; padding: 8px; border-radius: 4px; background-color: #f0f2f6;">
            <span style="font-size: 0.9rem; color: #666;">{original_label}</span><br>
            <span style="font-weight: bold;">{label}:</span>
            <span style="margin-left: 5px;">{value}</span>
            <span style="
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: {confidence_colors[confidence.lower()]};
                margin-left: 5px;
                animation: blink 1s ease-in-out infinite;
            "></span>
        </div>
        <style>
        @keyframes blink {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
            100% {{ opacity: 1; }}
        }}
        </style>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_results(extraction):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Document Information")
        st.markdown(f"**Document Type:** {extraction['document_type']}")
        
        st.markdown("### Key Fields")
        for field in extraction['extracted_fields']:
            display_field_with_confidence(
                field['field_name'],
                field['original_label'],
                field['value'],
                field['confidence']
            )
        
        if 'amounts' in extraction:
            st.markdown("### Amounts")
            st.markdown(f"""
                * Net Amount: {extraction['amounts']['net_amount']}
                * Tax Amount: {extraction['amounts']['tax_amount']}
                * Total Amount: {extraction['amounts']['total_amount']}
                * Currency: {extraction['amounts']['currency']}
            """)
    
    with col2:
        if 'line_items' in extraction and extraction['line_items']:
            st.markdown("### Line Items")
            for item in extraction['line_items']:
                st.markdown(f"""
                * **{item['description']}**
                  - Quantity: {item['quantity']}
                  - Unit Price (Net): {item['unit_price_net']}
                  - Total Price: {item['total_price']}
                """)
        
        if 'validation_warnings' in extraction and extraction['validation_warnings']:
            st.markdown("### Validation Warnings")
            for warning in extraction['validation_warnings']:
                st.warning(warning)

def main():
    st.set_page_config(layout="wide", page_title="Document Information Extraction")
    
    st.title("Document Information Extraction")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            text = process_pdf(uploaded_file)
            if text:
                extraction = extract_document_info(text)
                if extraction:
                    display_results(extraction)

if __name__ == "__main__":
    main()