import streamlit as st
from langchain_anthropic import ChatAnthropic
import json
import PyPDF2
from datetime import datetime
import re

# Initialize Claude
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
)

RESPONSE_TYPES = {
    "Acceptance": "Positive response accepting the request/demand",
    "Partial Acceptance": "Accepting part of the request with conditions",
    "Rejection": "Polite rejection of the request/demand",
    "Request for Information": "Asking for additional details or clarification",
    "Acknowledgment": "Confirming receipt and promising future response",
    "Escalation": "Forwarding to higher authority/different department",
    "Counter Proposal": "Alternative suggestion to the original request"
}

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

def analyze_letter(text):
    prompt = f"""
    You are a business letter analyzer. Analyze this business letter and return ONLY a JSON object without any additional text.
    
    Letter text: {text}

    Extract and return this exact JSON structure with the information from the letter:
    {{
        "sender": {{
            "name": "sender's full name",
            "organization": "company name",
            "address": "full address",
            "contact": "phone and/or email"
        }},
        "recipient": {{
            "name": "recipient's name or department",
            "organization": "company name",
            "address": "full address"
        }},
        "letter_details": {{
            "date": "letter date",
            "subject": "letter subject",
            "reference_number": "any reference numbers"
        }},
        "content_analysis": {{
            "main_request": "primary request or demand",
            "key_points": ["list", "of", "key", "points"],
            "urgency_level": "high/medium/low",
            "deadline": "any mentioned deadline",
            "tone": "formal/neutral/urgent"
        }},
        "recommended_response_types": [
            "list of appropriate response types from: Acceptance, Partial Acceptance, Rejection, Request for Information, Acknowledgment, Escalation, Counter Proposal"
        ]
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Find JSON in the response using regex
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            st.error("No valid JSON found in response")
            st.write("Raw response:", response.content)
            return None
    except Exception as e:
        st.error(f"Error analyzing letter: {str(e)}")
        st.write("Raw response:", response.content)
        return None

# Previous imports and functions remain the same until generate_response

def clean_json_string(json_str):
    """Clean and normalize JSON string"""
    # Replace invalid escapes
    json_str = json_str.replace('\\"', '"')
    json_str = json_str.replace('\\', '')
    
    # Normalize line breaks
    json_str = json_str.replace('"body": "', '"body": "\\n')
    json_str = json_str.replace('."', '.\\n"')
    json_str = json_str.replace(',"', ',\\n"')
    json_str = json_str.replace('Dear ', '\\nDear ')
    json_str = json_str.replace('Sincerely,', '\\nSincerely,\\n')
    
    return json_str

def format_letter_text(response_data):
    """Format the letter text with proper structure"""
    # Company header
    header = """Nestle Deutschland AG
Lyoner Str 23
Frankfurt Am Main
60528 Germany"""

    # Recipient block (should be taken from analysis)
    recipient = """Dr. Michael Schmidt
ABC Tech Solutions GmbH
Frankfurter Allee 123
10247 Berlin
Germany"""

    # Current date
    current_date = datetime.now().strftime('%B %d, %Y')

    # Reference line
    reference = "Reference: Contract #2024-NST-456"

    # Clean up the body text
    body = response_data['body']
    body = body.replace('nn', '\n')  # Fix incorrect line breaks
    body = body.replace('  ', ' ')   # Fix double spaces
    
    # Signature block
    signature = """Dr. Maria Weber
Head of Procurement
Global Procurement
Nestle Deutschland AG"""

    # Combine all parts with proper spacing
    letter = f"""{header}

{current_date}

{recipient}

{reference}

Subject: {response_data['subject']}

{body}

{signature}"""

    return letter

def display_letter_format(response_data, analysis):
    # Prepare letter components
    header = """Nestle Deutschland AG
Lyoner Str 23
Frankfurt Am Main
60528 Germany"""

    date = datetime.now().strftime('%B %d, %Y')

    recipient = f"""{analysis['sender']['name']}
{analysis['sender']['organization']}
{analysis['sender']['address']}"""

    reference = f"Reference: {analysis['letter_details']['reference_number']}"

    subject = response_data['subject']

    body = response_data['body']
    body = body.replace('nn', '\n')
    body = body.replace('  ', ' ')

    signature = """Dr. Maria Weber
Head of Procurement
Global Procurement
Nestle Deutschland AG"""

    # Create copyable sections
    st.markdown("### 📋 Letter Components (click to copy)")
    
    with st.expander("Sender Details", expanded=True):
        st.code(header, language="text")
    
    with st.expander("Date", expanded=True):
        st.code(date, language="text")
    
    with st.expander("Recipient Details", expanded=True):
        st.code(recipient, language="text")
    
    with st.expander("Reference", expanded=True):
        st.code(reference, language="text")
    
    with st.expander("Subject", expanded=True):
        st.code(subject, language="text")
    
    with st.expander("Letter Body", expanded=True):
        st.code(body, language="text")
    
    with st.expander("Signature", expanded=True):
        st.code(signature, language="text")
    
    # Add a copy all button
    all_content = f"""{header}

{date}

{recipient}

{reference}

Subject: {subject}

{body}

{signature}"""
    
    st.markdown("---")
    st.download_button(
        label="💾 Download Complete Letter",
        data=all_content,
        file_name="response_letter.txt",
        mime="text/plain"
    )

# The rest of the code remains the same, just update the display_letter_format function

def generate_response(analysis, response_type, channel):
    company_details = {
        "name": "Nestle Deutschland AG",
        "address": "Lyoner Str 23\nFrankfurt Am Main\n60528 Germany",
        "signatory": {
            "name": "Dr. Maria Weber",
            "title": "Head of Procurement",
            "department": "Global Procurement"
        }
    }

    prompt = f"""Generate a {response_type} business letter response.

The letter MUST include these components in order:
1. Letter date: {datetime.now().strftime('%B %d, %Y')}
2. Recipient: 
   {analysis['sender']['name']}
   {analysis['sender']['organization']}
   {analysis['sender']['address']}
3. Reference line: {analysis['letter_details']['reference_number']}
4. Subject line
5. Salutation
6. Main content
7. Closing
8. Signature block:
   {company_details['signatory']['name']}
   {company_details['signatory']['title']}
   {company_details['signatory']['department']}

Return ONLY a JSON object exactly like this:
{{"subject": "your subject line", "body": "your complete letter content"}}

Use proper paragraph breaks and formatting."""

    try:
        response = llm.invoke(prompt)
        
        # Clean up the response
        text = response.content.strip()
        
        # Extract JSON part
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx]
            
            # Clean up the JSON string
            json_str = json_str.replace('\n', ' ')
            json_str = json_str.replace('\\n', '\n')
            json_str = json_str.replace('\\"', '"')
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {str(e)}")
                return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Rest of the code remains the same

def main():
    st.set_page_config(layout="wide", page_title="Business Letter Response Generator")
    
    st.title("Business Letter Response Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Letter Analysis")
        uploaded_file = st.file_uploader("Upload Letter", type="pdf")
        
        if uploaded_file:
            text = process_pdf(uploaded_file)
            if text and len(text.strip()) > 0:
                analysis = analyze_letter(text)
                if analysis:
                    st.subheader("Letter Details")
                    st.write(f"**Subject:** {analysis['letter_details']['subject']}")
                    st.write(f"**Date:** {analysis['letter_details']['date']}")
                    st.write(f"**Reference:** {analysis['letter_details']['reference_number']}")
                    
                    st.subheader("Main Request")
                    st.write(analysis['content_analysis']['main_request'])
                    
                    st.subheader("Key Points")
                    for point in analysis['content_analysis']['key_points']:
                        st.write(f"- {point}")
                    
                    st.subheader("Urgency Level")
                    st.write(f"**Level:** {analysis['content_analysis']['urgency_level']}")
                    if 'deadline' in analysis['content_analysis'] and analysis['content_analysis']['deadline']:
                        st.write(f"**Deadline:** {analysis['content_analysis']['deadline']}")
                    
                    with col2:
                        st.header("Response Generation")
                        response_type = st.selectbox(
                            "Select Response Type",
                            analysis['recommended_response_types']
                        )
                        
                        channel = st.radio(
                            "Select Response Channel",
                            ["Letter", "Email"]  # Changed default to Letter
                        )
                        
                        if st.button("Generate Response"):
                            st.markdown("---")
                            with st.spinner("Generating response..."):
                                response_data = generate_response(analysis, response_type, channel)
                                if response_data:
                                    st.header("Generated Response")
                                    if channel == "Email":
                                        display_email_format(response_data, analysis)
                                    else:
                                        display_letter_format(response_data, analysis)
                                        st.download_button(
                                            label="Download Letter",
                                            data=response_data['body'],
                                            file_name="response_letter.txt",
                                            mime="text/plain"
                                        )

if __name__ == "__main__":
    main()