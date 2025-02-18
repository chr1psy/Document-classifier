import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from typing import List, Dict
import json
import langdetect

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
)

def extract_text(uploaded_file, page_range=None):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    total_pages = len(pdf_reader.pages)
    
    if page_range:
        start, end = page_range
        pages = range(start-1, min(end, total_pages))
    else:
        pages = range(total_pages)
    
    return " ".join(pdf_reader.pages[i].extract_text() for i in pages)

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

def translate_text(text: str, source_lang: str) -> str:
    chunks = chunk_text(text)
    translated_chunks = []
    
    for chunk in chunks:
        prompt = f"""Translate this text from {source_lang} to German. 
        Maintain the original formatting and structure.
        
        Only provide the translation, no explanations:
        
        {chunk}
        """
        response = llm.invoke(prompt)
        print(response)
        translated_chunks.append(response.content)
    
    # Combine chunks if multiple
    if len(translated_chunks) > 1:
        combine_prompt = "Combine these translated segments into one coherent text formatted in markdown:\n" + "\n".join(translated_chunks)
        final_response = llm.invoke(combine_prompt)
        return final_response.content
    
    return translated_chunks[0]

def format_as_markdown(text: str) -> str:
    """Convert text to proper markdown format with preserved structure"""
    paragraphs = text.split('\n\n')
    markdown_text = ""
    
    for p in paragraphs:
        if p.strip():
            # Preserve any existing markdown formatting
            p = p.strip()
        if not p:
            continue

        # Detecting potential headings
        if p.upper() == p and len(p.split()) < 10:
            markdown_text += f"# {p}\n\n"
        # Detecting bullet points
        elif p.startswith(("* ", "- ", "+ ")):
            markdown_text += f"{p}\n"
        else:
            markdown_text += f"{p}\n\n"
    
    return markdown_text.strip()

def main():
    st.title("Document Translation")
    
    uploaded_file_trans = st.file_uploader("Upload PDF", type="pdf", key="translate_pdf")

    col1, col2 = st.columns([1, 1])
    if uploaded_file_trans:
        with st.spinner("Translating document..."):
            # Extract and process text
            text = extract_text(uploaded_file_trans)
            source_lang = detect_language(text)
            translated_text = translate_text(text, source_lang)
            
            # Format both original and translated text as markdown
            original_markdown = format_as_markdown(text)
            translated_markdown = format_as_markdown(translated_text)
            with col1:   
                # Display original text
                st.markdown("### Original Text")
                st.info(f"Detected language: {source_lang}")
                st.markdown(original_markdown)
                #print("Original Markdown Output:")
                #print(original_markdown)

            # Display translation
            with col2:
                st.markdown("### German Translation")
                st.text(translated_text)
                #print("Translated Markdown Output:")
                #print(translated_markdown)
if __name__ == "__main__":
    main()