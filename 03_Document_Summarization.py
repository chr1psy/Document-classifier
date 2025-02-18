import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from typing import List, Dict
import json

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

def extract_topics(text: str) -> List[str]:
    prompt = f"""
    Identify main topics from this document. Always use the language of the document in your response. Return as JSON array of strings. Try to format the response nicely.
    Text: {text[:3000]}
    """
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return []

def get_summary(text: str, style: str, max_words: int, focus: str) -> str:
    chunks = chunk_text(text)
    
    summary_prompts = {
        "executive": f"Create an executive summary in {max_words} words focusing on {focus}. Include key decisions and business impact.",
        "detailed": f"Create a detailed technical summary in {max_words} words focusing on {focus}. Include methodologies and specific data.",
        "bullet": f"Create a bullet-point summary with {max_words} words focusing on {focus}. Format as '• point' with clear hierarchy."
    }
    
    summaries = []
    for chunk in chunks:
        response = llm.invoke(summary_prompts[style] + f"\nText: {chunk}")
        summaries.append(response.content)
    
    final_prompt = f"Combine these summaries into a single coherent {style} summary:\n" + "\n".join(summaries)
    final_response = llm.invoke(final_prompt)
    return final_response.content

def main():
    st.title("Enhanced Document Summarization")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            total_pages = len(pdf_reader.pages)
            
            st.markdown("### Configuration")
            # Only show page range slider if document has more than 1 page
            if total_pages > 1:
                page_range = st.slider("Page Range", 1, total_pages, (1, total_pages))
            else:
                page_range = (1, 1)
                
            max_words = st.slider("Summary Length", 100, 1000, 250)
            
            col3, col4 = st.columns(2)
            with col3:
                style = st.selectbox(
                    "Summary Style",
                    ["executive", "detailed", "bullet"]
                )
            with col4:
                focus = st.selectbox(
                    "Focus Area",
                    ["business impact", "technical details", "action items", "key findings"]
                )
            
            if st.button("Generate Summary"):
                with st.spinner("Processing document..."):
                    text = extract_text(uploaded_file, page_range)
                    topics = extract_topics(text)
                    summary = get_summary(text, style, max_words, focus)
                    
                    # Display results in the right column
                    with col2:
                        #st.markdown("### Main Topics")
                        #st.write(topics)
                        
                        st.markdown("### Summary")
                        st.write(summary)

if __name__ == "__main__":
    main()