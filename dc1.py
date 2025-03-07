import streamlit as st
from langchain_anthropic import ChatAnthropic
import PyPDF2
import os
import json

# Load API key securely
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("❌ Anthropic API Key not found! Please add it to secrets.toml or set it as an environment variable.")
    st.stop()

# Categories dictionary (for reference in the UI)
CATEGORIES = {
    "Finance & Accounting": "Invoices, tax returns, payroll, audit reports, financial statements, accounts payable, balance sheets",
    "Legal & Compliance": "Contracts, NDAs, compliance reports, regulatory filings, litigation, corporate governance, intellectual property",
    "HR": "Job applications, employment contracts, termination letters, salary structures, performance reviews, training materials",
    "Marketing & Sales": "Ad campaigns, market research, sponsorships, digital marketing reports, sales contracts, pricing strategies",
    "Operations & Manufacturing": "Production schedules, supply chain logistics, quality assurance reports, maintenance logs, warehouse inventory, process optimization",
    "Procurement & Supply Chain": "Purchase agreements, RFQ, shipping invoices, supplier evaluations, vendor contracts",
    "IT & Cybersecurity": "Software licenses, security policies, DevOps strategies, network infrastructure, vulnerability assessments, API documentation, cloud computing",
    "Executive Office / Strategy": "Board meeting minutes, business continuity plans, investor reports, strategic planning, CEO communications",
    "Customer Service": "Support tickets, refund policies, customer complaints, service agreements, troubleshooting guides",
    "Facility Management": "Maintenance logs, lease agreements, safety regulations, security reports, emergency protocols, work orders",
    "CSR": "Sustainability reports, environmental impact assessments, carbon footprint reduction, ethical sourcing policies",
    "R&D": "Patent applications, feasibility studies, prototype testing, innovation roadmaps, new product development",
    "Spam / Fraud / Phishing": "Emails, fraudulent business proposals, fake job offers, scams, phishing attempts, malicious links",
    "General / Miscellaneous": "Company-wide announcements, newsletters, training manuals, travel policies, event invitations",
    "Ambiguous": "Document has nearly equal signals across multiple categories; requires human review."
}

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file, returning up to 4000 characters.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text[:4000] if text.strip() else None
    except Exception as e:
        st.error(f"❌ PDF Processing Error: {str(e)}")
        return None

def classify_document(text):
    """
    Classifies the document text using ChatAnthropic with an extended prompt that includes
    few-shot examples for all categories and specific guidance for ambiguous cases.
    """
    try:
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",  # update model if needed
            anthropic_api_key=ANTHROPIC_API_KEY,
            max_tokens=3000,
            temperature=0.0
        )
        
        prompt = f"""
You are an AI-powered document classifier for a large multinational enterprise. Your role is to analyze and classify complex documents into the most appropriate business category. You must follow the steps below exactly and provide detailed reasoning along with alternative categories if uncertainty exists. If you are unsure (i.e. your confidence is low), indicate this clearly and request human review.

Step 1 – Preprocessing:
• Limit the text to the first 4000 characters.
• Normalize whitespace and punctuation (e.g., convert “Hello   world!” to "Hello world!"), change curly quotes to straight quotes, and standardize dashes.
• Correct common typos and OCR errors (e.g., "invioce" → "invoice", "1nvestment" → "investment", "Oﬀice" → "Office").
• Segment the text into logical parts (sentences, paragraphs, bullet points).
• Maintain case consistency: preserve proper nouns and generate a lowercase version for keyword matching.
• Remove irrelevant symbols, HTML tags, and formatting artifacts.
• Verify language consistency and flag or normalize foreign terms if found.

Step 2 – Keyword & Phrase Extraction:
• Identify significant keywords/phrases (domain-specific terms, proper nouns, acronyms, quantitative terms, modifiers).
• Use these domain keywords:
  – Finance: invoice, audit, budget, cash flow, profit margin, revenue, tax return, debt-to-equity, etc.
  – Legal: contract, NDA, compliance, arbitration, litigation, etc.
  – HR: employee, performance review, recruitment, training, career development, employee engagement, retention, etc.
  – Marketing: campaign, digital, conversion, SEO, customer engagement, ROI, brand image, etc.
  – Operations: production, quality control, maintenance, inventory, scheduling, warehouse, efficiency, etc.
  – Procurement: RFQ, vendor, procurement, supplier, contract, negotiation, etc.
  – IT: cybersecurity, software update, network, vulnerability, intrusion, etc.
  – Executive: strategic planning, board meeting, CEO, organizational restructuring, etc.
  – Customer Service: support ticket, refund, complaint, service agreement, resolution, etc.
  – Facility: maintenance, repair, safety, lease, emergency, etc.
  – CSR: sustainability, environmental, eco-friendly, ethical sourcing, etc.
  – R&D: research, prototype, innovation, feasibility, development, etc.
  – Spam: free, click, prize, scam, fraudulent, etc.
  – General: announcement, newsletter, policy, travel, event, etc.

Step 3 – Semantic Analysis:
• Extract the context (surrounding sentence/paragraph) for each keyword.
• Analyze co-occurrence and domain-specific language; consider modifiers (e.g., "final audited report" vs. "draft report") and note if keywords appear in key positions (e.g., headers, introductions).

Step 4 – Relevance Scoring:
• Count and weight keyword occurrences based on prominence and context.
• Normalize the scores (0–1 scale) for each category.
• Identify the primary category (highest score) and list alternatives if scores are within 10% of the highest.
• IMPORTANT:
  - If robust signals are present for a single category, assign a high confidence (at least 0.90) for that classification.
  - If overall signals are low or nearly equal across multiple categories, mark the classification as "Ambiguous" and assign a confidence ≥ 0.90.
  
Step 5 – Handling Ambiguity & Exceptions:
• If multiple categories have nearly equal scores or if explicit spam/phishing markers (e.g., "click here to claim your prize") are present, classify accordingly.
• If no single category clearly dominates, label it "Ambiguous" with confidence ≥ 0.90 and request human review.
• Include detailed reasoning and, if uncertain, explicitly state that human review is needed.

Step 6 – Additional Analyses:
• Detect any personally identifiable information (PII) such as names, addresses, or emails.
• Perform sentiment analysis (Positive, Neutral, or Negative).
• Identify any regulatory references that may affect document retention.
• Provide an archival recommendation based on department-specific guidelines.
• Annotate any additional context or nuances that influenced your decision.

Step 7 – Output:
Return your result strictly in JSON format with the following keys:
{{
  "category": "Best Matching Category",
  "confidence": <value between 0 and 1>,
  "key_phrases": [list of phrases],
  "alternative_categories": [list of alternatives],
  "explanation": "Detailed explanation with evidence and reasoning.",
  "contains_pii": "yes/no",
  "sentiment_analysis": "Positive/Neutral/Negative",
  "archival_recommendation": "Retention duration and rationale"
}}

Follow these steps precisely. If your analysis is ambiguous or your confidence is low, include that in your explanation and request human review.

### Few-Shot Examples

#### Ambiguous Example
**Title:** Cross-Departmental Synergy and Strategic Performance Analysis  
**Content:**  
This document references finance (profit margin, budget), HR (employee engagement, recruitment), IT (network security), marketing (campaign analytics), and operations (production scheduling) in nearly equal measure. No single domain dominates.  
**Correct Classification:** Ambiguous  
**Explanation:** Balanced signals across multiple categories; no clear primary focus. Must assign confidence ≥ 0.90 to reflect certainty of ambiguity.

#### Finance & Accounting
**Example 1: Mixed Content but Finance-Focused**  
Title: Integrated Strategic Digital Innovation Report  
**Content:**  
This document discusses our revenue growth, budget allocations, and cost-saving measures. Although it references executive strategy and digital transformation, the primary emphasis is on improving profit margins, streamlining financial reporting, and ensuring compliance with accounting standards. Financial KPIs like cash flow and debt-to-equity ratios are central to the analysis.  
**Correct Classification:** Finance & Accounting  
**Explanation:** Despite mentions of strategy, the robust financial data (cash flow, budget, profit margins, tax return) clearly indicate financial management.  
*Assign a confidence of at least 0.90.*

#### Legal & Compliance
**Example 1: Mixed Business and Legal Focus**  
Title: Contractual Risk and Compliance Evaluation Report  
**Content:**  
This document examines our contract negotiation processes, regulatory filings, and compliance measures alongside cost-saving initiatives and budget considerations. Despite mixed financial and operational references, detailed analysis of contract terms, NDA clauses, and arbitration history is predominant.  
**Correct Classification:** Legal & Compliance  
**Explanation:** The focus on contract language, compliance measures, and legal dispute resolution indicates a legal-centric approach.  
*Assign a confidence of at least 0.90.*

#### HR
**Example 1: Employee Engagement and Development Report**  
Title: Comprehensive Employee Engagement and Development Report  
**Content:**  
This document outlines enhanced employee training programs, performance review processes, and recruitment initiatives. Although it briefly mentions budget constraints and cost-saving measures, the emphasis is clearly on improving employee satisfaction and career development.  
**Correct Classification:** HR  
**Explanation:** Detailed focus on employee performance, training, and recruitment confirms HR as the central theme.  
*Assign a confidence of at least 0.90.*

**Example 2: Employee Wellbeing and Recruitment Strategy Brief**  
Title: Employee Wellbeing and Recruitment Strategy Brief  
**Content:**  
The report discusses initiatives aimed at boosting employee retention, training, and recruitment. Operational and financial considerations are only peripheral.  
**Correct Classification:** HR  
**Explanation:** The document is dedicated to HR policy enhancements and workforce development.  
*Assign a confidence of at least 0.90.*

**Example 3: Employee Engagement and Performance Enhancement Report**  
Title: Employee Engagement and Performance Enhancement Report  
**Content:**  
This report provides a comprehensive review of our current employee engagement strategies and performance review processes. It covers the design of enhanced training programs, talent development initiatives, and recruitment practices that aim to boost overall workforce productivity. While it mentions financial budgeting, legal compliance, and even a note on digital marketing for employer branding, these serve only to support the core discussion on employee satisfaction, performance metrics, and career development. IT systems are referenced solely in terms of optimizing HR data tracking.  
**Correct Classification:** HR  
**Explanation:** Although the document mentions other domains, the dominant narrative centers on HR functions—employee engagement, performance reviews, training, and recruitment.  
*Assign a confidence of at least 0.90.*

#### Marketing & Sales
**Example 1: Digital Marketing Campaign Performance Review**  
Title: Integrated Marketing Campaign Performance Review  
**Content:**  
This report outlines the performance of recent digital marketing campaigns, customer engagement analytics, and brand positioning strategies. Although it briefly references budget management, the focus is on conversion rates, customer segmentation, and ROI analysis.  
**Correct Classification:** Marketing & Sales  
**Explanation:** The marketing metrics and campaign analytics dominate the narrative.  
*Assign a confidence of at least 0.90.*

**Example 2: Branding and Ethical Sourcing Strategy**  
Title: Branding and Ethical Sourcing Strategy  
**Content:**  
The marketing team has launched a new campaign promoting ethical sourcing of raw materials. This aligns with the company's sustainability goals and CSR commitments. The campaign highlights efforts to reduce environmental impact while maintaining a strong brand image in the market.  
**Correct Classification:** Marketing & Sales  
**Explanation:** Although CSR is mentioned, the primary focus is on the marketing campaign and brand positioning.  
*Assign a confidence of at least 0.90.*

#### Operations & Manufacturing
**Example 1: Production Optimization and Quality Assurance Analysis**  
Title: Production Optimization and Quality Assurance Analysis  
**Content:**  
This report details improvements in production scheduling, inventory management, and quality control measures. It mentions cost-saving initiatives only in passing, while focusing on production throughput and maintenance logs.  
**Correct Classification:** Operations & Manufacturing  
**Explanation:** The emphasis on production efficiency and maintenance indicates an operations focus.  
*Assign a confidence of at least 0.90.*

**Example 2: Supply Chain Optimization Report (Operations Focus)**  
Title: Supply Chain Optimization Report  
**Content:**  
The operations team has proposed new logistics strategies to reduce delays and improve warehouse efficiency. Although it mentions negotiations with suppliers and reviewing procurement contracts, the primary focus is on streamlining operations and enhancing warehouse performance.  
**Correct Classification:** Operations & Manufacturing  
**Explanation:** While supplier negotiations are noted, the central narrative is operational improvement and warehouse efficiency.  
*Assign a confidence of at least 0.90.*

#### Procurement & Supply Chain
**Example 1: Global Supplier Evaluation and Procurement Strategy**  
Title: Global Supplier Evaluation and Procurement Strategy  
**Content:**  
The document discusses vendor evaluations, RFQs, and shipping invoice analysis alongside contract negotiations. The focus is on procurement processes and supplier performance metrics.  
**Correct Classification:** Procurement & Supply Chain  
**Explanation:** Emphasis on RFQs, vendor assessments, and procurement strategies supports a procurement classification.  
*Assign a confidence of at least 0.90.*

#### IT & Cybersecurity
**Example 1: Cybersecurity Infrastructure and Software Update Report**  
Title: Cybersecurity Infrastructure and Software Update Report  
**Content:**  
This report provides detailed analyses of network vulnerabilities, software patch updates, and cybersecurity risk assessments. Although budget constraints are mentioned, the focus is on IT security and system updates.  
**Correct Classification:** IT & Cybersecurity  
**Explanation:** Cybersecurity protocols and vulnerability assessments dominate the narrative.  
*Assign a confidence of at least 0.90.*

#### Executive Office / Strategy
**Example 1: Corporate Strategic Vision and Executive Restructuring Plan**  
Title: Corporate Strategic Vision and Executive Restructuring Plan  
**Content:**  
This memo outlines the CEO’s vision for organizational restructuring, new board appointments, and strategic market positioning. The focus is on high-level leadership and strategic direction with minimal detailed financial data.  
**Correct Classification:** Executive Office / Strategy  
**Explanation:** The emphasis on leadership changes and strategic goals indicates an executive focus.  
*Assign a confidence of at least 0.90.*

#### Customer Service
**Example 1: Customer Support Efficiency and Refund Policy Analysis**  
Title: Customer Support Efficiency and Refund Policy Analysis  
**Content:**  
The document reviews customer service performance metrics, support ticket resolution times, and refund policy effectiveness. The key focus is on service quality and customer satisfaction.  
**Correct Classification:** Customer Service  
**Explanation:** Support tickets and refund policies are central to customer service.  
*Assign a confidence of at least 0.90.*

#### Facility Management
**Example 1: Facility Safety and Maintenance Audit Report**  
Title: Facility Safety and Maintenance Audit Report  
**Content:**  
This report outlines maintenance logs, safety inspections, and emergency protocols of our facilities. The primary focus is on repair schedules, lease agreements, and safety regulations.  
**Correct Classification:** Facility Management  
**Explanation:** Maintenance, safety, and facility processes dominate the narrative.  
*Assign a confidence of at least 0.90.*

#### CSR
**Example 1: Corporate Sustainability and Environmental Impact Assessment**  
Title: Corporate Sustainability and Environmental Impact Assessment  
**Content:**  
The document discusses environmental impact metrics, sustainability initiatives, and ethical sourcing policies. The narrative primarily evaluates ecological footprint and CSR programs.  
**Correct Classification:** CSR  
**Explanation:** The emphasis on sustainability and environmental assessments clearly indicates CSR.  
*Assign a confidence of at least 0.90.*

#### R&D
**Example 1: New Product Innovation and Prototype Testing Report**  
Title: New Product Innovation and Prototype Testing Report  
**Content:**  
This report details research findings, prototype testing results, and innovation roadmaps for upcoming products. The primary emphasis is on R&D and product development.  
**Correct Classification:** R&D  
**Explanation:** Research, testing, and innovation are the key themes, indicating R&D.  
*Assign a confidence of at least 0.90.*

#### Spam / Fraud / Phishing
**Example 1: Urgent Prize Claim and Free Offer Notice**  
Title: Urgent Prize Claim and Free Offer Notice  
**Content:**  
This message contains phrases like "click here to claim your prize" and "free offer", with multiple indications of scam and fraudulent intent.  
**Correct Classification:** Spam / Fraud / Phishing  
**Explanation:** Spam indicators and scam language are clearly present.  
*Assign a confidence of at least 0.90.*

#### General / Miscellaneous
**Example 1: Company-wide Announcement and Travel Policy Update**  
Title: Company-wide Announcement and Travel Policy Update  
**Content:**  
This document includes travel policy updates, internal newsletters, and general company announcements. It lacks deep domain-specific details and is broadly applicable.  
**Correct Classification:** General / Miscellaneous  
**Explanation:** Generic content and broad scope indicate a miscellaneous classification.  
*Assign a confidence of at least 0.90.*

---
Document:
{text}
"""

        response = llm.invoke(prompt)
        if response is None or not response.content.strip():
            st.error("❌ Claude API returned an empty response.")
            return None
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {e}")
            return None

    except Exception as e:
        st.error(f"❌ Claude API Error: {e}")
        return None

def main():
    st.set_page_config(page_title="📂 AI Document Classifier", layout="wide")
    st.title("🤖 ClassifAI")

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        st.write("📂 File uploaded successfully!")
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("❌ Failed to extract text.")
                return
        st.write("✅ Text extracted. Running classification...")
        classification = classify_document(text)
        if not classification:
            st.error("❌ Classification failed!")
            return

        # Display results
        category = classification.get('category', 'Unknown')
        confidence = classification.get('confidence', 0)
        st.write(f"### 📌 Classification: **{category}**")
        st.write(f"🎯 Confidence: {confidence * 100:.2f}%")
        st.write(f"💡 Key Phrases: {', '.join(classification.get('key_phrases', []))}")
        st.write(f"📖 Explanation: {classification.get('explanation', 'No explanation provided.')}")
        if classification.get('alternative_categories'):
            st.write("🔄 Alternative Categories:")
            for alt in classification.get('alternative_categories'):
                st.write(f"- {alt}")
        st.write(f"🔍 Contains PII: {classification.get('contains_pii', 'No')}")
        st.write(f"🧠 Sentiment: {classification.get('sentiment_analysis', 'Neutral')}")
        st.write(f"📂 Archival Recommendation: {classification.get('archival_recommendation', 'No recommendation')}")

        # If category is "Ambiguous," or confidence is below threshold, request feedback
        if category == "Ambiguous":
            st.warning("⚠️ The document is flagged as ambiguous. Please provide your feedback.")
            corrected_category = st.selectbox("Select the correct category:", list(CATEGORIES.keys()))
            user_reasoning = st.text_area("Please explain your reasoning:")
            if st.button("Confirm & Train AI"):
                # In production, store this feedback for future retraining.
                st.success("📚 Correction submitted. Thank you for your feedback!")
        elif confidence < 0.85:
            st.warning("⚠️ AI is uncertain! Please provide your feedback with detailed reasoning.")
            corrected_category = st.selectbox("Select the correct category:", list(CATEGORIES.keys()))
            user_reasoning = st.text_area("Please explain your reasoning:")
            if st.button("Confirm & Train AI"):
                st.success("📚 Correction submitted. Thank you for your feedback!")

if __name__ == "__main__":
    main()
