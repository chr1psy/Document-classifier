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
    detailed instructions and few-shot examples. If the document is ambiguous, the AI must
    set the category to "Ambiguous" with a confidence below 0.85 (e.g., around 0.45-0.50).
    """
    try:
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",  # Update model if needed
            anthropic_api_key=ANTHROPIC_API_KEY,
            max_tokens=3000,
            temperature=0.0
        )
        
        prompt = f"""
You are an AI-powered document classifier for a large multinational enterprise. Your role is to analyze and classify complex documents into the most appropriate business category from the following list:
- Finance & Accounting
- Legal & Compliance
- HR
- Marketing & Sales
- Operations & Manufacturing
- Procurement & Supply Chain
- IT & Cybersecurity
- Executive Office / Strategy
- Customer Service
- Facility Management
- CSR
- R&D
- Spam / Fraud / Phishing
- General / Miscellaneous
- Ambiguous (if signals are balanced)

Follow these steps precisely:

1. **Preprocessing:**
   - Limit the text to the first 4000 characters.
   - Normalize whitespace and punctuation (e.g., convert “Hello   world!” to "Hello world!"), change curly quotes to straight quotes, and standardize dashes.
   - Correct common typos and OCR errors (e.g., "invioce" → "invoice", "1nvestment" → "investment", "Oﬀice" → "Office").
   - Segment the text into logical parts (sentences, paragraphs, bullet points).
   - Maintain case consistency: preserve proper nouns and generate a lowercase version for keyword matching.
   - Remove irrelevant symbols, HTML tags, and formatting artifacts.
   - Verify language consistency and flag or normalize foreign terms if found.

2. **Keyword & Phrase Extraction:**
   - Identify significant keywords/phrases (domain-specific terms, proper nouns, acronyms, quantitative terms, modifiers).
   - Use these domain keywords:
     - Finance: invoice, audit, budget, cash flow, profit margin, revenue, tax return, debt-to-equity, etc.
     - Legal: contract, NDA, compliance, arbitration, litigation, etc.
     - HR: employee, performance review, recruitment, training, career development, employee engagement, retention, etc.
     - Marketing: campaign, digital, conversion, SEO, customer engagement, ROI, brand image, etc.
     - Operations: production, quality control, maintenance, inventory, scheduling, warehouse, efficiency, etc.
     - Procurement: RFQ, vendor, procurement, supplier, contract, negotiation, etc.
     - IT: cybersecurity, software update, network, vulnerability, intrusion, etc.
     - Executive: strategic planning, board meeting, CEO, organizational restructuring, etc.
     - Customer Service: support ticket, refund, complaint, service agreement, resolution, etc.
     - Facility: maintenance, repair, safety, lease, emergency, etc.
     - CSR: sustainability, environmental, eco-friendly, ethical sourcing, etc.
     - R&D: research, prototype, innovation, feasibility, development, etc.
     - Spam: free, click, prize, scam, fraudulent, etc.
     - General: announcement, newsletter, policy, travel, event, etc.

3. **Semantic Analysis:**
   - Extract context (surrounding sentences/paragraphs) for each keyword.
   - Analyze co-occurrence and domain-specific language; consider modifiers (e.g., "final audited report" vs. "draft report") and note if keywords appear in key positions (e.g., headers, introductions).

4. **Relevance Scoring & Confidence:**
   - Count and weight keyword occurrences based on prominence and context.
   - Normalize the scores (0–1 scale) for each category.
   - Identify the primary category (highest score) and list alternatives if scores are within 10% of the highest.
   - **IMPORTANT:**
     - If robust signals are present for one domain, assign a high confidence (≥ 0.90).
     - If overall signals are low or nearly equal across domains, set the category to "Ambiguous" and explicitly assign a confidence below 0.85 (for example, around 0.45-0.50).
   - Provide a detailed explanation for your decision.

5. **Output Format:**
Return the result strictly in JSON with the following keys:
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

6. **Review:**
   - If the category is "Ambiguous" or if the confidence is ≤ 0.85, human review is required.

### Few-Shot Examples

#### Ambiguous Example 1 (Expected Confidence ~0.45)
Title: Corporate Policy Update and Disclaimer  
Content:
This document references Finance ("invoice", "audit"), Legal ("contract", "NDA"), and HR ("employee", "performance review") but mainly contains a generic boilerplate disclaimer repeated throughout.
Correct Classification: Ambiguous  
Confidence: 0.45  
Explanation: The signals are evenly distributed and the main content is generic.
*Alternative Categories:* ["Finance & Accounting", "Legal & Compliance", "HR"]

#### Ambiguous Example 2 (Expected Confidence ~0.50)
Title: Integrated Production Metrics and Vendor Analysis  
Content:
This report mixes production metrics (Operations), supplier details (Procurement), and a brief executive summary. The content is scattered with inconsistent signals.
Correct Classification: Ambiguous  
Confidence: 0.50  
Explanation: Mixed signals from multiple domains lead to no clear dominant category.
*Alternative Categories:* ["Operations & Manufacturing", "Procurement & Supply Chain", "Executive Office / Strategy"]

#### Irrelevant Keywords Example (Expected Confidence ~0.72)
Title: Technical Report on Network Security Trends  
Content:
The document includes sporadic IT and Marketing terms; however, the central discussion focuses on "network security," "cyber attack," and "malware."
Correct Classification: IT & Cybersecurity  
Confidence: 0.72  
Explanation: The technical content on network security outweighs irrelevant marketing keywords.
*Alternative Categories:* ["Marketing & Sales"]

#### Clear-Cut Finance Example (Expected Confidence ~0.95)
Title: Quarterly Audit & Revenue Report  
Content:
Focuses on profit margins, cash flow, tax returns, and compliance with accounting standards; minimal mention of other domains.
Correct Classification: Finance & Accounting  
Confidence: 0.95  
Explanation: Strong financial data dominate, clearly indicating the Finance domain.

#### Clear-Cut HR Example 1 (Expected Confidence ~0.95)
Title: Comprehensive Employee Engagement and Development Report  
Content:
Details enhanced employee training programs, performance reviews, and recruitment initiatives with only peripheral cost-saving mentions.
Correct Classification: HR  
Confidence: 0.95  
Explanation: Clear focus on HR functions confirms HR as the primary domain.

#### Clear-Cut HR Example 2 (Expected Confidence ~0.95)
Title: Employee Satisfaction Survey and Talent Development Report  
Content:
Analyzes employee satisfaction survey results and outlines talent development initiatives; references to financial budgeting exist only in relation to HR.
Correct Classification: HR  
Confidence: 0.95  
Explanation: Predominantly focused on HR metrics and talent development.

#### Clear-Cut HR Example 3 (Expected Confidence ~0.95)
Title: Employee Engagement and Performance Enhancement Report  
Content:
Reviews employee engagement strategies and performance reviews, covering training programs, talent development, and recruitment practices. Financial and legal details are mentioned only as supporting context.
Correct Classification: HR  
Confidence: 0.95  
Explanation: The dominant narrative centers on HR functions.

#### Marketing & Sales Example 1 (Expected Confidence ~0.95)
Title: Integrated Marketing Campaign Performance Review  
Content:
Outlines digital marketing campaigns, customer engagement analytics, and brand positioning; budget management is briefly mentioned.
Correct Classification: Marketing & Sales  
Confidence: 0.95  
Explanation: Marketing metrics dominate the narrative.

#### Marketing & Sales Example 2 (Expected Confidence ~0.95)
Title: Branding and Ethical Sourcing Strategy  
Content:
Describes a campaign promoting ethical sourcing of raw materials with an emphasis on brand image enhancement, despite CSR mentions.
Correct Classification: Marketing & Sales  
Confidence: 0.95  
Explanation: The primary focus on campaign strategy and brand positioning justifies a Marketing & Sales classification.

#### Operations & Manufacturing Example 1 (Expected Confidence ~0.95)
Title: Production Optimization and Quality Assurance Analysis  
Content:
Details improvements in production scheduling, inventory management, and quality control; cost-saving is only mentioned in passing.
Correct Classification: Operations & Manufacturing  
Confidence: 0.95  
Explanation: Emphasis on production efficiency and maintenance indicates an operations focus.

#### Operations & Manufacturing Example 2 (Expected Confidence ~0.95)
Title: Supply Chain Optimization Report  
Content:
Discusses new logistics strategies to reduce delays and improve warehouse efficiency; supplier negotiations are mentioned but remain secondary.
Correct Classification: Operations & Manufacturing  
Confidence: 0.95  
Explanation: Operational improvements and warehouse efficiency dominate the narrative.

#### Procurement & Supply Chain Example (Expected Confidence ~0.95)
Title: Global Supplier Evaluation and Procurement Strategy  
Content:
Discusses vendor evaluations, RFQs, shipping invoice analysis, and contract negotiations focused on procurement processes.
Correct Classification: Procurement & Supply Chain  
Confidence: 0.95  
Explanation: Emphasis on RFQs and vendor assessments clearly supports procurement.

#### IT & Cybersecurity Example (Expected Confidence ~0.95)
Title: Cybersecurity Infrastructure and Software Update Report  
Content:
Provides detailed analyses of network vulnerabilities, software patch updates, and cybersecurity risk assessments; budget constraints are secondary.
Correct Classification: IT & Cybersecurity  
Confidence: 0.95  
Explanation: Technical cybersecurity content clearly dominates.

#### Executive Office / Strategy Example (Expected Confidence ~0.95)
Title: Corporate Strategic Vision and Executive Restructuring Plan  
Content:
Outlines the CEO’s vision for organizational restructuring and strategic market positioning with minimal financial details.
Correct Classification: Executive Office / Strategy  
Confidence: 0.95  
Explanation: Emphasis on leadership and strategic goals indicates an executive focus.

#### Customer Service Example (Expected Confidence ~0.95)
Title: Customer Support Efficiency and Refund Policy Analysis  
Content:
Reviews customer service metrics, support ticket resolution times, and refund policies; the focus is on service quality.
Correct Classification: Customer Service  
Confidence: 0.95  
Explanation: Customer service indicators are dominant.

#### Facility Management Example (Expected Confidence ~0.95)
Title: Facility Safety and Maintenance Audit Report  
Content:
Outlines maintenance logs, safety inspections, and emergency protocols with a focus on repair schedules and safety regulations.
Correct Classification: Facility Management  
Confidence: 0.95  
Explanation: Facility-related processes dominate the narrative.

#### CSR Example (Expected Confidence ~0.95)
Title: Corporate Sustainability and Environmental Impact Assessment  
Content:
Discusses environmental impact metrics, sustainability initiatives, and ethical sourcing; the narrative focuses on CSR programs.
Correct Classification: CSR  
Confidence: 0.95  
Explanation: Sustainability and environmental focus indicate CSR.

#### R&D Example (Expected Confidence ~0.95)
Title: New Product Innovation and Prototype Testing Report  
Content:
Details research findings, prototype testing results, and innovation roadmaps for upcoming products; the emphasis is on product development.
Correct Classification: R&D  
Confidence: 0.95  
Explanation: Research and innovation are the dominant themes.

#### Spam / Fraud / Phishing Example (Expected Confidence ~0.95)
Title: Urgent Prize Claim and Free Offer Notice  
Content:
Contains phrases like "click here to claim your prize" and "free offer" with multiple scam indicators.
Correct Classification: Spam / Fraud / Phishing  
Confidence: 0.95  
Explanation: Scam language clearly indicates this category.


#### Additional Mixed-Domain Few-Shot Examples

1. **HR with Legal Terms Example:**
   Title: Employee Contract and Engagement Review  
   Content:
   Reviews employee engagement strategies and training programs with detailed analysis of employment contract clauses and legal compliance in HR policies. The primary focus is on employee development.
   Correct Classification: HR  
   Expected Confidence: 0.95  
   Explanation: Legal details are secondary to the core HR narrative.

2. **HR with Finance Terms Example:**
   Title: Employee Compensation and Budget Alignment Report  
   Content:
   Outlines employee compensation strategies and performance reviews alongside discussion of HR budget allocations and cost-saving measures. The emphasis is on HR metrics.
   Correct Classification: HR  
   Expected Confidence: 0.95  
   Explanation: Financial terms support HR functions rather than dominate them.

3. **Marketing with CSR Example:**
   Title: Digital Campaign and Sustainability Branding Analysis  
   Content:
   Analyzes digital marketing campaign performance and customer engagement while discussing CSR initiatives related to ethical sourcing. The dominant focus is on marketing.
   Correct Classification: Marketing & Sales  
   Expected Confidence: 0.95  
   Explanation: Marketing metrics dominate despite CSR references.

4. **Marketing with Legal Terms Example:**
   Title: Advertising Compliance and Brand Promotion Strategy  
   Content:
   Discusses digital advertising strategies and brand promotion alongside legal guidelines and contract compliance related to advertising standards. The dominant theme is marketing.
   Correct Classification: Marketing & Sales  
   Expected Confidence: 0.95  
   Explanation: Marketing elements outweigh legal details.

5. **Operations with IT Example:**
   Title: Production Workflow and IT Infrastructure Optimization  
   Content:
   Details production scheduling, workflow automation, and process optimization interwoven with IT system upgrade details. The central focus is on operational efficiency.
   Correct Classification: Operations & Manufacturing  
   Expected Confidence: 0.95  
   Explanation: IT details support operational improvements.

6. **Procurement with Legal Example:**
   Title: Supplier Contract Negotiations and Procurement Strategy Review  
   Content:
   Examines procurement processes and supplier evaluations with detailed analysis of contract terms and legal risks. The dominant theme is procurement.
   Correct Classification: Procurement & Supply Chain  
   Expected Confidence: 0.95  
   Explanation: Legal aspects reinforce procurement.

7. **IT with Marketing Example:**
   Title: Digital Security and Social Media Monitoring Report  
   Content:
   Discusses network security and cybersecurity measures alongside analysis of social media trends. The central theme is IT security.
   Correct Classification: IT & Cybersecurity  
   Expected Confidence: 0.95  
   Explanation: Marketing mentions are peripheral.

8. **CSR with Marketing Example:**
   Title: Sustainability Campaign and Brand Messaging Evaluation  
   Content:
   Evaluates CSR initiatives and sustainability metrics alongside digital marketing campaign performance and brand messaging. The primary focus is on CSR.
   Correct Classification: CSR  
   Expected Confidence: 0.95  
   Explanation: Despite marketing metrics, the emphasis is on sustainability and CSR.

9. **R&D with IT Example:**
   Title: Innovative Product Prototype and System Architecture Review  
   Content:
   Details the R&D process for a new product prototype with feasibility studies and prototype testing, along with an overview of system architecture. The focus is on R&D.
   Correct Classification: R&D  
   Expected Confidence: 0.95  
   Explanation: IT details support R&D; the primary focus is research and development.

10. **Executive with Finance Example:**
    Title: Strategic Leadership and Financial Oversight Report  
    Content:
    Discusses executive leadership changes and strategic planning alongside a review of financial oversight metrics. The dominant theme is executive strategy.
    Correct Classification: Executive Office / Strategy  
    Expected Confidence: 0.95  
    Explanation: Leadership and strategic focus indicate an executive classification.

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
        confidence = classification.get('confidence', 0.0)
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

        # Request feedback if category is "Ambiguous" or if confidence <= 0.85
        if category == "Ambiguous" or confidence <= 0.85:
            st.warning("⚠️ AI is uncertain or ambiguous! Please provide your feedback with detailed reasoning.")
            corrected_category = st.selectbox("Select the correct category:", list(CATEGORIES.keys()))
            user_reasoning = st.text_area("Please explain your reasoning:")
            if st.button("Confirm & Train AI"):
                # In production, store this feedback for future retraining.
                st.success("📚 Correction submitted. Thank you for your feedback!")

if __name__ == "__main__":
    main()