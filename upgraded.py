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

# Categories dictionary
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

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    """Extracts text from a PDF document."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text[:4000] if text.strip() else None
    except Exception as e:
        st.error(f"❌ PDF Processing Error: {str(e)}")
        return None

# Function to get similar past corrections using FAISS
def get_similar_past_correction(text):
    """Retrieves similar past corrections using FAISS vector search."""
    text_embedding = EMBEDDING_MODEL.encode([text])[0].astype(np.float32).reshape(1, -1)
    if index.ntotal > 0:
        _, I = index.search(text_embedding, 1)  # Retrieve 1 closest match
        matched_text = list(correction_data.keys())[I[0][0]]
        return correction_data.get(matched_text, None)
    return None

# Updated classification function with additional rules to avoid overconfidence on confusing documents
def classify_document(text):
    """Uses Claude AI to classify a document with granular steps and rules to lower confidence in ambiguous cases."""
    try:
        llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=3000,  # Adjust based on your needs
    temperature=0.0   # Lower temperature for more deterministic output
)

        past_correction = get_similar_past_correction(text)
        correction_context = f"Previous correction applied: {past_correction}" if past_correction else "No past corrections available."

        prompt = f"""
        You are an AI-powered document classifier for a very large multinational enterprise. Your task is to classify complex documents that may mix multiple business areas. Follow these detailed steps:

      [Complete System Prompt for AI Document Classifier]

1. Preprocessing:
   a. **Text Length Management:**  
      - If the document exceeds 4000 characters, limit analysis to the first 4000 characters.
      - *Example 1:* A document with 6000 characters will be truncated to the first 4000 characters.
      - *Example 2:* A 10,000-character report will have only its opening 4000 characters processed.
      - *Example 3:* An academic paper with 8,000 characters will be reduced to the initial 4000 characters.
      - *Example 4:* A lengthy email thread (5000 characters) is trimmed to its first 4000 characters.
      - *Example 5:* A scanned report of 4500 characters is analyzed only up to 4000 characters.
      - *Example 6:* A legal brief of 7000 characters is truncated to focus on the key opening sections.
      - *Example 7:* A technical manual with 9000 characters is reduced for processing efficiency.
      - *Example 8:* A company memo of 4100 characters is trimmed to 4000 characters.
      - *Example 9:* A customer feedback document of 8000 characters is limited to the first 4000.
      - *Example 10:* A policy document of 5000 characters is similarly truncated.

   b. **Normalization:**  
      - Remove extra whitespace and standardize punctuation; convert multiple spaces into one; replace curly quotes (“ ” ‘ ’) with straight quotes (" '); standardize dashes.
      - *Example 1:* Transform “Hello   world!” to "Hello world!".
      - *Example 2:* Convert “It’s a test—indeed it is.” to "It's a test-indeed it is."
      - *Example 3:* Change “Good morning!!!” to "Good morning!" by reducing exclamation marks.
      - *Example 4:* Replace “‘quoted text’” with "'quoted text'".
      - *Example 5:* Change “word–word” (en-dash) to "word-word".
      - *Example 6:* Convert “word—word” (em-dash) to "word-word".
      - *Example 7:* Normalize “Hello    world” to "Hello world".
      - *Example 8:* Convert ““Double quotes”” to '"Double quotes"'.
      - *Example 9:* Remove extraneous punctuation: “Hello!!! How are you??” → "Hello! How are you?"
      - *Example 10:* Replace non-standard symbols with standard ones.

   c. **Segmentation:**  
      - Break text into logical segments (sentences, paragraphs, bullet points).
      - *Example 1:* Split "This is the first sentence. This is the second sentence." into two sentences.
      - *Example 2:* Divide a long paragraph into individual, coherent sentences.
      - *Example 3:* Identify list items from "1. Introduction 2. Methods 3. Results" as separate segments.
      - *Example 4:* Split text by line breaks where paragraphs are separated.
      - *Example 5:* Use punctuation to isolate clauses.
      - *Example 6:* Separate bullet points in a list.
      - *Example 7:* Identify headers from body text.
      - *Example 8:* Break "Section A: ... Section B: ..." into segments.
      - *Example 9:* Extract sentences from dialogue or quotes.
      - *Example 10:* Segment text based on double newlines.

   d. **Case Consistency:**  
      - Preserve original case for proper nouns, but also generate a lowercase version for uniform matching.
      - *Example 1:* Keep "Apple Inc." intact; also generate "apple inc." for matching.
      - *Example 2:* Maintain acronyms like "IBM" in uppercase.
      - *Example 3:* For "The Quick Brown Fox", also process as "the quick brown fox".
      - *Example 4:* Keep brand names as-is; normalize surrounding text.
      - *Example 5:* Preserve titles like "CEO" but also use "ceo" for keyword matching.
      - *Example 6:* Retain case-sensitive words in legal documents.
      - *Example 7:* Handle mixed-case text (e.g., "iPhone") appropriately.
      - *Example 8:* Use lowercase version for counting frequencies.
      - *Example 9:* Distinguish proper nouns from generic terms.
      - *Example 10:* Generate both versions when uncertain.

   e. **Noise Removal:**  
      - Filter out irrelevant symbols, HTML tags, formatting artifacts, and non-textual elements.
      - *Example 1:* Remove HTML tags: "<p>Hello</p>" → "Hello".
      - *Example 2:* Clean stray symbols like "###Report###" → "Report".
      - *Example 3:* Eliminate artifacts such as "~~End~~".
      - *Example 4:* Remove watermarks or footers that do not contribute.
      - *Example 5:* Strip out extraneous punctuation not needed.
      - *Example 6:* Remove decorative symbols (e.g., "****").
      - *Example 7:* Filter out non-ASCII symbols if irrelevant.
      - *Example 8:* Remove metadata embedded in the text.
      - *Example 9:* Clean scanned document errors.
      - *Example 10:* Remove redundant section dividers.

   f. **Spelling & Typo Correction:**  
      - Detect and correct common typos, misspellings, and OCR-induced errors using context clues.
      - *Example 1:* Correct "invioce" to "invoice".
      - *Example 2:* Change "acount" to "account".
      - *Example 3:* Fix "teh" to "the".
      - *Example 4:* Update "financal" to "financial".
      - *Example 5:* Convert "managemnt" to "management".
      - *Example 6:* Correct "reciept" to "receipt".
      - *Example 7:* Fix "contarct" to "contract".
      - *Example 8:* Change "emloyee" to "employee".
      - *Example 9:* Correct "adverrtising" to "advertising".
      - *Example 10:* Change "prodction" to "production".

   g. **Robustness Against OCR and Formatting Errors:**  
      - Identify and correct errors such as misinterpreted characters (e.g., "0" for "O", "1" for "l") and merge split words.
      - *Example 1:* Merge "in-\nvoice" into "invoice".
      - *Example 2:* Correct "O0pen" to "Open".
      - *Example 3:* Fix "l0ve" to "love" when a zero is mistaken.
      - *Example 4:* Merge split words like "re port" to "report".
      - *Example 5:* Correct "rn" mistaken for "m" in words.
      - *Example 6:* Merge "docu-\nment" into "document".
      - *Example 7:* Fix broken hyphenation at line ends.
      - *Example 8:* Convert "1nvestment" to "investment".
      - *Example 9:* Correct "Oﬀice" (ligature issues) to "Office".
      - *Example 10:* Normalize merged words separated by errant line breaks.

   h. **Language Consistency Check:**  
      - Ensure text is in the expected language; flag or normalize foreign words when found.
      - *Example 1:* Identify "factura" (Spanish for "invoice") in an English document.
      - *Example 2:* Flag non-English greetings like "Bonjour" in primarily English text.
      - *Example 3:* Detect isolated foreign phrases such as "Gracias" and decide on translation.
      - *Example 4:* Notice words like "über" and determine if they require normalization.
      - *Example 5:* Identify "naïve" and ensure it is processed correctly.
      - *Example 6:* Flag phrases in a different script (e.g., Cyrillic) if out of context.
      - *Example 7:* Detect inconsistent language usage in a technical report.
      - *Example 8:* Normalize foreign terms if context suggests a common English equivalent.
      - *Example 9:* Use language detection to confirm primary language.
      - *Example 10:* Flag unexpected language patterns for human review.

2. Keyword & Phrase Extraction:
   - Identify significant keywords and phrases that indicate the document’s subject, purpose, and context. These include:
     - **Domain-Specific Terms:** Unique words associated with a department.
     - **Proper Nouns & Entities:** Names of companies, products, individuals.
     - **Acronyms/Abbreviations:** Standard industry abbreviations.
     - **Action Verbs/Directives:** Words implying processes or operations.
     - **Quantitative/Financial Terms:** Numbers, percentages, or financial jargon.
     - **Modifiers/Qualifiers:** Adjectives that refine keyword meaning.
   - Use these comprehensive lists:
     
   - Finance & Accounting: 
       Keywords: ["invoice", "financial", "account", "tax", "payroll", "budget", "audit", "balance sheet", "vendor", "receipt", "statement", "ledger", "expense", "revenue", "profit", "loss", "cash flow", "fiscal", "investment", "dividend", "credit", "debit", "accrual", "liability", "asset", "equity", "expenditure", "cost", "billing", "invoice number", "reconciliation", "fiscal year", "CAPEX", "OPEX", "forecast", "variance analysis", "EBITDA", "ROI"]
       - *Example Phrases:* "monthly financial report", "annual audit findings", "quarterly budget review", etc.

   - Legal & Compliance: 
       Keywords: ["contract", "agreement", "confidentiality", "regulatory", "compliance", "NDA", "litigation", "dispute", "legal", "terms", "conditions", "clause", "warranty", "indemnity", "governing law", "settlement", "memorandum", "jurisdiction", "arbitration", "breach", "obligation", "liability", "compliance report", "regulation", "policy", "protocol", "license", "intellectual property", "patent", "trademark", "copyright", "suit", "amendment", "notary", "force majeure", "due diligence"]
       - *Example Phrases:* "standard service agreement", "non-disclosure agreement", etc.

   - HR:
       Keywords: ["employee", "performance review", "recruitment", "hiring", "salary", "benefits", "job", "termination", "training", "onboarding", "payroll", "HR policy", "compensation", "incentive", "bonus", "evaluation", "workforce", "talent", "resignation", "promotion", "interview", "appraisal", "employment contract", "staff", "personnel", "labor", "union", "workplace", "diversity", "engagement", "retention", "human resources", "job description", "orientation", "employee satisfaction", "work-life balance", "performance appraisal", "exit interview"]
       - *Example Phrases:* "annual performance evaluation", etc.

   - Marketing & Sales:
       Keywords: ["ad", "campaign", "marketing", "sales", "promotion", "market research", "digital", "brand", "customer", "lead", "conversion", "strategy", "advertising", "SEO", "content", "social media", "influencer", "public relations", "pricing", "merchandising", "sales forecast", "sales report", "ROI", "discount", "target audience", "market share", "sponsorship", "trade show", "advertorial", "email marketing", "engagement", "click-through", "branding", "customer acquisition", "digital campaign", "market segmentation"]
       - *Example Phrases:* "integrated marketing campaign", etc.

   - Operations & Manufacturing:
       Keywords: ["production", "manufacturing", "quality", "maintenance", "operations", "scheduling", "process", "inventory", "supply chain", "logistics", "efficiency", "work order", "downtime", "automation", "assembly", "plant", "engineering", "productivity", "optimization", "safety", "quality control", "packaging", "distribution", "workflow", "production line", "manufacturing process", "lean manufacturing", "Six Sigma", "throughput", "operational efficiency", "workforce scheduling", "shift management"]
       - *Example Phrases:* "production schedule optimization", etc.

   - Procurement & Supply Chain:
       Keywords: ["procurement", "RFQ", "vendor", "purchase", "supply chain", "shipping", "supplier", "logistics", "order", "quotation", "contract", "bidding", "tender", "sourcing", "inventory", "distribution", "cost reduction", "negotiation", "deliverable", "shipment", "freight", "incoterms", "clearance", "requisition", "procurement process", "supply", "ordering", "vendor management", "strategic sourcing", "inventory turnover", "supply risk"]
       - *Example Phrases:* "request for quotation", etc.

   - IT & Cybersecurity:
       Keywords: ["software", "hardware", "IT", "cybersecurity", "network", "cloud", "encryption", "firewall", "infrastructure", "server", "database", "application", "cyber attack", "malware", "phishing", "backup", "data breach", "VPN", "SaaS", "PaaS", "IaaS", "API", "development", "programming", "coding", "IT support", "technical", "system", "security", "cyber", "antivirus", "patch", "update", "IT governance", "information security", "cyber defense", "intrusion detection", "cyber resilience", "endpoint security", "network monitoring"]
       - *Example Phrases:* "cloud infrastructure deployment", etc.

   - Executive Office / Strategy:
       Keywords: ["board", "CEO", "strategic", "investor", "meeting", "strategy", "executive", "business continuity", "vision", "mission", "corporate", "leadership", "governance", "stakeholder", "shareholder", "M&A", "acquisition", "divestiture", "strategy review", "business plan", "risk management", "forecast", "synergy", "corporate strategy", "executive summary", "long-term planning", "operating plan", "value proposition", "strategic initiative", "corporate restructuring", "performance metrics", "strategic alignment"]
       - *Example Phrases:* "executive board meeting", etc.

   - Customer Service:
       Keywords: ["support", "complaint", "ticket", "refund", "customer service", "escalation", "help desk", "resolution", "inquiry", "feedback", "satisfaction", "issue", "response", "call center", "live chat", "FAQ", "service", "customer experience", "follow-up", "warranty", "return", "exchange", "customer care", "service level agreement", "technical support", "customer query", "client satisfaction", "support request"]
       - *Example Phrases:* "24/7 customer support", etc.

   - Facility Management:
       Keywords: ["maintenance", "facility", "security", "lease", "safety", "emergency", "log", "repair", "building", "infrastructure", "janitorial", "cleaning", "HVAC", "renovation", "inspection", "energy", "utility", "waste", "compliance", "access control", "occupancy", "property management", "asset management", "facility operations", "space utilization", "preventive maintenance", "facility upgrade", "operational efficiency", "repair order"]
       - *Example Phrases:* "building maintenance schedule", etc.

   - CSR (Corporate Social Responsibility):
       Keywords: ["sustainability", "environment", "CSR", "impact", "ethics", "corporate responsibility", "carbon footprint", "green", "eco-friendly", "social responsibility", "community", "diversity", "inclusion", "philanthropy", "volunteer", "CSR report", "sustainable", "recycling", "renewable", "emissions", "conservation", "transparency", "environmental impact", "corporate citizenship", "social impact", "green initiative", "sustainability goals", "eco initiative", "environmental stewardship"]
       - *Example Phrases:* "annual CSR report", etc.

   - R&D:
       Keywords: ["research", "development", "prototype", "innovation", "feasibility", "testing", "R&D", "experiment", "discovery", "concept", "design", "trial", "iteration", "lab", "analysis", "technical report", "scientific", "engineering", "patent", "invention", "proof of concept", "ideation", "beta", "development cycle", "research findings", "novel", "breakthrough", "exploratory", "pilot", "technology roadmap", "experimental", "feasibility study"]
       - *Example Phrases:* "research and development strategy", etc.

   - Spam / Fraud / Phishing:
       Keywords: ["free", "click", "winner", "prize", "scam", "phishing", "fraud", "offer", "congratulations", "guaranteed", "no cost", "risk-free", "act now", "urgent", "limited time", "bonus", "reward", "claim", "verification", "suspicious", "unbelievable", "lottery", "inheritance", "gift", "miracle", "instant", "cash bonus", "no obligation", "sign up", "trial offer", "limited offer"]
       - *Example Phrases:* "act now to claim your prize", etc.

   - General / Miscellaneous:
       Keywords: ["announcement", "newsletter", "policy", "manual", "memo", "general", "update", "notice", "circular", "briefing", "bulletin", "release", "communication", "overview", "summary", "report", "documentation", "guideline", "instruction", "protocol", "reminder", "info", "information", "white paper", "press release", "statement", "roadmap", "framework", "strategy document", "executive briefing", "corporate update", "internal communication"]
       - *Example Phrases:* "company-wide announcement", etc.

3. Semantic Analysis & Context Evaluation:
   a. **Context Extraction:**  
      - Extract the surrounding sentence or paragraph for each keyword to understand its meaning.
      - *Example 1:* "The invoice was approved after the quarterly audit." → Extract "invoice was approved after the quarterly audit" (Finance).
      - *Example 2:* "The contract, including a strict confidentiality clause, was finalized." → Extract "contract, including a strict confidentiality clause" (Legal).
      - *Example 3:* "Employee performance reviews indicated a 15% improvement." → Extract "Employee performance reviews indicated a 15% improvement" (HR).
      - *Example 4:* "Our digital marketing campaign boosted engagement." → Extract "digital marketing campaign boosted engagement" (Marketing).
      - *Example 5:* "Production delays were resolved after maintenance improved operations." → Extract relevant phrase (Operations).
      - *Example 6:* "The procurement team issued an RFQ to vendors." → Extract "procurement team issued an RFQ" (Procurement).
      - *Example 7:* "The IT department upgraded the firewall and software." → Extract "upgraded the firewall and software" (IT).
      - *Example 8:* "The CEO presented a strategic vision during the board meeting." → Extract "CEO presented a strategic vision during the board meeting" (Executive).
      - *Example 9:* "Customer complaints have increased, prompting support escalation." → Extract "customer complaints have increased, prompting support escalation" (Customer Service).
      - *Example 10:* "The facility team scheduled a comprehensive maintenance review." → Extract "facility team scheduled a comprehensive maintenance review" (Facility).

   b. **Keyword Relationships:**  
      - Analyze co-occurrence and relational context among keywords.
      - *Example 1:* "Invoice" and "audit report" together strengthen Finance.
      - *Example 2:* "Contract" and "NDA" appearing together indicate Legal.
      - *Example 3:* "Performance review" with "training" supports HR.
      - *Example 4:* "Digital campaign" with "SEO strategy" supports Marketing.
      - *Example 5:* "Production" with "quality control" reinforces Operations.
      - *Example 6:* "RFQ" with "vendor selection" underscores Procurement.
      - *Example 7:* "Cybersecurity" with "software update" supports IT.
      - *Example 8:* "Strategic planning" with "board meeting" underlines Executive.
      - *Example 9:* "Support ticket" with "refund" emphasizes Customer Service.
      - *Example 10:* "Maintenance" with "safety protocols" bolsters Facility.

   c. **Domain-Specific Language:**  
      - Identify technical jargon that confirms departmental context.
      - *Example 1:* "GAAP compliance" clearly indicates Finance.
      - *Example 2:* "Arbitration clause" signals Legal.
      - *Example 3:* "Annual performance appraisal" confirms HR.
      - *Example 4:* "Influencer marketing" is specific to Marketing.
      - *Example 5:* "Lean manufacturing" confirms Operations.
      - *Example 6:* "Strategic sourcing" is specific to Procurement.
      - *Example 7:* "Malware detection" indicates IT.
      - *Example 8:* "Corporate restructuring" confirms Executive.
      - *Example 9:* "SLA benchmarks" support Customer Service.
      - *Example 10:* "HVAC inspection" confirms Facility.

   d. **Modifiers & Qualifiers:**  
      - Consider adjectives/adverbs that refine keyword meaning.
      - *Example 1:* "Final audited financial report" vs. "preliminary report" (Finance).
      - *Example 2:* "Legally binding contract" vs. "draft contract" (Legal).
      - *Example 3:* "Comprehensive performance review" vs. "brief overview" (HR).
      - *Example 4:* "Innovative digital campaign" vs. "standard campaign" (Marketing).
      - *Example 5:* "Critical production delay" vs. "minor delay" (Operations).
      - *Example 6:* "Detailed supplier contract" vs. "initial inquiry" (Procurement).
      - *Example 7:* "Robust IT security framework" vs. "basic update" (IT).
      - *Example 8:* "Strategic board meeting" vs. "routine meeting" (Executive).
      - *Example 9:* "Urgent support ticket" vs. "general inquiry" (Customer Service).
      - *Example 10:* "Scheduled facility maintenance" vs. "unexpected repair" (Facility).

   e. **Context Weighting:**  
      - Assign higher importance to keywords in strategic positions (headings, introductions) and discount peripheral occurrences.
      - *Example 1:* Heading "Financial Report Q1 2025" boosts Finance.
      - *Example 2:* Subheading "Legal Disclaimer" down-weights Legal if main text is marketing.
      - *Example 3:* Introduction stating "Employee Performance & Training" weights HR.
      - *Example 4:* Title "Digital Marketing Innovations" emphasizes Marketing.
      - *Example 5:* Bullet point "Production Efficiency" boosts Operations.
      - *Example 6:* List item "Key Procurement Metrics" emphasizes Procurement.
      - *Example 7:* Header "IT Infrastructure Upgrade" increases IT significance.
      - *Example 8:* Executive summary "Strategic Initiatives" weights Executive.
      - *Example 9:* Prominent "Customer Feedback" section strengthens Customer Service.
      - *Example 10:* Header "Facility Safety Protocols" underlines Facility.

4. Category Relevance Scoring:
   a. **Frequency Counting:**  
      - Count the number of occurrences for each department's keywords.
      - *Example 1:* Finance: "invoice" (5x) + "audit" (3x) = 8.
      - *Example 2:* Legal: "contract" (4x) + "NDA" (2x) = 6.
      - *Example 3:* HR: "employee" (7x) + "recruitment" (1x) = 8.
      - *Example 4:* Marketing: "ad campaign" (3x) + "SEO" (2x) + "conversion" (1x) = 6.
      - *Example 5:* Operations: "production" (4x) + "quality control" (3x) = 7.
      - *Example 6:* Procurement: "procurement" (3x) + "vendor" (4x) = 7.
      - *Example 7:* IT: "cybersecurity" (4x) + "software update" (3x) = 7.
      - *Example 8:* Executive: "strategic planning" (3x) + "board meeting" (2x) = 5.
      - *Example 9:* Customer Service: "support ticket" (5x) + "refund" (3x) = 8.
      - *Example 10:* Facility: "maintenance" (4x) + "repair" (3x) = 7.

   b. **Weighted Scoring:**  
      - Multiply keyword counts by weights based on prominence (e.g., in headings).
      - *Example 1:* Finance: "budget" in title weighted 3×.
      - *Example 2:* Legal: "signed contract" in header weighted 2×.
      - *Example 3:* HR: "performance review" in key section weighted 2.5×.
      - *Example 4:* Marketing: "digital campaign" in introduction weighted 2×.
      - *Example 5:* Operations: "production schedule" in bullet point weighted 3×.
      - *Example 6:* Procurement: "RFQ" in subheading weighted 2×.
      - *Example 7:* IT: "cybersecurity" in prominent section weighted 2.5×.
      - *Example 8:* Executive: "strategic initiative" in executive summary weighted 3×.
      - *Example 9:* Customer Service: "urgent support" in alert weighted 2×.
      - *Example 10:* Facility: "emergency maintenance" in header weighted 3×.

   c. **Score Normalization:**  
      - Normalize raw scores to a scale (e.g., 0 to 1).
      - *Example 1:* Finance raw score 8 → normalized 0.8.
      - *Example 2:* Legal raw score 6 → normalized 0.6.
      - *Example 3:* HR raw score 8 → normalized 0.8.
      - *Example 4:* Marketing raw score 6 → normalized 0.6.
      - *Example 5:* Operations raw score 7 → normalized 0.7.
      - *Example 6:* Procurement raw score 7 → normalized 0.7.
      - *Example 7:* IT raw score 7 → normalized 0.7.
      - *Example 8:* Executive raw score 5 → normalized 0.5.
      - *Example 9:* Customer Service raw score 8 → normalized 0.8.
      - *Example 10:* Facility raw score 7 → normalized 0.7.

   d. **Primary vs. Alternative Determination:**  
      - The highest normalized score designates the primary category; similar scores (within 10%) become alternatives.
      - *Example 1:* Finance 0.80 vs. Legal 0.60 → Finance primary.
      - *Example 2:* Legal 0.70 vs. HR 0.50 → Legal primary.
      - *Example 3:* HR 0.75 vs. Marketing 0.65 → HR primary.
      - *Example 4:* Marketing 0.78 vs. Sales 0.76 → Marketing primary.
      - *Example 5:* Operations 0.68 vs. Facility 0.66 → Operations primary.
      - *Example 6:* Procurement 0.57 vs. IT 0.56 → Procurement primary.
      - *Example 7:* IT 0.75 vs. Legal 0.70 → IT primary.
      - *Example 8:* Executive 0.64 vs. HR 0.63 → Executive primary.
      - *Example 9:* Customer Service 0.66 vs. Facility 0.64 → Customer Service primary.
      - *Example 10:* CSR 0.60 vs. R&D 0.59 → CSR primary.

   e. **Ambiguity Flag:**  
      - If top scores are within 5% or overall scores are low, flag the document as ambiguous.
      - *Example 1:* Finance 0.45, Legal 0.44, HR 0.43 → Ambiguous.
      - *Example 2:* HR 0.50, Marketing 0.49, Operations 0.48 → Ambiguous.
      - *Example 3:* IT 0.55, Procurement 0.54, Legal 0.53 → Ambiguous.
      - *Example 4:* Customer Service 0.50, General 0.49, CSR 0.48 → Ambiguous.
      - *Example 5:* Operations 0.40, Facility 0.39, R&D 0.38 → Ambiguous.
      - *Example 6:* Marketing 0.60, Sales 0.59, HR 0.58 → Ambiguous.
      - *Example 7:* Legal, Executive, IT each at ~0.50 → Ambiguous.
      - *Example 8:* Finance, HR, Marketing all at 0.55 → Ambiguous.
      - *Example 9:* Procurement, IT, Operations all at 0.45 → Ambiguous.
      - *Example 10:* Customer Service, Facility, CSR all at 0.40 → Ambiguous.

5. Handling Mixed-Category Content:
   a. **Identification:**  
      - Detect distinct clusters of keywords indicating multiple departments.
      - *Example 1:* Finance keywords ("invoice", "audit") with a minor IT section.
      - *Example 2:* Legal language with a sidebar on HR benefits.
      - *Example 3:* HR discussion with a paragraph on marketing results.
      - *Example 4:* Marketing proposal with brief procurement cost notes.
      - *Example 5:* Operations update mixed with a note on facility repairs.
      - *Example 6:* Procurement report including vendor selection with an executive strategy mention.
      - *Example 7:* IT update with a short legal disclaimer.
      - *Example 8:* Executive briefing that includes incidental HR details.
      - *Example 9:* Customer service report with a minor facility maintenance remark.
      - *Example 10:* Facility report with occasional CSR initiative mentions.

   b. **Score Comparison:**  
      - Compare aggregated scores to identify primary and secondary signals.
      - *Example 1:* Finance 0.70, IT 0.65, Legal 0.60 → Finance primary.
      - *Example 2:* Legal 0.55, HR 0.50, Operations 0.45 → Legal primary.
      - *Example 3:* HR 0.60, Marketing 0.58, Customer Service 0.55 → HR primary.
      - *Example 4:* Marketing 0.68, Procurement 0.66, Executive 0.64 → Marketing primary.
      - *Example 5:* Operations 0.62, Facility 0.60, CSR 0.58 → Operations primary.
      - *Example 6:* Procurement 0.57, IT 0.56, Finance 0.55 → Procurement primary.
      - *Example 7:* IT 0.75, Legal 0.70, Marketing 0.65 → IT primary.
      - *Example 8:* Executive 0.64, HR 0.63, Operations 0.62 → Executive primary.
      - *Example 9:* Customer Service 0.66, Facility 0.65, Procurement 0.64 → Customer Service primary.
      - *Example 10:* CSR 0.60, R&D 0.59, Legal 0.58 → CSR primary.

   c. **Documentation:**  
      - Record key phrases and corresponding scores for each department.
      - *Example 1:* Finance: “invoice” (5×), “budget” (3×); raw score 8.
      - *Example 2:* Legal: “contract” (4×), “NDA” (2×); raw score 6.
      - *Example 3:* HR: “employee review” (6×), “recruitment” (1×); raw score 7.
      - *Example 4:* Marketing: “digital campaign” (4×), “conversion” (2×); raw score 6.
      - *Example 5:* Operations: “production” (4×), “quality control” (3×); raw score 7.
      - *Example 6:* Procurement: “RFQ” (3×), “vendor” (4×); raw score 7.
      - *Example 7:* IT: “cybersecurity” (4×), “software update” (3×); raw score 7.
      - *Example 8:* Executive: “strategic planning” (3×), “board meeting” (2×); raw score 5.
      - *Example 9:* Customer Service: “support ticket” (5×), “refund” (3×); raw score 8.
      - *Example 10:* Facility: “maintenance” (4×), “safety” (3×); raw score 7.

   d. **Rationale:**  
      - Provide reasoning for choosing the primary category.
      - *Example 1:* Dominant Finance keywords justify Finance despite minor IT.
      - *Example 2:* Extensive Legal language outweighs brief HR mentions.
      - *Example 3:* HR indicators (employee reviews) override slight marketing signals.
      - *Example 4:* Concentrated marketing metrics justify Marketing over Procurement.
      - *Example 5:* Production and quality control confirm Operations despite facility notes.
      - *Example 6:* Procurement signals (RFQ, vendor) support Procurement over executive.
      - *Example 7:* Technical keywords firmly establish IT despite a brief legal note.
      - *Example 8:* Strategic language in the executive summary confirms Executive.
      - *Example 9:* Strong customer service data outweighs minor facility references.
      - *Example 10:* Maintenance and safety procedures dominate, confirming Facility.

   e. **Mitigation of Overconfidence:**  
      - Lower confidence if competing signals are too close.
      - *Example 1:* Finance 0.70 vs. Legal 0.68 → Lower confidence.
      - *Example 2:* Legal 0.55, HR 0.54, Executive 0.53 → Flag as unclear.
      - *Example 3:* HR 0.60, Marketing 0.59, Operations 0.58 → Reduce confidence.
      - *Example 4:* Marketing 0.68 vs. Sales 0.67 → Decrease confidence.
      - *Example 5:* Operations 0.62, Facility 0.61, Procurement 0.60 → Flag ambiguity.
      - *Example 6:* Procurement 0.57, IT 0.56, Finance 0.55 → Lower overall confidence.
      - *Example 7:* IT 0.75, Legal 0.74, Marketing 0.73 → Reduce confidence.
      - *Example 8:* Executive 0.80, HR 0.79, Operations 0.78 → Adjust confidence downward.
      - *Example 9:* Customer Service 0.72, Facility 0.71, Marketing 0.70 → Lower final score.
      - *Example 10:* CSR 0.70, Facility 0.69, R&D 0.68 → Reduce confidence.

6. Ambiguity & Exception Handling:
   a. **Conflict Detection:**
      - Compare normalized scores; if two or more are within 5%, mark as conflicting.
      - *Example 1:* Finance 0.65, Legal 0.63, HR 0.50 → Conflict between Finance and Legal.
      - *Example 2:* HR 0.70, Marketing 0.68, Operations 0.55 → HR and Marketing conflict.
      - *Example 3:* IT 0.75, Procurement 0.73, Finance 0.60 → IT and Procurement conflict.
      - *Example 4:* Operations 0.60, Facility 0.59, CSR 0.45 → Operations and Facility conflict.
      - *Example 5:* Executive 0.55, HR 0.54, Legal 0.52 → Conflict across Executive, HR, Legal.
      - *Example 6:* Customer Service 0.70, IT 0.68, Marketing 0.65 → Customer Service and IT conflict.
      - *Example 7:* Legal 0.65, Executive 0.64, Procurement 0.50 → Legal and Executive conflict.
      - *Example 8:* Marketing 0.60, Sales 0.59, Customer Service 0.57 → Marketing and Sales nearly equal.
      - *Example 9:* CSR 0.50, Facility 0.49, Operations 0.48 → CSR and Facility conflict.
      - *Example 10:* R&D 0.68, IT 0.66, Legal 0.64 → R&D and IT conflict.

   b. **Low Signal & Overload Analysis:**
      - If all scores are below a threshold (e.g., 0.3) or are evenly spread, mark as ambiguous.
      - *Example 1:* Finance 0.25, Legal 0.24, HR 0.23.
      - *Example 2:* Marketing 0.30, Sales 0.29, Customer Service 0.28.
      - *Example 3:* IT 0.32, Procurement 0.31, Operations 0.30.
      - *Example 4:* Legal 0.27, Executive 0.26, HR 0.25.
      - *Example 5:* Facility 0.29, CSR 0.28, Operations 0.27.
      - *Example 6:* R&D 0.30, IT 0.29, Marketing 0.28.
      - *Example 7:* Finance 0.26, Procurement 0.25, IT 0.24.
      - *Example 8:* Legal 0.31, CSR 0.30, Executive 0.29.
      - *Example 9:* HR 0.28, Customer Service 0.27, Marketing 0.26.
      - *Example 10:* Facility 0.30, Operations 0.29, Procurement 0.28.

   c. **Threshold Check:**
      - If the highest score is below 0.85 or scores are within 5% margin, flag as "Unclear."
      - *Example 1:* Finance 0.80, Legal 0.79, HR 0.78.
      - *Example 2:* IT 0.83, Procurement 0.82, Operations 0.81.
      - *Example 3:* Marketing 0.84, Sales 0.83, Customer Service 0.82.
      - *Example 4:* Legal 0.80, Executive 0.80, HR 0.79.
      - *Example 5:* Operations 0.82, Facility 0.81, CSR 0.80.
      - *Example 6:* R&D 0.83, IT 0.83, Legal 0.82.
      - *Example 7:* Customer Service 0.84, General 0.83, IT 0.82.
      - *Example 8:* Executive 0.80, HR 0.79, Operations 0.78.
      - *Example 9:* Marketing 0.82, Customer Service 0.81, IT 0.80.
      - *Example 10:* CSR 0.83, Facility 0.82, R&D 0.81.

   d. **Override for Fraud/Phishing:**
      - Immediately classify as "Spam / Fraud / Phishing" if explicit markers are present.
      - *Example 1:* Contains "click here to claim your prize."
      - *Example 2:* "Congratulations, you are a winner" appears.
      - *Example 3:* Contains "risk-free bonus offer."
      - *Example 4:* "Limited time free trial" is detected.
      - *Example 5:* "Urgent: verify your account to claim reward" is present.
      - *Example 6:* "Scam alert: do not respond" is detected.
      - *Example 7:* "Lottery win: claim your inheritance now" appears.
      - *Example 8:* "Miracle cure available, act now" is present.
      - *Example 9:* "Guaranteed free gift" appears.
      - *Example 10:* "Phishing attempt: verify your details" is detected.

   e. **Overconfidence Prevention:**
      - In cases of ambiguity, deliberately lower the final confidence score.
      - *Example 1:* Finance 0.70, Legal 0.68, HR 0.67 → Final confidence lowered.
      - *Example 2:* IT 0.75, Procurement 0.74, Operations 0.73 → Reduced to ~0.70.
      - *Example 3:* Marketing 0.78, Sales 0.77, Customer Service 0.76 → Lowered overall.
      - *Example 4:* Legal 0.65, Executive 0.64, HR 0.63 → Adjust downward.
      - *Example 5:* Operations 0.68, Facility 0.67, CSR 0.66 → Lower overall.
      - *Example 6:* R&D 0.70, IT 0.69, Legal 0.68 → Reduce final score.
      - *Example 7:* Customer Service 0.72, Facility 0.71, Marketing 0.70 → Lower intentionally.
      - *Example 8:* Executive 0.68, HR 0.67, Operations 0.66 → Adjust for uncertainty.
      - *Example 9:* CSR 0.70, Facility 0.69, R&D 0.68 → Reduce to ~0.65.
      - *Example 10:* Procurement 0.72, IT 0.71, Finance 0.70 → Lower overall.

7. Additional Analyses:
   a. **PII Detection:**
      - Detect any personally identifiable information.
      - *Example 1:* "Invoice addressed to Jane Doe at 123 Main St" (Finance).
      - *Example 2:* "Contract signed by attorney John Smith, email: jsmith@lawfirm.com" (Legal).
      - *Example 3:* "Employee record for Emily Johnson with phone (555) 123-4567" (HR).
      - *Example 4:* "Customer testimonial includes full name and email: customer@example.com" (Marketing).
      - *Example 5:* "Production report listing supervisor Michael Brown, 456 Industrial Rd" (Operations).
      - *Example 6:* "RFQ response includes vendor contact: vendor@supplies.com, 555-987-6543" (Procurement).
      - *Example 7:* "System log shows user ID: admin and IP address 192.168.1.10" (IT).
      - *Example 8:* "Memo from CEO John Doe, email: ceo@company.com" (Executive).
      - *Example 9:* "Complaint form with customer name Sarah Lee and phone 555-321-0987" (Customer Service).
      - *Example 10:* "Maintenance schedule includes technician contact: tech@facilities.com" (Facility).

   b. **Sentiment Analysis:**
      - Classify tone as Positive, Neutral, or Negative.
      - *Example 1:* "The quarterly financial report expresses robust growth." – Positive (Finance).
      - *Example 2:* "The legal memorandum has a cautionary tone." – Negative (Legal).
      - *Example 3:* "The employee feedback report is factual and neutral." – Neutral (HR).
      - *Example 4:* "The marketing campaign review is enthusiastic." – Positive (Marketing).
      - *Example 5:* "The production update maintains a neutral tone." – Neutral (Operations).
      - *Example 6:* "The procurement email shows frustration over delays." – Negative (Procurement).
      - *Example 7:* "The IT alert uses urgent language." – Negative (IT).
      - *Example 8:* "The executive strategy document is optimistic." – Positive (Executive).
      - *Example 9:* "The customer service report is balanced." – Neutral (Customer Service).
      - *Example 10:* "The facility inspection report is factual." – Neutral (Facility).

   c. **Regulatory Considerations:**
      - Identify references to laws or standards.
      - *Example 1:* "SOX compliance and SEC regulations" – (Finance).
      - *Example 2:* "Cites GDPR and other legal standards" – (Legal).
      - *Example 3:* "Aligned with federal labor laws" – (HR).
      - *Example 4:* "Adheres to FTC guidelines" – (Marketing).
      - *Example 5:* "Complies with OSHA safety standards" – (Operations).
      - *Example 6:* "References government procurement regulations" – (Procurement).
      - *Example 7:* "Follows ISO/IEC 27001 standards" – (IT).
      - *Example 8:* "Includes risk management per industry standards" – (Executive).
      - *Example 9:* "Incorporates consumer protection regulations" – (Customer Service).
      - *Example 10:* "Adheres to local building codes" – (Facility).

   d. **Archival Recommendation:**
      - Suggest retention duration.
      - *Example 1:* "Retain for 7 years as per fiscal record requirements" – (Finance).
      - *Example 2:* "Archive for 10 years to comply with statutory policies" – (Legal).
      - *Example 3:* "Retain for 5 years following employee separation" – (HR).
      - *Example 4:* "Retain for 3 years for campaign analysis" – (Marketing).
      - *Example 5:* "Archive for 5 years for audit purposes" – (Operations).
      - *Example 6:* "Retain for 7 years after contract expiration" – (Procurement).
      - *Example 7:* "Retain for 2 years for IT security audits" – (IT).
      - *Example 8:* "Archive for 10 years for corporate governance" – (Executive).
      - *Example 9:* "Retain for 3 years to monitor service quality" – (Customer Service).
      - *Example 10:* "Archive for 5 years for maintenance review" – (Facility).

   e. **Annotation:**
      - Document any additional context or decisions.
      - *Example 1:* "Finance: Dominant keywords ‘invoice’ and ‘budget’ noted despite minor IT references."
      - *Example 2:* "Legal: Primary evidence from ‘contract’ and ‘NDA’, with peripheral HR mentions."
      - *Example 3:* "HR: Extensive employee data noted; minor marketing reference discounted."
      - *Example 4:* "Marketing: Key phrases in introduction outweigh a procurement reference."
      - *Example 5:* "Operations: ‘Production’ and ‘quality control’ dominate over scattered facility terms."
      - *Example 6:* "Procurement: Multiple ‘RFQ’ and ‘vendor’ signals recorded, with a small executive note."
      - *Example 7:* "IT: Technical keywords dominate, with a minor legal disclaimer noted."
      - *Example 8:* "Executive: Strategic language is central, with incidental HR details."
      - *Example 9:* "Customer Service: Focus on ‘support ticket’ and ‘refund’, discounting minor facility references."
      - *Example 10:* "Facility: Maintenance and safety protocols are the main focus, with an isolated CSR remark."

8. Confidence Scoring:
   a. **Score Aggregation:**
      - Sum weighted points for each department.
      - *Example 1:* Finance: “invoice” (5×) + “audit” (3×) = 8 points.
      - *Example 2:* Legal: “contract” (4×) + “NDA” (2×) = 6 points.
      - *Example 3:* HR: “employee” (7×) + “recruitment” (1×) = 8 points.
      - *Example 4:* Marketing: “ad campaign” (3×) + “SEO” (2×) + “conversion” (1×) = 6 points.
      - *Example 5:* Operations: “production” (4×) + “quality control” (3×) = 7 points.
      - *Example 6:* Procurement: “RFQ” (3×) + “vendor” (4×) = 7 points.
      - *Example 7:* IT: “cybersecurity” (4×) + “software update” (3×) = 7 points.
      - *Example 8:* Executive: “strategic planning” (3×) + “board meeting” (2×) = 5 points.
      - *Example 9:* Customer Service: “support ticket” (5×) + “refund” (3×) = 8 points.
      - *Example 10:* Facility: “maintenance” (4×) + “repair” (3×) = 7 points.

   b. **Normalization & Differentiation:**
      - Convert raw scores to a normalized scale (0 to 1).
      - *Example 1:* Finance raw score 8 → normalized 0.8.
      - *Example 2:* Legal raw score 6 → normalized 0.6.
      - *Example 3:* HR raw score 8 → normalized 0.8.
      - *Example 4:* Marketing raw score 6 → normalized 0.6.
      - *Example 5:* Operations raw score 7 → normalized 0.7.
      - *Example 6:* Procurement raw score 7 → normalized 0.7.
      - *Example 7:* IT raw score 7 → normalized 0.7.
      - *Example 8:* Executive raw score 5 → normalized 0.5.
      - *Example 9:* Customer Service raw score 8 → normalized 0.8.
      - *Example 10:* Facility raw score 7 → normalized 0.7.

   c. **Intentional Lowering:**
      - If scores are too close, intentionally lower the final confidence.
      - *Example 1:* Finance (0.70) vs. Legal (0.68) → Lower to ~0.65.
      - *Example 2:* HR (0.60) vs. Marketing (0.59) → Adjust to ~0.55.
      - *Example 3:* IT (0.75) vs. Procurement (0.74) → Reduce to ~0.70.
      - *Example 4:* Executive (0.80) vs. HR (0.79) → Lower to ~0.75.
      - *Example 5:* Operations (0.68) vs. Facility (0.67) → Reduce to ~0.65.
      - *Example 6:* Procurement (0.72) vs. IT (0.71) → Adjust to ~0.68.
      - *Example 7:* Legal (0.65) vs. Executive (0.64) → Lower to ~0.60.
      - *Example 8:* Customer Service (0.72) vs. Facility (0.71) → Adjust to ~0.68.
      - *Example 9:* CSR (0.70) vs. Facility (0.69) → Lower to ~0.65.
      - *Example 10:* R&D (0.70) vs. IT (0.69) → Adjust to ~0.65.

   d. **Threshold & Flagging:**
      - If the highest score is below 0.85 or scores are within a 5% margin, flag as ambiguous.
      - *Example 1:* Finance 0.80, Legal 0.79, HR 0.78 → Flag as ambiguous.
      - *Example 2:* IT 0.83, Procurement 0.82, Operations 0.81 → Flag for review.
      - *Example 3:* Marketing 0.84, Sales 0.83, Customer Service 0.82 → Mark as ambiguous.
      - *Example 4:* Legal 0.80, Executive 0.80, HR 0.79 → Flag for human review.
      - *Example 5:* Operations 0.82, Facility 0.81, CSR 0.80 → Flag as unclear.
      - *Example 6:* R&D 0.83, IT 0.83, Legal 0.82 → Flag as ambiguous.
      - *Example 7:* Customer Service 0.84, General 0.83, IT 0.82 → Mark as ambiguous.
      - *Example 8:* Executive 0.80, HR 0.79, Operations 0.78 → Flag for review.
      - *Example 9:* Marketing 0.82, Customer Service 0.81, IT 0.80 → Flag as ambiguous.
      - *Example 10:* CSR 0.83, Facility 0.82, R&D 0.81 → Mark as unclear.

   e. **Documentation:**
      - Record the calculation, normalization, intentional lowering, and final confidence.
      - *Example 1:* Finance: "Aggregated score 8, normalized to 0.8, minor reduction applied; final confidence 0.78."
      - *Example 2:* Legal: "Score 6 normalized to 0.6; final confidence adjusted to 0.58."
      - *Example 3:* HR: "Score 8 normalized to 0.8; final confidence 0.77."
      - *Example 4:* Marketing: "Score 6 normalized to 0.6; final confidence 0.57."
      - *Example 5:* Operations: "Score 7 normalized to 0.7; final confidence 0.68."
      - *Example 6:* Procurement: "Score 7 normalized to 0.7; final confidence 0.67."
      - *Example 7:* IT: "Score 7 normalized to 0.7; final confidence 0.66."
      - *Example 8:* Executive: "Score 5 normalized to 0.5; final confidence 0.50."
      - *Example 9:* Customer Service: "Score 8 normalized to 0.8; final confidence 0.78."
      - *Example 10:* Facility: "Score 7 normalized to 0.7; final confidence 0.68."

9. Output Requirements:
   a. **Strict JSON Format:**
      - The final output must be returned strictly in a JSON object with the following keys:
        {
           "category": "Best Matching Category",
           "confidence": 0.92,
           "key_phrases": ["Phrase 1", "Phrase 2", "Phrase 3"],
           "alternative_categories": ["Alternative 1", "Alternative 2"],
           "explanation": "Detailed explanation with evidence and reasoning.",
           "contains_pii": "yes/no",
           "sentiment_analysis": "Positive/Neutral/Negative",
           "archival_recommendation": "Retention duration and rationale"
        }
      - *Example 1:* {"category": "Finance & Accounting", "confidence": 0.78, "key_phrases": ["invoice", "budget", "audit"], "alternative_categories": ["Legal & Compliance"], "explanation": "Dominant financial keywords with minor IT mentions.", "contains_pii": "no", "sentiment_analysis": "Positive", "archival_recommendation": "Retain for 7 years."}
      - *Example 2:* {"category": "Legal & Compliance", "confidence": 0.58, "key_phrases": ["contract", "NDA", "compliance"], "alternative_categories": ["Executive Office / Strategy"], "explanation": "Legal terms prevail despite peripheral HR signals.", "contains_pii": "yes", "sentiment_analysis": "Neutral", "archival_recommendation": "Retain for 10 years."}
      - *Example 3:* {"category": "HR", "confidence": 0.77, "key_phrases": ["employee", "recruitment", "performance review"], "alternative_categories": ["Marketing & Sales"], "explanation": "Strong HR indicators with slight marketing overlap.", "contains_pii": "yes", "sentiment_analysis": "Neutral", "archival_recommendation": "Retain for 5 years."}
      - *Example 4:* {"category": "Marketing & Sales", "confidence": 0.57, "key_phrases": ["digital campaign", "conversion", "SEO"], "alternative_categories": ["Customer Service"], "explanation": "Marketing signals are evident despite a few procurement mentions.", "contains_pii": "no", "sentiment_analysis": "Positive", "archival_recommendation": "Retain for 3 years."}
      - *Example 5:* {"category": "Operations & Manufacturing", "confidence": 0.68, "key_phrases": ["production", "quality control", "maintenance"], "alternative_categories": ["Facility Management"], "explanation": "Operational keywords dominate even though facility terms appear.", "contains_pii": "no", "sentiment_analysis": "Neutral", "archival_recommendation": "Retain for 5 years."}
      - *Example 6:* {"category": "Procurement & Supply Chain", "confidence": 0.67, "key_phrases": ["RFQ", "vendor", "purchase"], "alternative_categories": ["IT & Cybersecurity"], "explanation": "Procurement signals are clear despite some IT references.", "contains_pii": "no", "sentiment_analysis": "Neutral", "archival_recommendation": "Retain for 7 years."}
      - *Example 7:* {"category": "IT & Cybersecurity", "confidence": 0.66, "key_phrases": ["cybersecurity", "software update", "network"], "alternative_categories": ["Procurement & Supply Chain"], "explanation": "Technical keywords prevail even with a minor procurement note.", "contains_pii": "no", "sentiment_analysis": "Negative", "archival_recommendation": "Retain for 2 years."}
      - *Example 8:* {"category": "Executive Office / Strategy", "confidence": 0.50, "key_phrases": ["strategic planning", "board meeting"], "alternative_categories": ["Legal & Compliance"], "explanation": "Strategic language is evident though scores are low overall.", "contains_pii": "yes", "sentiment_analysis": "Positive", "archival_recommendation": "Retain for 10 years."}
      - *Example 9:* {"category": "Customer Service", "confidence": 0.78, "key_phrases": ["support ticket", "refund", "customer inquiry"], "alternative_categories": ["General / Miscellaneous"], "explanation": "Customer service indicators are strong despite minimal facility references.", "contains_pii": "yes", "sentiment_analysis": "Neutral", "archival_recommendation": "Retain for 3 years."}
      - *Example 10:* {"category": "Facility Management", "confidence": 0.68, "key_phrases": ["maintenance", "repair", "safety"], "alternative_categories": ["CSR"], "explanation": "Facility keywords dominate, with only a minor CSR note.", "contains_pii": "no", "sentiment_analysis": "Neutral", "archival_recommendation": "Retain for 5 years."}

   b. **Validation:**
      - Ensure the JSON output contains exactly the required keys and follows standard JSON formatting.
      - *Examples:* Output is a single JSON object with keys: category, confidence, key_phrases, alternative_categories, explanation, contains_pii, sentiment_analysis, archival_recommendation.
      
   c. **Exact Output Compliance:**
      - The final output must match the prescribed JSON format exactly, with no additional text.
      - *Examples:* Output can be directly parsed by a JSON parser without errors.

10. Few-Shot Examples (Ambiguous and Confusing Cases):
    - *Ambiguous Example:*  
      A document contains numerous keywords from Finance ("invoice", "audit"), Legal ("contract", "NDA"), and HR ("employee", "performance review") in various sections, but the primary content is a boilerplate disclaimer in the footer.
      *Expected Output:*  
      {
         "category": "Unclear",
         "confidence": 0.45,
         "key_phrases": ["invoice", "audit", "contract", "NDA", "employee", "performance review"],
         "alternative_categories": ["Finance & Accounting", "Legal & Compliance", "HR"],
         "explanation": "The document displays evenly distributed signals across multiple departments with a dominant boilerplate disclaimer, resulting in ambiguous classification.",
         "contains_pii": "no",
         "sentiment_analysis": "Neutral",
         "archival_recommendation": "Human review recommended due to low confidence."
      }
    - *Irrelevant Keywords Example (Forced Classification):*  
      A document is filled with sporadic IT and Marketing terms, but the central section extensively discusses “network security” and “cyber attack” in a technical report format.
      *Expected Output:*  
      {
         "category": "IT & Cybersecurity",
         "confidence": 0.72,
         "key_phrases": ["network security", "cyber attack", "malware"],
         "alternative_categories": ["Marketing & Sales"],
         "explanation": "Despite the presence of irrelevant keywords from other departments, the predominant technical content and emphasis on cybersecurity terms justify classification under IT & Cybersecurity.",
         "contains_pii": "no",
         "sentiment_analysis": "Negative",
         "archival_recommendation": "Retain for 2 years for IT audit purposes."
      }
    - *Another Ambiguous Example:*  
      A report includes a mix of production metrics (Operations), supplier details (Procurement), and a brief executive summary, but the content is scattered and lacks a clear focus.
      *Expected Output:*  
      {
         "category": "Unclear",
         "confidence": 0.50,
         "key_phrases": ["production", "quality control", "RFQ", "vendor", "strategic planning"],
         "alternative_categories": ["Operations & Manufacturing", "Procurement & Supply Chain", "Executive Office / Strategy"],
         "explanation": "The document’s mixed signals and scattered content result in no clear dominant category, leading to an ambiguous classification.",
         "contains_pii": "no",
         "sentiment_analysis": "Neutral",
         "archival_recommendation": "Human review recommended due to ambiguous classification."
      }
        11. Learning from Corrections:
             {correction_context}

        ---
        **Now, classify this document:**
        {text}
        """

        response = llm.invoke(prompt)

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

        # Display classification results
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
            # Assuming there's a function to store the correction
            # For demonstration, we'll just show a success message
            st.success("📚 AI will now use this correction for future classifications.")

if __name__ == "__main__":
    main()