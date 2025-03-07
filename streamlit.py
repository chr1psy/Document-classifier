rom langchain_anthropic import ChatAnthropic
import os

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

test_prompt = "Classify this document: 'This is a phishing email with fraudulent intent.'"
response = llm.invoke(test_prompt)

print("Claude API Response:", response)