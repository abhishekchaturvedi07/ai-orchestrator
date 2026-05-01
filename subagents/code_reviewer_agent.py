# ai-orchestrator/subagents/code_reviewer_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def execute_code_review(code_snippet: str) -> str:
    """
    Layer 4 Subagent: A highly specialized AI that only does Code Reviews.
    """
    print("🕵️‍♂️ [SUBAGENT] Code Reviewer Agent activated...")
    
    # We can use a different model, temperature, or strictness for this specific agent!
    llm = ChatOpenAI(
        openai_api_base="http://host.docker.internal:1234/v1",
        openai_api_key="lm-studio",
        temperature=0.0 # Strictly deterministic for code reviews
    )
    
    system_prompt = """You are a Principal Security & Code Review Engineer.
    Your ONLY job is to review the provided code snippet.
    
    Analyze the code for:
    1. Security vulnerabilities (e.g., SQL injection, hardcoded secrets)
    2. Performance bottlenecks
    3. Clean code practices
    
    Output your review in Markdown. If the code is perfect, just say 'LGTM (Looks Good To Me).'
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please review this code:\n\n{code}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"code": code_snippet})
    
    return result.content