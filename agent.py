# ai-orchestrator/agent.py
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from hooks.security_filter import run_pre_routing_hook

#  THE UPGRADE: Importing our decoupled "Skill"
from skills.web_search import execute_web_search

from subagents.code_reviewer_agent import execute_code_review

# --- HELPER: Read Layer 1 (Memory) ---
def load_memory_layer(file_name: str) -> str:
    """Reads Markdown files from the memory/ directory to use as system prompts."""
    filepath = os.path.join(os.path.dirname(__file__), "memory", file_name)
    try:
        with open(filepath, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"⚠️ [MEMORY WARNING] Could not find {file_name}")
        return ""

# Load our System Prompts as Code
AGENT_PERSONA = load_memory_layer("persona.md")
ARCHITECTURE_RULES = load_memory_layer("architecture.rules.md")


# 1. DEFINE THE STATE
class GraphState(TypedDict):
    question: str
    document_id: str
    context: str
    route: str # Will be 'DATABASE', 'WEB', or 'GENERAL'
    answer: str

# Initialize our Local LLM
llm = ChatOpenAI(
    openai_api_base="http://host.docker.internal:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.1 
)


# 2. DEFINE THE NODES (The actions the AI can take)

def analyze_query_node(state: GraphState, vector_store):
    """The Supervisor: Decides where to send the question using the Persona memory."""
    print("🚦 [SUPERVISOR] Analyzing Query Intent...")


    # Layer 3 Guardrail Check
    security_error = run_pre_routing_hook(state["question"])
    if security_error:
        # If the hook catches something malicious, we force it to the GENERAL route
        # and override the answer immediately.
        state["answer"] = security_error 
        return {"route": "BLOCKED"} # We'll need to handle this route!


    question_lower = state["question"].lower()

    # 1. FAST-PATH: Keyword Guardrail
    doc_keywords = ["document", "pdf", "file", "context", "page"]
    if any(keyword in question_lower for keyword in doc_keywords):
        print("⚡ [GUARDRAIL] Document keyword detected. Forcing route to: DATABASE")
        return {"route": "DATABASE"}

    # 2. LLM ROUTING: Using Layer 1 Memory
    system_instruction = f"""
    {AGENT_PERSONA}
    
    You are a strict binary classification router.
    
    If the user pastes code or asks for a code review, output: REVIEW
    If the user asks a question requiring factual knowledge or history, output: WEB
    If the user is ONLY making conversational small talk, output: GENERAL

    Output EXACTLY ONE WORD. No punctuation. No explanation.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "{question}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    
    raw_output = result.content.strip().upper()
    print(f"🤖 [SUPERVISOR RAW LLM OUTPUT]: '{raw_output}'")
    
    # Safely extract the route
    if "REVIEW" in raw_output:
        route = "REVIEW"
    elif "WEB" in raw_output:
        route = "WEB"
    elif "DATABASE" in raw_output:
        route = "DATABASE"
    else:
        route = "GENERAL"
        
    print(f"🔀 [SUPERVISOR] Final Route chosen: {route}")
    return {"route": route}


def retrieve_database_node(state: GraphState, vector_store):
    """Pulls context from ChromaDB (Your existing RAG logic)."""
    print("🔍 [AGENT] Searching Vector Database...")
    results = vector_store.similarity_search(
        query=state["question"], 
        k=3, 
        filter={"document_id": state["document_id"]}
    )
    context = "\n\n".join([doc.page_content for doc in results])
    return {"context": context}


def search_web_node(state: GraphState, vector_store):
    """Pulls context from the live internet using the Layer 2 Skill."""
    print("🌐 [AGENT] Executing Skill: Web Search...")
    # 👉 THE UPGRADE: Calling our modular skill function
    context = execute_web_search(state["question"])
    return {"context": context}


def generate_response_node(state: GraphState, vector_store):
    """The final step: Generates the answer, constrained by Architecture Rules."""
    print("🧠 [AGENT] Generating Final Response...")
    
    if state["route"] == "GENERAL":
        context = "No context needed. Have a polite conversation."
    else:
        context = state["context"]
        
    # 👉 THE UPGRADE: Injecting both Persona and Architecture Rules into the final generation
    system_instruction = f"""
    {AGENT_PERSONA}
    
    {ARCHITECTURE_RULES}
    
    Answer the user's question strictly using the provided context below.
    
    CONTEXT:
    {context}
    """
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "{question}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    
    return {"answer": result.content}


# 3. DEFINE THE ROUTING LOGIC (The Edges)
def route_to_next_node(state: GraphState):
    if state.get("route") == "BLOCKED":
        return "blocked_exit" # 👉 THE FIX: Return a string
    elif state["route"] == "REVIEW":
        return "code_review" 
    elif state["route"] == "DATABASE":
        return "retrieve_database"
    elif state["route"] == "WEB":
        return "search_web"
    else:
        return "generate_response"




def code_review_node(state: GraphState, vector_store):
    """Delegates the user's question directly to the Code Review Subagent."""
    print("🤝 [SUPERVISOR] Delegating task to Code Review Subagent...")
    review_result = execute_code_review(state["question"])
    
    # We skip generate_response and just return the subagent's exact answer!
    return {"answer": review_result}

def build_agentic_graph(vector_store):
    print("🏗️ [AGENT] Compiling LangGraph State Machine...")
    
    # Initialize the graph
    workflow = StateGraph(GraphState)

    # Add our nodes, passing the vector_store to them using lambda functions
    workflow.add_node("analyze", lambda state: analyze_query_node(state, vector_store))
    workflow.add_node("retrieve_database", lambda state: retrieve_database_node(state, vector_store))
    workflow.add_node("search_web", lambda state: search_web_node(state, vector_store))
    workflow.add_node("generate_response", lambda state: generate_response_node(state, vector_store))
    
    # 👉 THE FIX: Added the new node inside the function!
    workflow.add_node("code_review", lambda state: code_review_node(state, vector_store))

    # Define the flow (The edges)
    workflow.set_entry_point("analyze")
    
    # Conditional Edge: Where do we go after 'analyze'?
    workflow.add_conditional_edges(
        "analyze",
        route_to_next_node,
        {
            "code_review": "code_review", # 👉 THE FIX: Registered the new route
            "retrieve_database": "retrieve_database",
            "search_web": "search_web",
            "generate_response": "generate_response",
            "blocked_exit": END # 👉 THE FIX: Map the string to the END object
            
        }
    )
    
    # Standard Edges: Where do we go after retrieval?
    workflow.add_edge("retrieve_database", "generate_response")
    workflow.add_edge("search_web", "generate_response")
    
    # Final Exit Edges
    workflow.add_edge("generate_response", END)
    workflow.add_edge("code_review", END) # 👉 THE FIX: Subagent outputs directly to user

    # Compile the graph
    return workflow.compile()