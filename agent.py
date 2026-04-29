# ai-orchestrator/agent.py
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, END

# 1. DEFINE THE STATE
# This acts as the "Memory" passed between nodes as the graph executes
class GraphState(TypedDict):
    question: str
    document_id: str
    context: str
    route: str # Will be 'DATABASE', 'WEB', or 'GENERAL'
    answer: str

# Initialize our Local LLM and Web Tool
llm = ChatOpenAI(
    openai_api_base="http://host.docker.internal:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.1 
)
web_search_tool = DuckDuckGoSearchRun()

# 2. DEFINE THE NODES (The actions the AI can take)

def analyze_query_node(state: GraphState, vector_store):
    """The Traffic Cop: Decides where to send the question."""
    print("🚦 [AGENT] Analyzing Query Intent...")
    question_lower = state["question"].lower()

    # 👉 THE FIX: Enterprise Hybrid Routing (Heuristics + LLM)

    # 1. FAST-PATH: Keyword Heuristics
    # If the user explicitly mentions the document, bypass the LLM entirely!
    doc_keywords = ["document", "pdf", "file", "context", "page"]
    if any(keyword in question_lower for keyword in doc_keywords):
        print("⚡ [AGENT GUARDRAIL] Document keyword detected. Forcing route to: DATABASE")
        return {"route": "DATABASE"}

    # 2. LLM ROUTING: For everything else
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict binary classification router.
        
If the user asks a question that requires factual knowledge, definitions, history, or identifying a person/thing (e.g., "what is node js", "who is einstein"), output: WEB
If the user is ONLY making conversational small talk (e.g., "hello", "how are you"), output: GENERAL

Output EXACTLY ONE WORD. No punctuation. No explanation."""),
        ("user", "{question}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    
    raw_output = result.content.strip().upper()
    print(f"🤖 [AGENT RAW LLM OUTPUT]: '{raw_output}'")
    
    # Safely extract the route
    if "WEB" in raw_output:
        route = "WEB"
    elif "DATABASE" in raw_output:
        route = "DATABASE"
    else:
        route = "GENERAL"
        
    print(f"🔀 [AGENT] Final Route chosen: {route}")
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
    """Pulls context from the live internet."""
    print("🌐 [AGENT] Searching the Web...")
    context = web_search_tool.invoke(state["question"])
    return {"context": context}


def generate_response_node(state: GraphState, vector_store):
    """The final step: Generates the answer based on whatever context was gathered."""
    print("🧠 [AGENT] Generating Final Response...")
    
    if state["route"] == "GENERAL":
        context = "No context needed. Have a polite conversation."
    else:
        context = state["context"]
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant. Answer the question using the provided context.\n\nCONTEXT:\n{context}"),
        ("user", "{question}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    
    return {"answer": result.content}


# 3. DEFINE THE ROUTING LOGIC (The Edges)
def route_to_next_node(state: GraphState):
    """Reads the state and tells the graph which node to go to next."""
    if state["route"] == "DATABASE":
        return "retrieve_database"
    elif state["route"] == "WEB":
        return "search_web"
    else:
        return "generate_response"



def build_agentic_graph(vector_store):
    print("🏗️ [AGENT] Compiling LangGraph State Machine...")
    
    # Initialize the graph
    workflow = StateGraph(GraphState)

    # Add our nodes, passing the vector_store to them using lambda functions
    workflow.add_node("analyze", lambda state: analyze_query_node(state, vector_store))
    workflow.add_node("retrieve_database", lambda state: retrieve_database_node(state, vector_store))
    workflow.add_node("search_web", lambda state: search_web_node(state, vector_store))
    workflow.add_node("generate_response", lambda state: generate_response_node(state, vector_store))

    # Define the flow (The edges)
    workflow.set_entry_point("analyze")
    
    # Conditional Edge: Where do we go after 'analyze'?
    workflow.add_conditional_edges(
        "analyze",
        route_to_next_node,
        {
            "retrieve_database": "retrieve_database",
            "search_web": "search_web",
            "generate_response": "generate_response"
        }
    )
    
    # Standard Edges: Where do we go after retrieval?
    workflow.add_edge("retrieve_database", "generate_response")
    workflow.add_edge("search_web", "generate_response")
    workflow.add_edge("generate_response", END)

    # Compile the graph
    return workflow.compile()