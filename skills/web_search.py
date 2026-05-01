from langchain_community.tools import DuckDuckGoSearchRun

def execute_web_search(query: str) -> str:
    """
    Skill: Live Web Search
    Description: Uses DuckDuckGo to scrape the live internet for factual information.
    """
    print(f"🌐 [SKILL INVOKED] Searching the web for: {query}")
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        return result
    except Exception as e:
        print(f"❌ [SKILL ERROR] Web search failed: {e}")
        return "I am currently unable to access the live internet."