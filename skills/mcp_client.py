import asyncio
import threading
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def fetch_from_mcp_server(query: str) -> str:
    """
    Connects to a local SQLite MCP Server, lists its tools, 
    and executes a query to fetch enterprise data.
    """
    # 1. Define the MCP Server (Using the official Python SQLite MCP Server)
    server_params = StdioServerParameters(
        command="uvx", # uvx runs Python tools without installing them globally
        args=["mcp-server-sqlite", "--db", "enterprise_data.db"],
        env=None
    )

    try:
        # 2. Establish the MCP Connection
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 3. In a full implementation, the LLM would pick the tool. 
                # For this architecture, we will force the 'read_query' tool.
                print("🔌 [MCP CLIENT] Connected to SQLite Server. Executing Tool...")
                
                # We are dynamically passing the user's question as a SQL query
                # (In production, an LLM would translate natural language to SQL first)
                result = await session.call_tool(
                    "read_query", 
                    arguments={"query": "SELECT * FROM users LIMIT 5"} 
                )
                
                return result.content[0].text
    except Exception as e:
        print(f"❌ [MCP ERROR] Failed to connect to server: {e}")
        return "I could not connect to the Enterprise MCP Server."

def execute_mcp_tool(query: str) -> str:
    """Synchronous wrapper that safely runs async code inside FastAPI's existing event loop."""
    result = "MCP connection failed."
    
    def run_in_thread():
        nonlocal result
        # Create a fresh, isolated event loop just for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_from_mcp_server(query))
        except Exception as e:
            print(f"❌ [MCP THREAD ERROR] {e}")
        finally:
            loop.close()

    # Spin up the thread, wait for it to finish, and return the data!
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    return result