# Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.genai import types
from .data_input_agent import data_input_pipeline_agent
from .query_agent import query_agent

print("✅ ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # This instruction tells the root agent HOW to use its tools (which are the other agents).
    instruction="""You are a personal accounting coordinator. Your goal is to answer the user's query by orchestrating a workflow.
    The user will interact with you to provide data OR to ask you a question about existing data. The data the user will provide is a picture of a sales reciept.
1. First, you MUST identify whether the user wants to provide data or ask a question.
2. If the user wants to provide a picture of a document call 'data_input_pipeline_agent'
3. If the user wants to ask a question and get the response from the database call the 'query_agent'

When you invoke the 'data_input_pipeline_agent' show the user the information that was extracted  """,
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[AgentTool(data_input_pipeline_agent), AgentTool(query_agent)]
)

print("✅ root_agent created.")