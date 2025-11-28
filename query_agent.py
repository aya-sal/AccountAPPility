from google.adk.agents.llm_agent import Agent

query_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='query_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions about the previous reciepts',
)
