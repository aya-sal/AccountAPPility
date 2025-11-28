from google.adk.agents.llm_agent import Agent

data_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='data_input_agent',
    description='A helpful assistant for user questions.',
    instruction='ask the user to provide the picture of the reciept they want to file',
)
