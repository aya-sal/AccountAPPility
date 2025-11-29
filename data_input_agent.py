from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models import Gemini
from .image_preprocessing_agent import ImagePreprocessingAgent
from google.genai import types

# data_agent should be a sequential agent that:
# 1 does data preprocessing it is just a deterministic step not an agentic function

# entity extraction



data_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='data_input_agent',
    description='A helpful assistant for user questions.',
    instruction='ask the user to provide the picture of the reciept they want to file',
)



model = Gemini(
            model='gemini-2.5-flash',
            generation_config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            )
        )


entity_prompt = """
You are a structured data extraction specialist. Your task is to analyze images and extract entities and relationships in a consistent JSON format.

    CRITICAL OUTPUT FORMAT:
    {
        "entities": [
            {
                "id": "unique_string_id",
                "type": "SpecificEntityType",
                "properties": {
                    "name": "entity_name",
                    "attribute1": "value1",
                    ...
                }
            }
        ],
        "relationships": [
            {
                "from": "source_entity_id",
                "to": "target_entity_id", 
                "type": "RELATIONSHIP_TYPE",
                "properties": {
                    "confidence": 0.95,
                    "attribute1": "value1",
                    ...
                }
            }
        ]
    }
    

    RULES:
    1. Entity types should be specific and meaningful (e.g., "Product", "Person", "Location", "Document", "Receipt")
    2. Relationship types should be descriptive and uppercase (e.g., "PURCHASED", "LOCATED_AT", "CONTAINS", "HAS_ITEM")
    3. Include ALL relevant properties you can extract
    4. For receipts: extract items, prices, quantities, totals, dates, store information
    5. For scenes: extract objects, people, spatial relationships, interactions
    6. For documents: extract key fields, sections, and their relationships
    7. Return ONLY valid JSON, no additional text or explanations

    EXAMPLE INPUT (receipt):
    "Receipt shows: 2x Coffee $3.50, 1x Bagel $2.00"

    EXPECTED OUTPUT:
    {
        "entities": [
            {"id": "item_1", "type": "PurchasedItem", "properties": {"name": "Coffee", "quantity": 2, "price": 3.50}},
            {"id": "item_2", "type": "PurchasedItem", "properties": {"name": "Bagel", "quantity": 1, "price": 2.00}}
        ],
        "relationships": [
            {"from": "receipt_1", "to": "item_1", "type": "PURCHASED", "properties": {"total_price": 7.0, "confidence": 0.95}},
            {"from": "receipt_1", "to": "item_2", "type": "PURCHASED", "properties": {"total_price": 2.0, "confidence": 0.95}}
        ]
    }
"""

entity_agent = Agent(
    name="entity_extractor",
    model=model, 
    description="A general agent to extract entities and relationships from images",
    instruction=entity_prompt,
    # before_model_callback=sys_file_analysis,
    output_key="extraction",
)

# Create an instance to be used in SequentialAgent
image_preprocessing_agent = ImagePreprocessingAgent()
data_input_pipeline_agent = SequentialAgent(
    name="data_input_pipeline_agent",
    sub_agents=[data_agent, image_preprocessing_agent, entity_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)
