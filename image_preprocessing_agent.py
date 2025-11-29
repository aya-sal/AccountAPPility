from google.adk.agents.llm_agent import Agent
from PIL import Image
from io import BytesIO

class ImagePreprocessingAgent(Agent):
    """
    Receives an image input, applies preprocessing, and prepares it for downstream agents.
    Preprocessing may include: resizing, normalization, and basic quality corrections.
    This agent is deterministic and does not use LLM for its processing step.
    """

    def __init__(self, name="image_preprocessing_agent", description="Preprocesses images for downstream tasks"):
        # Note: Agent class requires a model parameter, but we override run() so it won't be used
        super().__init__(
            name=name,
            model='gemini-2.5-flash-lite',  # Required by Agent class, but not used since run() is overridden
            description=description,
            instruction="Preprocess incoming image and output the processed image data. Do not perform entity extraction."
        )

    def run(self, image_input, **kwargs):
        """
        Args:
            image_input: Raw image data (e.g., bytes, path, or file-like).

        Returns:
            dict: {
                "status": "success",
                "preprocessed_image": <image bytes or suitable format>
            }
        """
        try:
            # Place image preprocessing logic here.
            # This is placeholder code. Substitute with real logic as required.
            # For example: open image, resize, normalize, etc.
            

            if isinstance(image_input, bytes):
                img = Image.open(BytesIO(image_input))
            elif isinstance(image_input, str):
                img = Image.open(image_input)
            else:
                img = Image.open(image_input)

            # Example preprocessing: convert to RGB, resize to (512, 512)
            img = img.convert("RGB")
            img = img.resize((512, 512))

            # Serialize image back to bytes
            output = BytesIO()
            img.save(output, format='JPEG')
            processed_bytes = output.getvalue()
            output.close()

            return {
                "status": "success",
                "preprocessed_image": processed_bytes
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Image preprocessing failed: {str(e)}"
            }




