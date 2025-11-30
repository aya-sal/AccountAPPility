from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class LocalEmbeddingManager:
    """Generate sentence embeddings using BAAI/bge-m3."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        print(f"Loading embedding model: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # bge-m3 uses 1024-dim dense embeddings
        self.dimensions = 1024

    def get_embedding(self, text: str) -> List[float]:
        """Return a normalized embedding suitable for Neo4j vector indexing."""

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        embedding = self.mean_pooling(outputs, inputs["attention_mask"])

        # Normalize for cosine similarity in Neo4j
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding[0].cpu().tolist()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (B, L, H)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        summed = torch.sum(token_embeddings * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        return summed / counts