import torch
from transformers import AutoTokenizer, AutoModel


class FinBertEmbedder:
    def __init__(self, model_name='yiyanghkust/finbert-tone', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def embed(self, texts):
        encoded_input = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_input)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu()
