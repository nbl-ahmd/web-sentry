from joblib import load
import dill 
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import idna
from urllib.parse import urlparse, urlunparse

# Load the pre-trained model
model_path = "./model/phishing_ensemble_model.joblib"  # Update this path
ensemble_classifier = load(model_path)

class MultiLLMFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def extract_features(self, texts, batch_size=16):
        features = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_embeddings)
        return np.vstack(features)

llm_extractor = MultiLLMFeatureExtractor()

def test_url(url):
    
    X_llm = llm_extractor.extract_features([url])
    print("Extracted features shape:", X_llm.shape)
    prediction = ensemble_classifier.predict(X_llm)[0]
    probability = ensemble_classifier.predict_proba(X_llm)[0][1]
    
    return {
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "probability": probability
    }

def check_homograph(url):
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc if parsed_url.netloc else parsed_url.path
        non_ascii_chars = [char for char in domain if ord(char) > 127]
        
        if non_ascii_chars:
            punycode_domain = idna.encode(domain).decode('ascii')
            punycode_url = urlunparse(
                (parsed_url.scheme, punycode_domain, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment)
            )
            return test_url(punycode_url)
        else:
            return test_url(url)
    except idna.IDNAError:
        return {"error": "Invalid URL format"}
