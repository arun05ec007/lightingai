
from elasticsearch import Elasticsearch
import groq
import openai
import voyageai
import cohere
from mistralai import Mistral
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"]= "sk-proj-6wWhAgh2dd6AuN9ftMSuT3BlbkFJgJNjEts0NTDuKtxAZHeM"

access_token = "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"

allcreds = {
    "es": Elasticsearch(
        [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
        verify_certs=False
    ),
    "client": groq.Client(api_key=''),
    "openAI" : openai.Client(api_key =''),
    "mistralai": Mistral(api_key=''),
    "Voyage":voyageai.Client(api_key=''),
    "cohere_reranker": cohere.Client(api_key=''),
    "colbert" : AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
    "colbert_tokenizer" : AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
    "Alibaba" : AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True),
    "Alibaba_tokenizer" : AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True),
    "sinequatokenizer" : AutoTokenizer.from_pretrained("sinequa/vectorizer.vanilla"),
    "sinequamodel" : AutoModel.from_pretrained("sinequa/vectorizer.vanilla")
}
