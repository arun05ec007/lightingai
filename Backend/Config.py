
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
    "client": groq.Client(api_key='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'),
    "openAI" : openai.Client(api_key ='sk-proj-6wWhAgh2dd6AuN9ftMSuT3BlbkFJgJNjEts0NTDuKtxAZHeM'),
    "mistralai": Mistral(api_key='KHuxRX2E8T8RxlguX1NVgrq1ujb7h8zX'),
    "Voyage":voyageai.Client(api_key='pa-F0O4Q_ypTdM2bDjMg3kOgja9ybmzvRRyXkMjnEuNRvI'),
    "cohere_reranker": cohere.Client(api_key='2Y8jF8avzmAnMN1S6vGOkWr2ihj8prQcJpf8RIpw'),
    "colbert" : AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
    "colbert_tokenizer" : AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5"),
    "Alibaba" : AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True),
    "Alibaba_tokenizer" : AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
}
