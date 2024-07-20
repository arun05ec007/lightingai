
from elasticsearch import Elasticsearch
import groq
import openai
import cohere
from mistralai.client import MistralClient
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

access_token = "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"

allcreds = {
    "es": Elasticsearch(
        [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
        verify_certs=False
    ),
    "client": groq.Client(api_key='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'),
    "openAI" : openai.Client(api_key ='sk-proj-VDHws6f41WZvdEP5NHZZT3BlbkFJssA15eF4K17KIQejjjMs'),
    "mistralai": MistralClient(api_key='KHuxRX2E8T8RxlguX1NVgrq1ujb7h8zX'),
    "cohere_reranker": cohere.Client(api_key='2Y8jF8avzmAnMN1S6vGOkWr2ihj8prQcJpf8RIpw'),
    "colbert" : AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
    "tokenizer" : AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
}
