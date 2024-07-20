
from elasticsearch import Elasticsearch
import groq
from transformers import AutoModel, AutoTokenizer


allcreds = {
    "es": Elasticsearch(
        [{'host': 'my-private-ntwk', 'port': 9200, 'scheme': 'http'}],
        verify_certs=False
    ),
    "client": groq.Client(api_key='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'),
    "colbert" : AutoModel.from_pretrained("BAAI/bge-large-en-v1.5"),
    "tokenizer" : AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
}
