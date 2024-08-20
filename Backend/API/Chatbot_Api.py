import torch
from mistralai import UserMessage
from Config import allcreds
import numpy as np
import torch.nn.functional as F
from testcase import testcase

from deepeval import evaluate
from deepeval.metrics import (ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric)
from deepeval.test_case import LLMTestCase


data=[]
str_phrase =[]
temp = []
final =[]

contextual_precision = ContextualPrecisionMetric(threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True)
contextual_recall = ContextualRecallMetric(threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True)
contextual_relevancy = ContextualRelevancyMetric(threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True)
faithfulness = FaithfulnessMetric(threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True)



device = "cuda" if torch.cuda.is_available() else "cpu"

# Get Creds from config.py

es_client = allcreds["es"]
groq_client = allcreds["client"]
openAI_client = allcreds["openAI"]
mistralai_client = allcreds["mistralai"]

colbert_creds=allcreds["colbert"].to(device)
colbert_tokenizer=allcreds["colbert_tokenizer"]

alibaba_creds=allcreds["Alibaba"].to(device)
alibaba_tokenizer=allcreds["Alibaba_tokenizer"]

sinequa_creds=allcreds["sinequamodel"].to(device)
sinequa_tokenizer=allcreds["sinequatokenizer"]

reranker = allcreds["cohere_reranker"]
vo = allcreds["Voyage"]


actual_output = ""

class Chat_api:

    def __init__(self):
        data=[]
        str_phrase =[]
        temp = []
        final =[]
        pass

    def eval_test(self,question, final, actual_output):

        input_quest = question
        expected_output = testcase[question]

        test_case = LLMTestCase(
            input= input_quest,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context= [hit['docstring'] for hit in final]
            )

        contextual_precision.measure(test_case)
        print("Retrival precision")
        print("Score: ", contextual_precision.score)
        print("Reason: ", contextual_precision.reason)
        print("-------------------------------------------------------------------------------------")

        contextual_recall.measure(test_case)
        print("Retrival recall")
        print("Score: ", contextual_recall.score)
        print("Reason: ", contextual_recall.reason)
        print("-------------------------------------------------------------------------------------")

        contextual_relevancy.measure(test_case)
        print("Retrival relevancy")
        print("Score: ", contextual_relevancy.score)
        print("Reason: ", contextual_relevancy.reason)
        print("-------------------------------------------------------------------------------------")

        answer_relevancy.measure(test_case)
        print("Generated relevancy")
        print("Score: ", answer_relevancy.score)
        print("Reason: ", answer_relevancy.reason)
        print("-------------------------------------------------------------------------------------")

        faithfulness.measure(test_case)
        print("Generated Faithfulness")
        print("Score: ", faithfulness.score)
        print("Reason: ", faithfulness.reason)
        print("-------------------------------------------------------------------------------------")
    
    def gpt_query(self,Question:str,File_name:str):   # Pass Question and filename as input query 
        
        data=[]
        str_phrase =[]
        temp = []
        final =[]                                                                                 
        setting_embed = es_client.get(index="settings", id="settings")['_source']['embedding']

        if(setting_embed == "OpenAI") :
            embeddings = openAI_client.embeddings.create(input = Question,model="text-embedding-3-small", encoding_format="float").data[0].embedding
            # embeddings = F.normalize(embeddings, p=2, dim=1)
            # embeddings = np.array(embeddings)[0]  

        if(setting_embed == "OpenAI256") :
            embeddings = openAI_client.embeddings.create(input = Question,model="text-embedding-3-small", encoding_format="float").data[0].embedding[:256]

        if(setting_embed == "Colbert") :
            tokens = colbert_tokenizer(Question, padding=True, truncation=True, return_tensors="pt").to(device)       # Get embeddings for the question using colbert model    
            with torch.no_grad():
                embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]
        
        if(setting_embed == "Alibaba") :
            tokens = alibaba_tokenizer(Question, padding=True, truncation=True, return_tensors="pt").to(device)           
            with torch.no_grad():                                                                                        
                embeddings = alibaba_creds(**tokens).last_hidden_state[:, 0] 
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = np.array(embeddings)[0] 

        if(setting_embed == "Sinequa") :
            tokens = sinequa_tokenizer(Question, padding=True, truncation=True, return_tensors="pt").to(device)           
            with torch.no_grad():                                                                                        
                embeddings = sinequa_creds(**tokens).last_hidden_state[:, 0] 
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = np.array(embeddings)[0]

        if(setting_embed == "OpenAI256" or setting_embed == "Sinequa") : 

            query = es_client.search( index="chatbot_index_256", body={                                       # Query to fetch first 3 document from elastic search index 
                    "_source": [
                        "docstring"],
                        'size':10, 
                    'query': { 'script_score': { 'query': {'match_all': {}                                # Match all documents in the index
                            }, 
                    'script': { 'source': "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",    # Script to calculate the cosine similarity score
                            'params': {'query_vector': embeddings                                      # Pass the query vector (embeddings) as a parameter to the script
                                } 
                            }
                        }
                    }, 
                    "fields": [
                                "filename",
                                "pagenumber"
                            ]
                } )
        else :
            query = es_client.search( index="chatbot_index", body={                                       # Query to fetch first 3 document from elastic search index 
                    "_source": [
                        "docstring"],
                        'size':10, 
                    'query': { 'script_score': { 'query': {'match_all': {}                                # Match all documents in the index
                            }, 
                    'script': { 'source': "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",    # Script to calculate the cosine similarity score
                            'params': {'query_vector': embeddings                                      # Pass the query vector (embeddings) as a parameter to the script
                                } 
                            }
                        }
                    },
                    "fields": [
                                "filename",
                                "pagenumber"
                            ]
                } )
        
        hits = query['hits']['hits']
        for hit in hits:
            data.append(hit)
            str_phrase.append({"docstring": hit["_source"]["docstring"],
                                "filename" : hit["fields"]["filename"][0]})
                                #"pagenumber" : hit["fields"]["pagenumber"][0]})  

        temp = [hit['docstring'] for hit in str_phrase]
        results = reranker.rerank(query=Question, documents=temp, top_n=5, model="rerank-english-v3.0", return_documents = True)
        #voresults = vo.rerank(Question, temp, model="rerank-lite-1", top_k=3)

        for result in results.results:
            val_index = int(result.index)
            final.append({"docstring": result.document.text,
                    "filename" : str_phrase[val_index]["filename"],
                    "index" : result.index,
                    "score" : result.relevance_score}) 

        # prompt = f""" Context information is below.
        #             ---------------------
        #             {final}
        #             ---------------------
        #             Given the context information and not prior knowledge, answer the query.
        #             provide the citiation using the filename availabe in the data.
        #             Query: {str(Question)}
        #             Answer:
        #             """

        chat_completion = openAI_client.chat.completions.create(                                       # calling llama -3 model (groq cloud ai) passing user question and elastic document to get response.
            messages=[
                {
                    "role": "system",
                    "content": f"""you are a helpful assistant,who answers users question based on the given data , also provide the citiation using the filename provided
                   
                    Data:{final}
                    """
                },
                {
                    "role": "user",
                    "content": f'{str(Question)}',
                }
            ],
            model="gpt-3.5-turbo",
            max_tokens=4090,
            temperature=0
        )

        # messages = [{"role"="user", "content"="prompt"}]
        # chat_completion = mistralai_client.chat.complete(model="mistral-large-latest", messages=messages, max_tokens=4090,
        #     temperature=0)

        actual_output = chat_completion.choices[0].message.content
        bool_evaluate = es_client.get(index="settings", id="settings")['_source']['eval']
        if(bool_evaluate):
            self.eval_test (Question,final, actual_output)
        return {1:data, 2:chat_completion.choices[0].message.content, 4: results}
    
    
Chat_bot=Chat_api()

