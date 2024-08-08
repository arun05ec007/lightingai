from io import BytesIO
from fastapi import  File, UploadFile, Body
import PyPDF2
import re
from langchain.text_splitter import LatexTextSplitter, SpacyTextSplitter
import torch
import numpy as np
from Config import allcreds
from typing import Dict, Annotated
import torch.nn.functional as F
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings



device = "cuda" if torch.cuda.is_available() else "cpu"

# Get Creds from config.py

es_client = allcreds["es"]
groq_client = allcreds["client"]

colbert_creds=allcreds["colbert"].to(device)
colbert_tokenizer=allcreds["colbert_tokenizer"]

alibaba_creds=allcreds["Alibaba"].to(device)
alibaba_tokenizer=allcreds["Alibaba_tokenizer"]

openAI_client = allcreds["openAI"]
mistralai_client = allcreds["mistralai"]
vo = allcreds["Voyage"]
embeddings_model = CohereEmbeddings(cohere_api_key="2Y8jF8avzmAnMN1S6vGOkWr2ihj8prQcJpf8RIpw", model="embed-english-v3.0", embedding_types=['float'])

# Class to extract file content and index in elastic.

class Extract_text:

    def __init__(self):
        pass
    
    def normalize_l2(self,x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)


    
    def IndexData(self,filename,docstring,count,page_num,embeddings):

        jsondata ={                                                                                                 # Index data into elastic search
            "filename":filename,
            "docstring":docstring,
            "count":count,
            "pagenumber" : page_num,
            "embeddings":embeddings
        }

        es_client.index(index="chatbot_index", document=jsondata)
        return "Data got indexed"

    def vector_creation(self,text_chunks,filename, page_num):

        count=0
        setting_embed = es_client.get(index="settings", id="settings")['_source']['embedding']

        if(setting_embed == "OpenAI") :
            resp_openAI = openAI_client.embeddings.create(input = text_chunks,model="text-embedding-3-small", encoding_format="float").data[0].embedding
            # np_emb = self.normalize_l2(resp_openAI)
            # print(np_emb.shape)
            # print(type(np_emb))
            self.IndexData(filename,text_chunks,count,page_num,resp_openAI)
        
        if(setting_embed == "OpenAI256") :
            resp_openAI = openAI_client.embeddings.create(input = text_chunks,model="text-embedding-3-small", encoding_format="float").data[0].embedding[:256]
            self.IndexData(filename,text_chunks,count,page_num,resp_openAI)
        
        if(setting_embed == "Colbert") :
            tokens = colbert_tokenizer(text_chunks, padding=True, truncation=True, return_tensors="pt").to(device)           # Tokenize text chunks
            with torch.no_grad():                                                                                        # Get embeddings for the passed chunks using colbert model
                embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]                            # Compute embeddings and convert to NumPy array 
                # print(type(embeddings))
                # print(embeddings.shape)
                self.IndexData(filename,text_chunks,count,page_num,embeddings)                                           # Store the embeddings with associated data
                count+=1

        if(setting_embed == "Alibaba") :
            tokens = alibaba_tokenizer(text_chunks, padding=True, truncation=True, return_tensors="pt").to(device)           
            with torch.no_grad():                                                                                        
                embeddings = alibaba_creds(**tokens).last_hidden_state[:, 0] 
                embeddings = F.normalize(embeddings, p=2, dim=1)  
                embeddings = np.array(embeddings)[0]  
                # print(type(embeddings))
                # print(embeddings.shape)                                                     
                self.IndexData(filename,text_chunks,count,page_num,embeddings)                                           # 
                count+=1

        # Convert bytes to a readable blob for PyPDF2
    async def Extract_text(self,file: UploadFile = File(...)):
        try:
            try:
                contents =await file.read()                                                                # Read file content.
                file_name=file.filename      
                print(file_name)                                                                           # Get file content
            except Exception:
                return {"message": "There was an error reading the file"}

            blob_to_read = BytesIO(contents)                                                                   # convert content into blob. 
            file_reader = PyPDF2.PdfReader(blob_to_read)                                                       # Read the blob using PdfReader
            text_content = ""
            for page_num in range(len(file_reader.pages)):                                                     # Loop through each pages and extract text.
                page = file_reader.pages[page_num]
                text_content += page.extract_text()
            
            # Load a pre-trained sentence transformer model
            latex_splitter = LatexTextSplitter(chunk_size=1024, chunk_overlap=75) 
            #spacy_splitter = SpacyTextSplitter(chunk_size=1024, chunk_overlap=75)                             # Split the content into chunks
            docs = latex_splitter.create_documents(texts=[text_content])                                       # Create a list of chunks
            #docs = spacy_splitter.split_text(text_content)
            print("Document chunk count : ", len(docs))

            # Load a pre-trained sentence transformer model
            # text_splitter = SemanticChunker( OpenAIEmbeddings(), breakpoint_threshold_type="gradient")            # Split the content into chunks
            # docs_semantic = text_splitter.create_documents(texts=[text_content])                               # Create a list of chunks
            # print("Document semantic chunk count : ", len(docs_semantic))

            for doc in docs:                                                      
                docstring=doc.page_content
                #docstring=doc
                docstring =re.sub('[^A-Za-z0-9]+', ' ', docstring)                                        
                self.vector_creation(docstring,file_name,page_num)                                             # Passing each chunk for vector creation
                                                                            
        except Exception as e:
            raise e
    
    def setSettings(self,settings:Annotated[dict, Body()]):
         return es_client.index(index="settings", document=settings, id="settings")
    
    def getSettings(self):
         return es_client.get(index="settings", id="settings")['_source']

Index_data=Extract_text()                                                                                      # Create instance for Extract_text_Index class
   