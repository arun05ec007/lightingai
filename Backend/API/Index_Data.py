from io import BytesIO
from fastapi import  File, UploadFile, Body
import PyPDF2
import re
from langchain.text_splitter import LatexTextSplitter
import torch
import numpy as np
from Config import allcreds
from typing import Dict, Annotated


device = "cuda" if torch.cuda.is_available() else "cpu"

# Get Creds from config.py

es_client = allcreds["es"]
groq_client = allcreds["client"]
colbert_creds=allcreds["colbert"].to(device)
tokenizer_creds=allcreds["tokenizer"]



# Class to extract file content and index in elastic.

class Extract_text:

    def __init__(self):
        pass

    
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
        tokens = tokenizer_creds(text_chunks, padding=True, truncation=True, return_tensors="pt").to(device)           # Tokenize text chunks
        with torch.no_grad():                                                                                       # Get embeddings for the passed chunks using colbert model
            embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]                     # Compute embeddings and convert to NumPy array 
            self.IndexData(filename,text_chunks,count,page_num,embeddings)                                                   # Store the embeddings with associated data
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

            blob_to_read = BytesIO(contents)                                                                            # convert content into blob. 
            file_reader = PyPDF2.PdfReader(blob_to_read)                                                                # Read the blob using PdfReader
            text_content = ""
            for page_num in range(len(file_reader.pages)):                                                              # Loop through each pages and extract text.
                page = file_reader.pages[page_num]
                text_content += page.extract_text()
            
            # Load a pre-trained sentence transformer model
            latex_splitter = LatexTextSplitter(chunk_size=2048, chunk_overlap=50)                              # Split the content into chunks
            docs = latex_splitter.create_documents(texts=[text_content])                                       # Create a list of chunks
            
            for doc in docs:                                                      
                docstring=doc.page_content
                docstring =re.sub('[^A-Za-z0-9]+', ' ', docstring)                                        
                self.vector_creation(docstring,file_name,page_num)                                             # Passing each chunk for vector creation
                                                                            
        except Exception as e:
            raise e
    
    def setSettings(self,settings:Annotated[dict, Body()]):
         return es_client.index(index="settings", document=settings, id="settings")
    
    def getSettings(self):
         return es_client.get(index="settings", id="settings")['_source']

Index_data=Extract_text()                                                                                      # Create instance for Extract_text_Index class
   