from fastapi import FastAPI
from API.Childroutes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()    # Initializing the fast API Service

app.include_router(router)  # Including child routes

origins = '*'

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



















# es_client = allcreds["es"]
# groq_client = allcreds["client"]
# colbert_creds=allcreds["colbert"]
# tokenizer_creds=allcreds["tokenizer"]


# @app.post("/upload_file")
# async def upload(file: UploadFile = File(...)):
#     try:
#         contents =await file.read()
#         file_name=file.filename
#     except Exception:
#         return {"message": "There was an error reading the file"}
    
#     # Convert bytes to a readable blob for PyPDF2
#     blob_to_read = BytesIO(contents)
#     file_reader = PyPDF2.PdfReader(blob_to_read)
#     text_content = ""

#     for page_num in range(len(file_reader.pages)):
#         page = file_reader.pages[page_num]
#         text_content += page.extract_text()
    
#     # Load a pre-trained sentence transformer model
#     latex_splitter = LatexTextSplitter(chunk_size=2000, chunk_overlap=0)
#     docs = latex_splitter.create_documents(texts=[text_content])
#     for doc in docs:
#         docstring=doc.page_content
#         vector_creation(docstring,file_name)

# all_embeddings = []

# def vector_creation(text_chunks,filename):
#     # data = text_chunks.split()
#     count=0
#     # for chunk in data:
#     tokens = tokenizer_creds(text_chunks, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]

#         IndexData(filename,text_chunks,count,embeddings)
#         count+=1
#     return all_embeddings
    

# def IndexData(filename,docstring,count,embeddings):
#     jsondata ={
#         "filename":filename,
#         "docstring":docstring,
#         "count":count,
#         "embeddings":embeddings
#     }

#     es_client.index(index="chatbot_index", document=jsondata)


# @app.post("/Create_Index")
# def Create_Index(Indexname:str):
#     json_data= {"mappings":{
#             "properties": {
#                 "count": {
#                     "type": "integer"
#                 },
#                 "docstring": {
#                     "type": "text"
#                 },
#                 "embeddings": {
#                     "type": "dense_vector",
#                     "dims": 768,
#                     "index": True,
#                     "similarity": "cosine"
#                 },
#                 "filename": {

#                     "type": "text"
#                 }
#             }
#         }
#     }
#     es_client.indices.create(index=Indexname,body=json_data)

    

# data=[]
# @app.post("/Chat_api")
# def Chat_api(Question:str,File_name:str):

#     tokens = tokenizer_creds(Question, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]
#     query = es_client.search( index="chatbot_index", body={
#             "_source": [
#                 "docstring"
#             ],'size':3, 'query': { 'script_score': { 'query': {'match_all': {}
#                     }, 
#             'script': { 'source': "cosineSimilarity(params.query_vector, 'embeddings') + 1.0", 'params': {'query_vector': embeddings
#                         }
#                     }
#                 }
#             }
#         } )
 
#     hits = query['hits']['hits']
#     for hit in hits:
#         data.append(hit)
#     # print(data)
#     chat_completion = groq_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": f"""you are a helpful assistant,who answers users question based on the given data
                
#                 Data:{data}
# """
#             },
#             {
#                 "role": "user",
#                 "content": f'{str(Question)}',
#             }
#         ],
#         model="llama3-8b-8192",
#     )
    
#     print(chat_completion.choices[0].message.content)




