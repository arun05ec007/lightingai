from Config import allcreds

es_client = allcreds["es"]

class index:  

    def __init__(self):
        pass
    
    def Create_Index(self):                       # Create Index elastic index with mappings

        json_data= {"mappings":{
                "properties": {
                    "count": {
                        "type": "integer"
                    },
                    "docstring": {
                        "type": "text"
                    },
                    "embeddings": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "filename": {

                        "type": "text"
                    }
                }
            }
        }
        es_client.indices.create(index="chatbot_index",body=json_data)

    def Delete_Index(self, index_name) : 
        es_client.indices.delete(index="chatbot_index")

index_creation=index()