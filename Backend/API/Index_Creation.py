from Config import allcreds

es_client = allcreds["es"]

class index:  

    def __init__(self):
        pass
    
    def Create_Index(self):  # Create Index elastic index with mappings
        try:                     
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
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m" : 16,
                                "ef_construction" : 100
                            }
                        },
                        "filename": {
                            "type": "keyword"
                        },
                        "pagenumber": {
                            "type": "integer"
                        }
                    }
                }
            }
            es_client.indices.create(index="chatbot_index",body=json_data)
            
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
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m" : 16,
                                "ef_construction" : 100
                            }
                        },
                        "filename": {
                            "type": "keyword"
                        },
                        "pagenumber": {
                            "type": "integer"
                        }
                    }
                }
            }
            es_client.indices.create(index="chatbot_index_256",body=json_data)

            return "Index created successfully"
        except Exception as e:
            return "Index already exists"

    def Delete_Index(self, index_name) : 
        try:
            es_client.indices.delete(index = index_name)
            return "Index deleted successfully"
        except Exception as e:
            return "Index doesn't exists"

index_creation=index()