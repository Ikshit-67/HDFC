import os

import constants  
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


# Replace with actual API key
os.environ["OPENAI_API_KEY"] = constants.API_KEY  


loader = TextLoader('realistic_housing_projects_data.txt')

loader = TextLoader('data.txt')  
index = VectorstoreIndexCreator().from_loaders([loader])

while True:  # This will start an infinite loop
    query = input("\n Please enter your query (type 'exit' to quit): ")
    
    if query.lower() == 'exit':  # This will break the loop if user types 'exit'
        print("Exiting...")
        break
    
    # Print the query result
    print(index.query(query, llm=ChatOpenAI()))
