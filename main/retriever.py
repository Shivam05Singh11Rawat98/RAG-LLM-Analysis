from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings

from pinecone import Pinecone

from dotenv import load_dotenv

load_dotenv()
import os
import warnings


def retrieve_from_pinecone(user_query="What information do you have on Instance Sync Permissions"):
    ## Pinecone context code:
    embeddings = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    ## Pinecone code:
    index_name = "text-mining"

    # connect to index
    index = pc.Index(index_name)

    # view index stats
    print("Index stats:",index.describe_index_stats())

    ### Use this to retrieve from existing vector store
    pinecone = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

    context= pinecone.similarity_search(user_query)[:5]
    return context
