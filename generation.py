from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import retriever
import pandas as pd
import ollama
from langchain_community.llms import Ollama

def get_context(user_query):    
    # user_query = "How much discretionary funding does the President's 2023 Budget request for USDA?"
    context=retriever.retrieve_from_pinecone(user_query)[:5]
    print(context)
    return context
    
query_df = pd.read_csv('test.csv')
contexts = []

for query in query_df['queries']:
    context = get_context(query)
    contexts.append(context)
    
    
query_df['context'] = contexts
query_df.to_csv('updated_w_context_data.csv', index=False)


def get_template(user_query, context):
    return """
            Using the context provided below, answer the user's question while ensuring the response
            is related to the U.S. budget. If the question or context falls outside this scope, politely inform the user that no relevant information was found and suggest rephrasing or changing the question. 
            Context: {context}
            User question: {user_query}
            """
    
def get_llm(model_name):
    if model_name == 'gpt':
        return 0
    else:
        return ChatOllama(model=model_name, temperature=0)
        
def get_response(user_query, context,model_name):
    llm = get_llm(model_name)
    template = get_template(user_query,context)
    prompt = ChatPromptTemplate.from_template(template)
    print(prompt)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
            "context": context,
            "user_query": user_query
            })

## llama 3.2 response
def process_models(model_name):
    responses=[]
    for i in range(query_df.shape[0]):
        query = query_df['queries'][i]
        context = query_df['context'][i]
        print(f"query: {query}, context: {context}")
        responses.append(get_response(query, context, model_name))
    return responses

            
query_df['llama_response'] = process_models('llama3.2')
query_df['gemma_response'] = process_models('gemma:2b')
query_df['mistral_response'] = process_models('mistral')
query_df['deepseek_response'] = process_models('deepseek-r1:1.5b')
#query_df['gpt_response'] = gpt_response
query_df['mistral_response']


query_df.to_csv('updated_query_df.csv', index=False)