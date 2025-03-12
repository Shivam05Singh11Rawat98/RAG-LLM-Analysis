from sentence_transformers import SentenceTransformer, util
import pandas as pd

df = pd.read_csv('updated_query_df.csv')

df.columns

def similarity_score(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    print(similarity.item())
    return similarity.item()

def process_models(model_column):
    similarities=[]
    for i in range(df.shape[0]):
        manual_text = df['manual_response'][i]
        llm_text = df[model_column][i]
        
        similarities.append(similarity_score(manual_text,llm_text))
    return similarities

            
df['llama_similarity'] = process_models('llama_response')
df['gemma_similarity'] = process_models('gemma_response')
df['mistral_similarity'] = process_models('mistral_response')
df['deepseek_similarity'] = process_models('deepseek_response')
#df['gpt_response'] = gpt_response


df.to_csv('similarity.csv', index=False)


similar_df = pd.read_csv('similarity.csv')

print(similar_df['llama_similarity'].head())

