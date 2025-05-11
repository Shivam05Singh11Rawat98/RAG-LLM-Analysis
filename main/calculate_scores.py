from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from rouge import Rouge
import nltk
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')
from evaluate import load
from bert_score import score

def calculate_scores(reference, generated):
    print(f"Calculating scores for reference: {reference} and generated: {generated}")
    smoothing = SmoothingFunction().method1
    # Convert to string to avoid AttributeError
    reference_tokens = str(reference).split()
    generated_tokens = str(generated).split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    rouge = Rouge()
    rouge_scores = rouge.get_scores(str(generated), str(reference))
    meteor = meteor_score([nltk.word_tokenize(reference)],nltk.word_tokenize(generated))
    ter = load('ter')
    ter_score = ter.compute(
        predictions=[generated],
        references=[reference]
    )['score']
    
    P, R, F1 = score([generated], [reference], lang="en")
    return bleu_score, rouge_scores[0]['rouge-1']['f'], rouge_scores[0]['rouge-2']['f'], rouge_scores[0]['rouge-l']['f'], ter_score/100, meteor, F1

def process_models(df, model_column):
    print(f"Processing {model_column}...")
    blue_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    meteor_scores = []
    ter_scores = []
    bert_scores = []
    # Iterate through the dataframe and calculate scores
    for i in range(len(df[model_column])):
        reference = df['manual_response'][i]
        generated = df[model_column][i]
        blue_score, rouge1_score, rouge2_score, rougel_score, ter_score, meteor, bert_score = calculate_scores(reference, generated)
        blue_scores.append(blue_score)
        rouge1_scores.append(rouge1_score)
        rouge2_scores.append(rouge2_score)
        rougel_scores.append(rougel_score)
        meteor_scores.append(meteor)
        ter_scores.append(ter_score)
        bert_scores.append(bert_score)
    print(f"Finished processing {model_column}.")
    # Save the scores to the dataframe
    return blue_scores, rouge1_scores, rouge2_scores, rougel_scores, meteor_scores, ter_scores,bert_scores

if __name__ == "__main__":
    df = pd.read_csv('similarity.csv')
    generated_columns = ['llama_response', 'gemma_response', 'mistral_response', 'deepseek_response', 'gpt_response']
    # llama_response, 'gemma_response', 'mistral_response', 'deepseek_response', 'gpt_response']
    for column in generated_columns:
        bleu_scores, rouge1_scores, rouge2_scores, rougel_scores, meteor_scores, ter_scores,bert_scores = process_models(df, column)
        df[f'{column}_bleu'] = bleu_scores
        df[f'{column}_rouge1'] = rouge1_scores
        df[f'{column}_rouge2'] = rouge2_scores
        df[f'{column}_rougel'] = rougel_scores
        df[f'{column}_meteor'] = meteor_scores
        df[f'{column}_ter'] = ter_scores
        df[f'{column}_bert'] = bert_scores
    # Save the dataframe with scores to a new CSV file
    print("Saving scores to CSV...")
    df.to_csv('similarity_scores.csv', index=False)
