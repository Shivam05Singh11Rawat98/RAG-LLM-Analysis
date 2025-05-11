# RAG-LLM-Analysis
**RAG on US Budget Documents**

This study investigates how effectively Large Language Models (LLMs) can retrieve and interpret information from complex U.S. federal budget documents to improve transparency and accessibility in public finance. A Retrieval-Augmented Generation (RAG) approach is employed, utilizing SentenceTransformer embeddings and the Pinecone vector database for semantic, query-based retrieval. Five LLMs—LLaMA 3.2B, Mistral 7B, GPT-3.5-turbo 20B, Gemma2 2B, and DeepSeek-R1 1.5B—are evaluated on 50 curated queries covering fiscal years 2023 to 2025. Their responses are compared to manually crafted reference answers using evaluation metrics such as BLEU, ROUGE-1/2/L, METEOR, TER, BERTScore, and semantic similarity. Among the models, LLaMA 3.2B delivered the best overall results, with a BLEU score of 0.2403, ROUGE-L of 0.4550, METEOR of 0.5545, and the lowest TER. GPT-3.5-turbo performed closely behind, scoring 0.2091 on BLEU and 0.4938 on METEOR. DeepSeek-R1, the smallest model, had the weakest results, including a BLEU score of 0.0164 and a TER of 91.6. These findings highlight the superior ability of larger models, especially when combined with robust retrieval systems, to produce accurate and coherent responses to complex, domain-specific financial queries.


![image](https://github.com/user-attachments/assets/720e8988-6a78-46a2-8de3-1c2086c58fd6)
![image](https://github.com/user-attachments/assets/c7590dc7-a03d-403c-84fb-4b1aef297850)

![image](https://github.com/user-attachments/assets/20ad84a9-0c06-4f61-8251-d60ceab887e7)
![image](https://github.com/user-attachments/assets/9c282ed3-1d3f-4c37-aa29-4b733359cf4a)

![image](https://github.com/user-attachments/assets/ed3c3797-bbba-42a1-ac2e-c3cbf8de56b5)


![image](https://github.com/user-attachments/assets/af080955-d624-464a-ba99-5b1c5c8e12c8)

**Model Parameters and Evaluation Results**
This study assesses the performance of five Large Language Models (LLMs) in responding to budget-related queries within a retrieval-augmented generation (RAG) framework. The models evaluated include LLaMA 3.2B, Mistral 7B, Gemma2 2B, DeepSeek-R1 1.5B, and GPT-3.5-turbo 20B. Each model was tested on 50 carefully curated questions designed to cover a broad range of topics in U.S. government finance, such as expenditure distribution, policy programs, and agency-level investments.

For every query, a reference response was manually generated based on official budget documents, serving as a benchmark for evaluating the model-generated answers. The RAG setup retrieved the top-5 semantically relevant text chunks from a Pinecone vector database, which were then supplied to the LLMs for context-aware response generation.

A standardized prompt was used to ensure relevance and consistency in outputs. It instructed models to answer strictly within the scope of the U.S. budget based on the given context. If the query or retrieved content fell outside that scope, the models were directed to inform the user accordingly and suggest rephrasing. A temperature setting of 0 was applied to minimize hallucinations and promote factual, deterministic outputs.

To evaluate response quality, several well-established natural language metrics were used: BERTScore, BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, Translation Edit Rate (TER), and semantic similarity. These metrics assess semantic accuracy, lexical overlap, fluency, and fidelity to the reference responses.

Among the models tested, LLaMA 3.2B performed best overall, leading in BLEU (0.2403), ROUGE-1 (0.4704), ROUGE-2 (0.3220), ROUGE-L (0.4550), METEOR (0.5545), and semantic similarity (0.7776), reflecting its ability to produce fluent and information-rich answers. GPT-3.5-turbo closely followed, topping the BERTScore metric (0.910) and maintaining high consistency across others, indicating strong alignment with reference answers.

Mistral demonstrated moderate performance, with a high METEOR score (0.397) and BERTScore (0.883), though its lower BLEU score (0.102) and similarity rating (0.730) suggested more generalized or paraphrased outputs. Gemma2 showed balanced results with BLEU (0.120), METEOR (0.320), and similarity (0.710), indicating reasonable lexical and semantic alignment.

DeepSeek-R1, being the smallest model, showed the weakest results, scoring the lowest BLEU (0.016), highest TER, and a similarity score of 0.643—highlighting challenges in handling domain-specific financial language.

Overall, the findings indicate that larger models like LLaMA 3.2B and GPT-3.5-turbo are more capable in tasks that demand detailed understanding of complex financial texts. In contrast, smaller models may require additional fine-tuning or targeted pretraining to be effective in such specialized domains.


**Recommendations and Future Directions:**
This analysis underscores the importance of adopting more advanced evaluation approaches that account for both semantic understanding and contextual relevance. Relying solely on surface-level metrics such as ROUGE or BLEU can overlook responses that are semantically accurate but lexically varied. Integrating human evaluations or expert feedback will be crucial for achieving more precise and meaningful assessments in future studies.

The comparatively lower performance of DeepSeek-R1 also indicates that models with smaller parameter sizes may face challenges in handling domain-specific content like government budgeting. Future efforts should consider leveraging larger models or fine-tuning smaller ones on targeted public finance datasets to enhance domain adaptability.

Moreover, enhancing the retrieval component of the RAG framework through techniques like reranking, query expansion, or feedback-based refinement could significantly improve the relevance of retrieved context, thereby boosting the accuracy and reliability of model-generated responses.


