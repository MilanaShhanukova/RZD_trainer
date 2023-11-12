import pandas as pd

from questions_answering import get_parts_texts, respond
from make_embeddings import create_embeddings
from interaction import QuestionsGenerator

generated_data = 'generated_data.json'
small_texts_parts = get_parts_texts(generated_data)
embeddings_raw, embeddings_text = create_embeddings(small_texts_parts, 'cuda:0')
model_path = "model-q4_K.gguf"
qna = QuestionsGenerator(model_path)

df = pd.read_excel('Questions_list.xlsx')
answers = []
ppl = {}
for row in df.iterrows():
    print(row[1]['question'])
    text, ctx, llm_prompt, full_llm_answer = respond(row[1]['question'], embeddings_raw, embeddings_text, qna)
    answers.append(full_llm_answer[0])
    ppl[row[1]['index']] = full_llm_answer[1]
print(answers)
df['answer'] = answers
df.to_excel('submisssion_test.xlsx')
print(ppl)