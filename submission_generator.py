import pandas as pd

from questions_answering import get_parts_texts, respond
from make_embeddings import create_embeddings
from interaction import QuestionsGenerator


def submission_generator(excel_path="Questions_list.xlsx"):
    df = pd.read_excel(excel_path)
    answers = []
    ppl = {}
    for row in df.iterrows():
        _, _, _, full_llm_answer = respond(
            row[1]["question"], embeddings_raw, embeddings_text, qna
        )
        answers.append(full_llm_answer[0])
        ppl[row[1]["index"]] = full_llm_answer[1]
    df["answer"] = answers
    df.to_excel("submisssion_test.xlsx")
    print(f"Perplexity is {ppl}")


if __name__ == "__main__":
    generated_data = "generated_data.json"
    small_texts_parts = get_parts_texts(generated_data)
    embeddings_raw, embeddings_text = create_embeddings(small_texts_parts, "cuda:0")
    model_path = "model-q4_K.gguf"
    qna = QuestionsGenerator(model_path)

    submission_generator()
