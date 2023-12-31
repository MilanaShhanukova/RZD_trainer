import torch
import numpy as np
import json
from make_embeddings import create_embeddings, get_embedding
from interaction import QuestionsGenerator


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_parts_texts(generated_data_path):
    with open(generated_data_path, "r", encoding="utf-8") as file:
        dict = json.load(file)

        # save only sub small texts
        parapraphs = []
        for topic in dict.keys():
            for subtopic in dict[topic]:
                for point_inx in range(len(dict[topic][subtopic])):
                    parapraphs.append(dict[topic][subtopic][point_inx])

        string_dict = {i: k for i, k in enumerate(parapraphs)}
    return string_dict


def respond(text, embeddings_raw, embeddings_text, qna, to_text=False):
    print(f"user question: {text}")
    text = "query: " + text
    embedding = get_embedding(text)

    # Поиск cosine simularity
    scores = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for embed in embeddings_raw:
        score = cos(torch.Tensor(embedding), torch.Tensor([embed]))
        scores.append(score[0])
    sorted_index = np.argsort(scores)[::-1]

    # User question
    context = embeddings_text[sorted_index[0]]
    llm_prompt = qna.prepare_user_prompt_answer(text)

    # LLM answer
    full_llm_answer = qna.get_questions(
        llm_prompt, context, top_k=30, top_p=0.9, temperature=0.3, repeat_penalty=1.1
    )
    return text, context, llm_prompt, full_llm_answer


if __name__ == "__main__":
    input_text = "Как часто могут отправляться поезда в одном направлении?"

    generated_data = "generated_data.json"
    model_path = "model-q4_K.gguf"

    small_texts_parts = get_parts_texts(generated_data)
    embeddings_raw, embeddings_text = create_embeddings(
        small_texts_parts, device
    )  # create embeddings

    qna = QuestionsGenerator(model_path)
    text, context, llm_prompt, full_llm_answer = respond(
        input_text, embeddings_raw, embeddings_text, qna
    )

    small_texts_parts = get_parts_texts(generated_data)
    embeddings_raw, embeddings_text = create_embeddings(small_texts_parts, device)

    while True:
        user_message = input("Вопрос: ")

        text, context, llm_prompt, full_llm_answer = respond(
            user_message, embeddings_raw
        )

        print(f"USER QUESTIONS:\n{text}\n")
        print(f"CONTEXT:\n{context}\n")
        print(f"ANSWER:\n{full_llm_answer}")
