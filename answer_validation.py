from sentence_transformers import SentenceTransformer, util

# should be outside
model = SentenceTransformer("cointegrated/rubert-tiny2")


def get_solution_open_answer(answer_true: str, answer_user: str):
    """
    Function to understand whether the answer is correct.
    """
    embedding_1 = model.encode(answer_true, convert_to_tensor=True)
    embedding_2 = model.encode(answer_user, convert_to_tensor=True)

    score = util.pytorch_cos_sim(embedding_1, embedding_2)
    if score[0].item() > 0.7:
        return 1
    return 0
