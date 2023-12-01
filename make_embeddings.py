from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch
import numpy as np
import os
import json


model_name = "intfloat/multilingual-e5-base"


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def create_embeddings(string_dict, device="cuda"):
    embeddings_raw_name = "embeddings.npy"
    embeddings_text_name = "embeddings.txt"

    if not os.path.exists(embeddings_raw_name):
        print("Генерация эмбеддингов")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

        embeddings = []
        string_list = list(string_dict.values())
        string_list = [i for i in string_list if i is not None or i != ""]
        string_list = ["passage: " + i for i in string_list]

        with torch.no_grad():
            for line in string_list:
                batch_dict = tokenizer(
                    line,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**batch_dict)
                embedding = average_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                ).cpu()
                embeddings.append(embedding[0])
            embeddings = torch.stack(embeddings).cpu().detach().numpy()

        np.save(embeddings_raw_name, embeddings)
        with open(embeddings_text_name, "w", encoding="utf-8") as f:
            for line in string_list:
                f.write(line + "\n")

        embeddings_raw = embeddings
        embeddings_text = string_list

    else:
        print("Загрузка готовых эмбеддингов")
        embeddings_raw = np.load(embeddings_raw_name)
        with open(embeddings_text_name, "r", encoding="utf-8") as f:
            embeddings_text = f.readlines()

    return embeddings_raw, embeddings_text


def get_embedding(text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    batch_dict = tokenizer(
        text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**batch_dict)
    embedding = (
        average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        .cpu()
        .detach()
        .numpy()
    )
    return embedding


def get_parts_texts(generated_data_path: str):
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


if __name__ == "__main__":
    # add argument parser
    generated_data = "./generated_data.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    small_texts_parts = get_parts_texts(generated_data)

    embeddings_raw, embeddings_text = create_embeddings(small_texts_parts, device)
