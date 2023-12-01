# RZD_trainer
This is a repository of our solution for the hackathon "Цифровой прорыв." The main idea behind this project is to create an assistant that helps teach train workers. Our solution includes the generation of questions based on the texts and also answering these questions.

The pipeline includes the following steps:

1. Parse the PDF documents into topics. All the main topics with raw texts of small subtopics can be found in ./data/generated_data.json. Overall, we have parsed 12 big topics and 69 subtopics.
2. Next, we generated open and multiple-choice questions. The generation process was done using the LLM [Saiga-Mistral 7b](https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf). In total we have generated over 1200 questions. 
3. To better answer the questions according to the documents, we use embeddings similarity based on the cosine distance with LM [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base). 

The whole process was described in this [video](https://rutube.ru/video/a329139dbda52e5dd506ed3e81081e12/?t=4039&r=plwd) (rus)

This solution was ranked the 3rd. 
