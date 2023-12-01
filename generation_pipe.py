import json
from interaction import QuestionsGenerator
from tqdm import tqdm
import numpy as np


def create_generator(question_type):
    qr_generator = QuestionsGenerator(model_path=r"model-q4_K.gguf")
    user_prompt = qr_generator.prepare_user_prompt(question_type, questions_nums=4)
    return qr_generator, user_prompt


def generation_pipe(json_path: str, topic: str, question_type: str) -> str:
    all_ppl = []

    with open(json_path, "r", encoding="utf-8") as outfile:
        data = json.load(outfile)

    subinfo = data[topic]

    print(
        f"Start generating for {topic} topic! The type of question to generate is {question_type}"
    )
    generated_questions = {}

    qr_generator, user_prompt = create_generator(question_type)
    for subtopic, texts in tqdm(list(subinfo.items())):
        generated_questions[subtopic] = {question_type: []}
        print(f"Topic for generation: {subtopic}")

        for t in texts:
            try:
                questions_raw, ppl = qr_generator.get_questions(user_prompt, context=t)
                all_ppl.append(ppl)
            except TypeError:
                qr_generator, user_prompt = create_generator(question_type)
                questions_raw = qr_generator.get_questions(user_prompt, context=t)
                print(f"Text with index was not used")

            if question_type == "open question":
                questions, answers = qr_generator.process_open_questions(questions_raw)
                print(questions, answers)

                for q_idx in range(len(questions)):
                    generated_questions[subtopic][question_type].append(
                        {
                            "question": questions[q_idx],
                            "answer": answers[q_idx] if q_idx < len(answers) else None,
                            "context": t,
                        }
                    )
                with open(
                    f"LLM/{topic}_{subtopic}.json", "w+", encoding="utf-8"
                ) as outfile:
                    json.dump(
                        generated_questions, outfile, ensure_ascii=False, indent=4
                    )

            elif question_type == "multiple_choice":
                questions, answers, variants = qr_generator.process_multiple_choice(
                    questions_raw
                )
                for q_idx in range(len(questions)):
                    generated_questions[subtopic][question_type].append(
                        {
                            "question": questions[q_idx],
                            "answer": answers[q_idx],
                            "variants": variants[q_idx],
                            "context": t,
                        }
                    )
                with open(
                    f"LLM/{topic}_{subtopic}.json", "w+", encoding="utf-8"
                ) as outfile:
                    json.dump(
                        generated_questions, outfile, ensure_ascii=False, indent=4
                    )
    info = {topic: generated_questions}
    print("\n\nMean PPL:,", np.mean(all_ppl), "\n\n")
    return generated_questions


generation_pipe(
    r"generated_data.json",
    "ПОРЯДОК ОРГАНИЗАЦИИ ДВИЖЕНИЯ ПОЕЗДОВ ПРИ ПЕРЕРЫВЕ ДЕЙСТВИЯ ВСЕХ СИСТЕМ ИНТЕРВАЛЬНОГО РЕГУЛИРОВАНИЯ ДВИЖЕНИЯ ПОЕЗДОВ И СВЯЗИ",
    "multiple_choice",
)
