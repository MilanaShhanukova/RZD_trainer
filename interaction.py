import fire
import re
from typing import List
from llama_cpp import Llama


class QuestionsGenerator:
    def __init__(self, model_path: str):
        SYSTEM_TOKEN = 1587
        USER_TOKEN = 2188
        BOT_TOKEN = 12435
        self.LINEBREAK_TOKEN = 13

        self.ROLE_TOKENS = {
            "user": USER_TOKEN,
            "bot": BOT_TOKEN,
            "system": SYSTEM_TOKEN,
        }

        self.SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

        self.model, self.system_tokens = self.prepare_model(model_path)

    def prepare_model(self, model_path: str):
        model = Llama(
            model_path=model_path,
            n_ctx=2000,
            n_parts=1,
        )
        system_tokens = self.get_system_tokens(model)
        tokens = system_tokens
        model.eval(tokens)
        return model, tokens

    def prepare_user_prompt_answer(self, question: str) -> str:
        prompt = f"Дай четкий и короткий ответ на вопрос '{question}', основываясь только на этом тексте:"
        return prompt

    def prepare_user_prompt(self, question_type: str, questions_nums: int) -> str:
        prompt = f"Сгенерируй {questions_nums} вопроса вида {question_type} на понимание данного текста и дай к ним ответы. Используй информацию только из этого текста: "
        return prompt

    def process_multiple_choice(self, output: str) -> (List, List, List):
        """
        Get answer, question and variants in the multiple choice answers from the LLM
        """
        pattern = r"Вопрос(\s\d+)?:(.*?)\nОтвет: (.*?)\n"
        text = re.sub(r"[a-dA-D1-9]\)|[a-dA-D1-9]\.", "-", output)

        matches = re.findall(pattern, text, re.DOTALL)

        questions, answers, answer_variants = [], [], []
        for match in matches:
            question = match[1].strip()
            answer = match[2].strip()

            question = question.split("\n-")

            answer_variants.append([variant.strip() for variant in question[1:]])

            questions.append(question[0])
            answers.append(answer)

        return questions, answers, answer_variants

    def process_open_questions(self, output: str) -> (List, List):
        """
        Get answer, question in the output of LLM open question generation
        """
        pattern = r"Вопрос(\d+)?:\s(.*?)\nОтвет: (.*?)\n"
        matches = re.findall(pattern, output, re.DOTALL)

        questions, answers = [], []
        for match in matches:
            question = match[1].strip()
            answer = match[2].strip()

            questions.append(question)
            answers.append(answer)
        if not questions or not answers:
            pattern_question = r"\b[\w\s]+[?]"
            pattern_answer = r"Ответ: .+"

            questions = re.findall(pattern_question, output, re.DOTALL)
            answers = re.findall(pattern_answer, output, re.DOTALL)
        return questions, answers

    def check_questions(self, question, answer, context):
        """
        Checks that the question is related to the context.
        """
        pass

    def get_system_tokens(self, model):
        system_message = {"role": "system", "content": self.SYSTEM_PROMPT}
        return self.get_message_tokens(model, **system_message)

    def get_message_tokens(self, model, role, content):
        message_tokens = model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, self.ROLE_TOKENS[role])
        message_tokens.insert(2, self.LINEBREAK_TOKEN)
        message_tokens.append(model.token_eos())
        return message_tokens

    def get_questions(
        self,
        prompt: str,
        context: str,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repeat_penalty=1.1,
    ) -> str:
        tokens = self.system_tokens

        user_message = prompt + context
        message_tokens = self.get_message_tokens(
            model=self.model, role="user", content=user_message
        )
        role_tokens = [
            self.model.token_bos(),
            self.ROLE_TOKENS["bot"],
            self.LINEBREAK_TOKEN,
        ]

        tokens += message_tokens + role_tokens

        full_prompt = self.model.detokenize(tokens)

        generator = self.model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty,
        )

        decoded_tokens = []
        for token in generator:
            token_str = self.model.detokenize([token]).decode("utf-8", errors="ignore")
            decoded_tokens.append(token_str)
            tokens.append(token)
            if token == self.model.token_eos():
                break
        return "".join(decoded_tokens)
