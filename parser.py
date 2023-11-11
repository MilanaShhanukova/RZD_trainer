from typing import List, Dict
import re
import json
import pypdf


class Parser:
    def __init__(self, filepath: str):
        self.full_text = self.get_all_text(filepath)
        self.spltd_text = self.full_text.split("I. Общие положения")
        self.chapters = self.get_chapters()

    def get_all_text(self, filepath: str) -> str:
        all_text = ""
        reader = pypdf.PdfReader(filepath)
        number_of_pages = len(reader.pages)
        print("Страниц в документе:", number_of_pages)

        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            all_text += text
        print("Всего символов в документе:", len(all_text))
        return all_text

    def get_chapters(self) -> List:
        chapters = []
        # spltd_text = self.full_text.split('I. Общие положения')
        for i in range(len(self.spltd_text)):
            chpt = self.spltd_text[i].splitlines()[-12:]
            result_string = "".join(chpt)
            words = re.findall(r"\b[А-ЯЁ]+\b", result_string)
            result_string = " ".join(words)
            if result_string:
                chapters.append(result_string)
        return chapters

    def small_postprocessing(self):
        def change_sub_text(text):
            text = text.replace("\n", " ").split()
            text = " ".join(text)
            return text

        self.spltd_text = [change_sub_text(text) for text in self.spltd_text[1:]]

    def clean_text(self, text: str) -> str:
        pattern = r"\([^)]*\)"
        text = re.sub(pattern, "", text)

        pattern = r"\d{1,2}\.\d{1,3}\s-\d{4}\."
        text = re.sub(pattern, "", text)

        pattern = re.sub(r"[^А-Яа-я\s]", "", text)
        text = re.sub(pattern, "", text)

        pattern = r"ст\.\s\d+"
        text = re.sub(pattern, "", text)

        pattern = r"N\s\d+"
        text = re.sub(pattern, "", text)

        pattern = r"\b\d{4}\b"
        text = re.sub(pattern, "", text)

        pattern = r"<\d+>"
        text = re.sub(pattern, "", text)

        pattern = r"\d{1,2}\s[а-я]{4,}"
        text = re.sub(pattern, "", text)

        splitted_sentences = text.split(".")
        for i in range(len(splitted_sentences)):
            if (
                "ГОСТ" in splitted_sentences[i]
                or "Абзац" in splitted_sentences[i]
                or '"О' in splitted_sentences[i]
                or "акт" in splitted_sentences
            ):
                text = text.replace(splitted_sentences[i], " ")
        text = text.replace("Российской Федерации", " ")
        return text

    def get_paragraph_splitted(self, paragraph: str) -> List:
        clean_subpart = self.clean_text(paragraph)
        splitted_parts = re.split(r"\d\.\s", clean_subpart)

        stop_words = [
            "Собрание законодательства",
            "Зарегистрирован Министерством юстиции",
            "Устав железнодорожного транспорта",
            "-ФЗ",
            "г. ",
            "Подпункт",
        ]

        for idx, sent in enumerate(splitted_parts):
            for w in stop_words:
                sent = sent.replace(w, " ")
                sent = re.sub(r"[^\w\s.]", "", sent)
                sent = re.sub(r"\s{2,}", " ", sent)
            splitted_parts[idx] = sent
        splitted_parts = [
            sent for sent in splitted_parts if len(sent.replace(".", "")) > 10
        ]
        return splitted_parts

    def get_subtopics(self, topic_text: str) -> Dict:
        subtopic2text = {}
        matches = re.split(r"[IVXLCDM]+\.\s(.*?)(?=\d+\.)", topic_text)
        matches = [match.strip() for match in matches if match.strip()]
        matches = ["Общие положения"] + matches
        subtopic2text = dict(zip(matches[::2], matches[1::2]))

        for topic, text in subtopic2text.items():
            topics_parts = self.get_paragraph_splitted(text)

            subtopic2text[topic] = topics_parts
        return subtopic2text

    def prepare_dictionary(self) -> Dict:
        # topic_level_0: {subtopic_1: [texts], subtopic_2: [texts]}
        info_topics = {}
        for topic_idx in range(len(self.chapters)):
            info_topics[self.chapters[topic_idx]] = self.get_subtopics(
                self.spltd_text[topic_idx]
            )
        return info_topics

    def save_info(self, info):
        with open("./generated_data.json", "w") as outfile:
            json.dump(info, outfile, ensure_ascii=False)


if __name__ == "__main__":
    # add argparse + save dir
    filepath = "./rzd.pdf"
    parser = Parser(filepath)
    parser.small_postprocessing()

    info = parser.prepare_dictionary()

    parser.save_info(info)
