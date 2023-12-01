import json
import os
import random
import string

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    filters,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
)

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from questions_answering import get_parts_texts, respond
from make_embeddings import create_embeddings
from interaction import QuestionsGenerator


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"incoming start: {update.message.from_user.first_name}")
    open_chats[update.effective_chat.id] = {}
    keyboard = [
        [InlineKeyboardButton("Начать тестирование", callback_data="start_button")],
        [InlineKeyboardButton("Задать вопрос боту", callback_data="question_button")],
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Fttftf - Виртуальный тренажер",
        reply_markup=markup,
    )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Update(callback_query=CallbackQuery(chat_instance='5459393346535110676', data='start_button', from_user=User(first_name='рыба', id=5449975720, is_bot=False, language_code='ru'), id='4960723410556417369', message=Message(channel_chat_created=False, chat=Chat(first_name='рыба', id=5449975720, type=<ChatType.PRIVATE>), date=datetime.datetime(2023, 11, 12, 1, 30, 30, tzinfo=<UTC>), delete_chat_photo=False, from_user=User(first_name='fttftf_bot', id=6959250678, is_bot=True, username='fttftf_bot'), group_chat_created=False, message_id=6, reply_markup=InlineKeyboardMarkup(inline_keyboard=((InlineKeyboardButton(callback_data='start_button', text='Начать тестирование'),),)), supergroup_chat_created=False, text='test bot')), update_id=138828566)
    # <telegram.ext._callbackcontext.CallbackContext object at 0x0000026BCB9102E0>
    print(f"incoming button: {update.callback_query.data}")

    query_id = update.callback_query.data

    if query_id == "start_button":
        keyboard = []
        for i, data in enumerate(qna_files):
            keyboard.append(
                [InlineKeyboardButton(data["name"], callback_data=f"qna_{i}")]
            )
        print(keyboard)
        markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Выберите тему, по которой будет проводиться тестирование:",
            reply_markup=markup,
        )
    if query_id == "question_button":
        open_chats[update.effective_chat.id] = {}
        open_chats[update.effective_chat.id]["do_question"] = True
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="Задавайте ваш вопрос боту"
        )
    if query_id.split("_")[0] == "qna":
        theme_id = int(query_id.split("_")[1])
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f'Ваша тема: {qna_files[theme_id]["name"]}',
        )
        open_chats[update.effective_chat.id] = {}
        open_chats[update.effective_chat.id]["theme"] = theme_id
        open_chats[update.effective_chat.id]["question_num"] = 0

        keyboard = []
        for i, key in enumerate(qna_files[theme_id]["data"].keys()):
            keyboard.append([InlineKeyboardButton(key, callback_data=f"qnasub_{i}")])
        print(keyboard)
        markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Выберите раздел:",
            reply_markup=markup,
        )
    if query_id.split("_")[0] == "qnasub":
        subtheme_id = int(query_id.split("_")[1])
        open_chats[update.effective_chat.id]["subtheme"] = subtheme_id

        text = send_random_question(update.effective_chat.id)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f'Вопрос {open_chats[update.effective_chat.id]["question_num"]}: {text}',
        )


async def message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"incoming message: {update.message.text}")
    chat_id = update.effective_chat.id

    if "answer" in open_chats[update.effective_chat.id]:
        input_text_stripped = strip_string(update.message.text)
        answer_stripped = strip_string(open_chats[update.effective_chat.id]["answer"])
        answer_list = [answer_stripped]
        if "variants" in open_chats[update.effective_chat.id]:
            for v in open_chats[update.effective_chat.id]["variants"]:
                answer_list.append(strip_string(v))
        r = Rouge()
        metric_string = f'bleu: {sentence_bleu(answer_list, input_text_stripped)}, rouge f1: {r.get_scores(input_text_stripped, answer_list[0])[0]["rouge-1"]["f"]}'

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f'Правильный ответ: {open_chats[update.effective_chat.id]["answer"]}\n{metric_string}',
        )
        text = send_random_question(chat_id)
        if "question_num" in open_chats[update.effective_chat.id]:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f'Вопрос {open_chats[update.effective_chat.id]["question_num"]}: {text}',
            )
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=text)
    if "do_question" in open_chats[update.effective_chat.id]:
        input_text = update.message.text
        text, ctx, llm_prompt, full_llm_answer = respond(
            input_text, embeddings_raw, embeddings_text, qna
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"{full_llm_answer[0]}\nppl: {full_llm_answer[1].numpy()}",
        )


def strip_string(text):
    return str.lower(text.translate(str.maketrans("", "", string.punctuation)))


def send_random_question(chat_id):
    theme_id = open_chats[chat_id]["theme"]
    theme = qna_files[theme_id]["data"]
    for i, key in enumerate(theme.keys()):
        if i == open_chats[chat_id]["subtheme"]:
            # print(theme[key])
            subtheme = theme[key]["multiple_choice"]
    question = random.choice(subtheme)
    q = question["question"]
    a = question["answer"]
    if "variants" in question:
        a_list = question["variants"]
        open_chats[chat_id]["variants"] = a_list
    open_chats[chat_id]["answer"] = a
    open_chats[chat_id]["question_num"] += 1

    if open_chats[chat_id]["question_num"] > 5:
        del open_chats[chat_id]["theme"]
        del open_chats[chat_id]["subtheme"]
        del open_chats[chat_id]["question_num"]
        del open_chats[chat_id]["answer"]
        del open_chats[chat_id]["variants"]

        return "Вы ответили на все вопросы!"
        # await context.bot.send_message(chat_id=chat_id, text='Вы ответили на все вопросы!')

    # await context.bot.send_message(chat_id=chat_id, text=q)
    return q


if __name__ == "__main__":
    with open("cfg.json", "r") as jsf:
        config = json.load(jsf)

    generated_data = "E:\dev\RZD_trainer\LLM\generated_data.json"
    small_texts_parts = get_parts_texts(generated_data)

    embeddings_raw, embeddings_text = create_embeddings(
        small_texts_parts, "cuda:0"
    )  # create embeddings

    model_path = "E:\dev\RZD_trainer\model-q4_K.gguf"
    qna = QuestionsGenerator(model_path)

    open_chats = {}
    qna_files = []
    for f in os.listdir("E:\dev\RZD_trainer\qna_data"):
        if ".json" in f:
            with open(f"E:\dev\RZD_trainer\qna_data\{f}", "r", encoding="utf-8") as jsf:
                data = json.load(jsf)
            theme_name = f.split("_")[0]
            qna_files.append({"name": theme_name, "data": data})

    app = ApplicationBuilder().token(config["token"]).build()

    start_handler = CommandHandler("start", start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), message)
    app.add_handler(start_handler)
    app.add_handler(message_handler)
    app.add_handler(CallbackQueryHandler(button))

    print("loading finished")

    app.run_polling()
