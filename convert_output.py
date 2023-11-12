import ast

import pandas as pd
import json

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
#answer = 'устанавливаются временные знаки поднятия ножа и закрылок'
answer = "Перегон имеет раздельный пункт без дежурного по железнодорожной станции."
variants = [
  'Перегон имеет раздельный пункт без дежурного по железнодорожной станции.',
  'Журнал поездных телефонограмм не содержит образец, установленный для поездных телефонограмм.',
  'Перегон находится в зоне ответственности диспетчера поездного.',
  'На перегоне имеется раздельный пункт с дежурным по железнодорожной станции.'
]
print(sentence_bleu(variants, answer))
r = Rouge()
print(r.get_scores(answer, variants[0])[0]['rouge-1']['f'])

exit()

df = pd.read_excel('output.xlsx')
print(df)

data = {
    'ПРАВИЛА ТЕХНИЧЕСКОЙ ЭКСПЛУАТАЦИИ ЖЕЛЕЗНЫХ ДОРОГ РОССИЙСКОЙ ФЕДЕРАЦИИ': {
        'multiple_choice': []
    },
    'ИНСТРУКЦИЯ ПО СИГНАЛИЗАЦИИ НА ЖЕЛЕЗНОДОРОЖНОМ ТРАНСПОРТЕ РОССИЙСКОЙ ФЕДЕРАЦИИ': {
        'multiple_choice': []
    }
}

for row in df.iterrows():
    to_main = str.replace(row[1]['main_theme'], '\n', ' ')
    data[to_main]['multiple_choice'].append({
        'question': row[1]['question'],
        'answer': row[1]['answer_summary'],
        'variants': ast.literal_eval(row[1]['answers_merged'])
    })

with open('output.json', 'w') as jsf:
    json.dump(data, jsf)
