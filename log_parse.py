import pandas as pd

with open('log_to_parse.txt', 'r', encoding='utf-8') as logf:
    logs = logf.readlines()

last_question = ''
question = 'none'
ans_dict = {}
for l in logs:
    if len(l) > 5:
        if l[0:5] == 'text:':
            if not question == last_question:
                answer = l.strip().split('text: ')[1]
                print(l.strip().split('text: ')[1])
                ans_dict[last_question] = answer
                #print('----')
        if l[0:10] == 'user quest':
            question = l.strip().split('user question: ')[1]
            if not question == last_question:
                #print(l.strip().split('user question: ')[1])
                last_question = question
                question = ''
print(ans_dict)

df = pd.read_excel('submission_logtest.xlsx', index_col=None)
for quest in ans_dict:
    #print(ans_dict[quest])
    #print(df[df['question'] == quest])
    df.loc[df['question'] == quest, 'answer'] = ans_dict[quest]
print(df)
df.to_excel('submission_parsed.xlsx', index=False)