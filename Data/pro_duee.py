import json

file_name = 'Data/DuEE_Fin/duee_fin_event_schema.json'
with open(file_name, 'r', encoding='utf-8') as file:
    schema = {}
    for line in file.readlines():
        one = json.loads(line)
        schema[one['event_type']] = []
        for item in one['role_list']:
            schema[one['event_type']].append(item['role'])
num = 0
all_num = 0
for mode in ['train.json', 'test.json']:
    file_name = 'Data/duee_row/' + mode
    save_name = mode
    file_save = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            doc = json.loads(line)

            doc_id = doc['id']

            text = doc['text']
            pos = text.find('原标题')
            if pos == -1:
                sentences = ['原标题：' + doc['title']]
            else:
                sentences = []
            
            pos = text.find('\n')
            while pos != -1:
                sentences.append(text[:pos])
                text = text[pos + 1:]
                pos = text.find('\n')
            sentences.append(text)

            recguid_eventname_eventdict_list = []
            if 'event_list' not in doc:
                continue
            for index, event in enumerate(doc['event_list']):
                event_dic = {}
                for role in schema[event['event_type']]:
                    have_role = False
                    for event_role in event['arguments']:
                        if role == event_role['role']:
                            event_dic[role] = event_role['argument']
                            have_role = True
                    if have_role == False:
                        event_dic[role] = None
                
                recguid_eventname_eventdict_list.append([index, event['event_type'], event_dic])

            ann_mspan2guess_field = {}
    
            for event in doc['event_list']:
                for event_role in event['arguments']:
                    if event_role['argument'] in ann_mspan2guess_field and event_role['role'] != ann_mspan2guess_field[event_role['argument']]:
                        num += 1
                    # ann_mspan2guess_field[event_role['argument']] = event_role['role']
                    ann_mspan2guess_field[event_role['argument']] = 'E'
                    all_num += 1
            
            ann_mspan2dranges = {}
            not_exit_key = []
            for key in ann_mspan2guess_field:
                ann_mspan2dranges[key] = []
                for index, sent in enumerate(sentences):
                    pos = sent.find(key)
                    if pos != -1:
                        ann_mspan2dranges[key].append([index, pos, pos + len(key)])
                if len(ann_mspan2dranges[key]) == 0:
                    not_exit_key.append(key)
            for key in not_exit_key:
                del ann_mspan2dranges[key]
                del ann_mspan2guess_field[key]
                for index, item in enumerate(recguid_eventname_eventdict_list):
                    for role, value in item[2].items():
                        if value == key:
                            recguid_eventname_eventdict_list[index][2][role] = None
            
            
            file_save.append([doc_id, {'sentences':sentences,
                                    'ann_valid_mspans':None,
                                    'ann_valid_dranges':None,
                                    'ann_mspan2dranges':ann_mspan2dranges,
                                    'ann_mspan2guess_field':ann_mspan2guess_field,
                                    'recguid_eventname_eventdict_list':recguid_eventname_eventdict_list}])

    json_str=json.dumps(file_save,indent=4,ensure_ascii=False)
    with open(save_name,'w',encoding='utf-8') as json_file:
        json_file.write(json_str)
    print(num, all_num, num / all_num)

