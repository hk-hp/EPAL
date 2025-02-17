import torch
import json
import numpy as np
from procnet.data_example.DocEEexample import DocEELabel
from procnet.utils.util_structure import UtilStructure
from procnet.model.basic_model import BasicModel
import copy
from procnet.utils.util_data import UtilData

row = torch.load('raw.pt')[:1000]
dic_data = torch.load('all.pt')
pred = dic_data ['pred']
gold = dic_data ['gold']
copy_num = dic_data ['copy_num']
if len(copy_num) != len(gold):
    for i in range(len(gold) - len(copy_num)):
        copy_num.append(0)

with open('Data/DuEE_Fin/duee_fin_event_schema.json', 'r', encoding='utf-8') as file:
    SCHEMA = {}
    for line in file.readlines():
        one = json.loads(line)
        SCHEMA[one['event_type']] = []
        for item in one['role_list']:
            SCHEMA[one['event_type']].append(item['role'])

event_type_label_set = set()
event_role_label_set = set()
for k, v in SCHEMA.items():
    event_type_label_set.add(k)
    [event_role_label_set.add(x) for x in v]
event_type_index_to_type = ['Null'] + sorted(list(event_type_label_set))
event_type_type_to_index = {event_type_index_to_type[i]: i for i in range(len(event_type_index_to_type))}
event_role_index_to_relation = ['Null'] + sorted(list(event_role_label_set))
event_role_relation_to_index = {event_role_index_to_relation[i]: i for i in range(len(event_role_index_to_relation))}
        
wrong_role_dic = {key : 0 for key in event_role_relation_to_index}
right_role_dic = {key : 0 for key in event_role_relation_to_index}
not_pre_dic = {key : 0 for key in event_role_relation_to_index}
pre_more_dic = {key : 0 for key in event_role_relation_to_index}
event_not_pre_dic = {key : 0 for key in event_role_relation_to_index}
event_pre_more_dic = {key : 0 for key in event_role_relation_to_index}

#with open('Data/test.json', encoding='utf-8') as f:
#    train = json.load(f)

dic = None
with open('chinese_roberta_wwm_ext/vocab.txt', 'r', encoding='utf-8') as file:
    dic = file.read().split('\n')

class DocEEBasicModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.seq_bio_index_to_cate = ['Null'] + sorted(list(event_role_label_set))
        self.seq_bio_cate_to_index = {self.seq_bio_index_to_cate[i]: i for i in range(len(self.seq_bio_index_to_cate))}

        self.seq_BIO_index_to_tag = ['O'] + [x + '-B' for x in self.seq_bio_index_to_cate[1:]] + [x + '-I' for x in self.seq_bio_index_to_cate[1:]]
        self.seq_BIO_tag_to_index = {self.seq_BIO_index_to_tag[i]: i for i in range(len(self.seq_BIO_index_to_tag))}
        self.pad_token_id = 0
        
        self.b_tag_index, self.i_tag_index, self.one_b_tag_index, self.one_i_tag_index = self.init_bio_tag_index_information()

    def init_bio_tag_index_information(self) -> (set, set, int, int):
        b_tag = []
        i_tag = []
        for tag in self.seq_BIO_index_to_tag:
            if tag.endswith('-B'):
                b_tag.append(tag)
            elif tag.endswith('-I'):
                i_tag.append(tag)
        the_chosen_tag = b_tag[0][:-2]
        the_chosen_b_tag_index = self.seq_BIO_tag_to_index[the_chosen_tag + '-B']
        the_chosen_i_tag_index = self.seq_BIO_tag_to_index[the_chosen_tag + '-I']
        b_tag_index = [self.seq_BIO_tag_to_index[x] for x in b_tag]
        i_tag_index = [self.seq_BIO_tag_to_index[x] for x in i_tag]
        return set(b_tag_index), set(i_tag_index), the_chosen_b_tag_index, the_chosen_i_tag_index

    def get_bio_positions(self, bio_res: list) -> list:
        for x in bio_res:
            sum_b = 0
            for i in self.b_tag_index:
                sum_b += x[i]
            sum_i = 0
            for i in self.i_tag_index:
                sum_i += x[i]
            x[self.one_b_tag_index] = sum_b
            x[self.one_i_tag_index] = sum_i

        bio_index = []
        for x in bio_res:
            index = UtilStructure.find_max_number_index(x)
            bio_index.append(index)

        final_index = []
        for i in range(len(bio_index)):
            if bio_index[i] != -100:
                final_index.append(bio_index[i])
            else:
                final_index.append(self.seq_BIO_tag_to_index['O'])

        for i in range(len(final_index)):
            if final_index[i] in self.i_tag_index:
                final_index[i] = self.one_i_tag_index
            elif final_index[i] in self.b_tag_index:
                final_index[i] = self.one_b_tag_index

        bio_tag = [self.seq_BIO_index_to_tag[x] for x in final_index]
        # for the CLS of the first token, which should be 'O'
        bio_tag[0] = 'O'
        if bio_tag[1][-1] == 'I':
            bio_tag[1] = bio_tag[1][:-1] + 'B'
        position = BasicModel.find_BIO_spans_positions(bio_tag)
        bio_tag = self.validify_BIO_span(bio_tag, position, 'ignore')
        position = BasicModel.find_BIO_spans_positions(bio_tag)
        position = [tuple(pos) for pos in position]
        return position

def measure_event_table_filling(pred_record_mat_list, gold_record_mat_list):
    wrong_event = [False for i in range(len(pred_record_mat_list))]
    
    type_to_index: dict = copy.deepcopy(event_type_type_to_index)
    type_to_index.pop('Null')
    type_to_index = {k: v - 1 for k, v in type_to_index.items()}
    index_to_type = copy.deepcopy(event_type_index_to_type)
    index_to_type = index_to_type[1:]
    index_to_type = {i: index_to_type[i] for i in range(len(index_to_type))}
    event_num = len(type_to_index)
    event_schema = SCHEMA
    event_type_roles_list = []
    event_type_list = []
    for i in range(event_num):
        event_type = index_to_type[i]
        event_type_list.append(event_type)
        roles = event_schema[event_type]
        event_type_roles_list.append((event_type, roles))
    
    index = 0
    for pred_record_mat, gold_record_mat in zip(pred_record_mat_list, gold_record_mat_list):
        for event_idx, (pred_records, gold_records) in enumerate(zip(pred_record_mat, gold_record_mat)):

            roles = event_type_roles_list[event_idx][1]
            if len(gold_records) != 0:
                if len(pred_records) == 0:
                    for gold_record in gold_records:
                        for role_idx, arg_tup in enumerate(gold_record):
                            if arg_tup is not None:
                                not_pre_dic[roles[role_idx]] += 1
                                event_not_pre_dic[roles[role_idx]] += 1
                                wrong_role_dic[roles[role_idx]] += 1
                                wrong_event[index] = True
                else:  
                    pred_records = sorted(pred_records,
                                        key=lambda x: sum(1 for a in x if a is not None),
                                        reverse=True)
                    gold_records = list(gold_records)

                    while len(pred_records) > 0 and len(gold_records) > 0:
                        pred_record = pred_records[0]

                        _tmp_key = lambda gr: sum([1 for pa, ga in zip(pred_record, gr) if pa == ga])
                        best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                        gold_record = gold_records[best_gr_idx]

                        for role_idx, (pred_arg, gold_arg) in enumerate(zip(pred_record, gold_record)):
                            if pred_arg == gold_arg:
                                right_role_dic[roles[role_idx]] += 1
                            else:
                                wrong_role_dic[roles[role_idx]] += 1
                                wrong_event[index] = True
                                if pred_arg == None:
                                    not_pre_dic[roles[role_idx]] += 1
                                elif gold_arg == None:
                                    pre_more_dic[roles[role_idx]] += 1
                        
                        del pred_records[0]
                        del gold_records[best_gr_idx]

                    for pred_record in pred_records:
                        for role_idx, arg_tup in enumerate(pred_record):
                            if arg_tup is not None:
                                wrong_role_dic[roles[role_idx]] += 1
                                pre_more_dic[roles[role_idx]] += 1
                                event_pre_more_dic[roles[role_idx]] += 1
                                wrong_event[index] = True
                    for gold_record in gold_records:
                        for role_idx, arg_tup in enumerate(gold_record):
                            if arg_tup is not None:
                                wrong_role_dic[roles[role_idx]] += 1
                                not_pre_dic[roles[role_idx]] += 1
                                event_not_pre_dic[roles[role_idx]] += 1
                                wrong_event[index] = True
            else:
                if len(pred_records) == 0:
                    pass
                else:
                    for pred_record in pred_records:
                        for role_idx, arg_tup in enumerate(pred_record):
                            if arg_tup is not None:
                                pre_more_dic[roles[role_idx]] += 1
                                wrong_role_dic[roles[role_idx]] += 1
                                event_pre_more_dic[roles[role_idx]] += 1
                                wrong_event[index] = True
        index += 1
    return wrong_event

test_json = UtilData.read_raw_json_file('Data/DuEE_Fin/test.json')
model = DocEEBasicModel()

entitys = []
for index, item in enumerate(row):
    one_sentence = test_json[index][1]['sentences']
    sentence = ''
    length = []
    for word in one_sentence:
        sentence += word
        length.append(len(word))
    
    acc = 0
    before = 0
    cut_num = []
    for one_length in length:
        acc += one_length
        if acc > 512:
            cut_num.append(before)
            acc = 0
        before += one_length
    cut_num.append(len(sentence))
    
    bio_pre = item['BIO_pred']
    entity = model.get_bio_positions(bio_res=bio_pre)

    cut_index = 0
    chin_e = set()
    for e in entity:
        if e[1] - cut_index - 1 <= cut_num[cut_index]:
            start = e[0] - cut_index - 1
            end = e[1] - cut_index - 1
            chin_e.add(sentence[start: end])
        else:
            cut_index += 1
            start = e[0] - cut_index - 1
            end = e[1] - cut_index - 1
            chin_e.add(sentence[start: end])
    entitys.append((len(entity), chin_e))

wrong_event = measure_event_table_filling(pred, gold)
all_num = 0
for key in right_role_dic:
    all_num += right_role_dic[key] + wrong_role_dic[key] 

role_wrong = 0
pre_large = 0
pre_small = 0
nan_to_role = 0
role_to_nan = 0
for key in right_role_dic:
    num = right_role_dic[key] + wrong_role_dic[key] 
    if num == 0:
        continue
    pad = ''
    for i in range(15 - len(key)):
        pad += '一'
    rat = right_role_dic[key] / num  * 100
    wrong = wrong_role_dic[key] - not_pre_dic[key] - pre_more_dic[key]
    print(key,pad, '{:.2f} \t {:.2f} \t'.format(rat, num / all_num * 100), num, '\t', not_pre_dic[key], '\t', pre_more_dic[key], '\t',wrong, '\t',event_not_pre_dic[key],'\t',event_pre_more_dic[key])
    role_wrong += not_pre_dic[key] - event_not_pre_dic[key] + pre_more_dic[key] - event_pre_more_dic[key] + wrong
    pre_small += event_not_pre_dic[key]
    pre_large += event_pre_more_dic[key]
    role_to_nan += not_pre_dic[key] - event_not_pre_dic[key]
    nan_to_role += pre_more_dic[key] - event_pre_more_dic[key]
print('角色预测错误总数:{}\n,事件预测少了总数:{}\n,事件预测多了总数:{}\n,正常角色预测成nan:{}\n,没有角色预测出角色:{}'.format(role_wrong, pre_small, pre_large, role_to_nan, nan_to_role))

             
for one_sentence in pred:
    for one_class in one_sentence:
        for i, one_event in enumerate(one_class):
            new_event = []
            for one_role in one_event:
                if one_role == None:
                    new_event.append(None)
                else:
                    new_role = ''
                    for word in one_role:
                        new_role += dic[word]
                    new_event.append(new_role)
            one_class[i] = new_event

for one_sentence in gold:
    for one_class in one_sentence:
        for i, one_event in enumerate(one_class):
            new_event = []
            for one_role in one_event:
                if one_role == None:
                    new_event.append(None)
                else:
                    new_role = ''
                    for word in one_role:
                        new_role += dic[word]
                    new_event.append(new_role)
            one_class[i] = new_event


event_num_wrong = []
event_class_wrong = []
event_role_wrong = []
one_event_role_wrong = []
for i, label in enumerate(gold):
    for j, one_class in enumerate(label):
        if len(pred[i][j]) != len(one_class):
            if len(pred[i][j]) == 0 or len(one_class) == 0:
                event_class_wrong.append((i, j + 1, one_class, pred[i][j], copy_num[i]))
            else:
                event_num_wrong.append((i, j + 1, one_class, pred[i][j], copy_num[i]))

                for one_event in one_class:
                    have_same = False
                    for pre_evnet in pred[i][j]:             
                        if one_event == pred[i][j]:
                            have_same = True
                    if have_same == False:
                        one_event_role_wrong.append((i, j + 1, one_class, pred[i][j]))
            # break
        else:
            if one_class != pred[i][j] and wrong_event[i]:
                event_role_wrong.append((i, j + 1, one_class, pred[i][j], copy_num[i]))

            for one_event in one_class:
                have_same = False
                for pre_evnet in pred[i][j]:             
                    if one_event == pred[i][j]:
                        have_same = True
                if have_same == False:
                    one_event_role_wrong.append((i, j + 1, one_class, pred[i][j], copy_num[i]))

event_wrong = np.zeros([3,14], dtype=int)
for item in event_class_wrong:
    event_wrong[0][0] += abs(len(item[2]) - len(item[3]))
    event_wrong[0][item[1]] += abs(len(item[2]) - len(item[3]))
for item in event_num_wrong:
    event_wrong[1][0] += abs(len(item[2]) - len(item[3]))
    event_wrong[1][item[1]] += abs(len(item[2]) - len(item[3]))
for item in one_event_role_wrong:
    event_wrong[2][0] += 1
    event_wrong[2][item[1]] += 1
event_sum = np.zeros([1,14], dtype=int)
for item in gold:
    for i, one_class in enumerate(item):
        event_sum[0][i + 1] += len(one_class)
event_sum[0][0] += 1
event_rate = np.array((event_wrong / event_sum)*100, dtype=int) / 10
np.set_printoptions(formatter={'all':lambda x: str(x)},threshold=100)
print(event_wrong)
print(event_rate)

gold_num = np.zeros(30, dtype=int)
pre_num = np.zeros(30, dtype=int)
wrong = {'pre_lage':0,'pre_small':0}
total = 0
for one_gold, one_pre in zip(gold, pred):
    temp = [0,0,0]
    for gold_class, pre_class in zip(one_gold, one_pre):
        temp[0] += len(gold_class)
        temp[1] += len(pre_class)
        if temp[0] < temp[1]:
            wrong['pre_lage'] += 1
            temp[2] += 1
        elif temp[0] > temp[1]:
            wrong['pre_small'] += 1
            temp[2] += 1
        
    if temp[2] != 0:
        total += 1
    gold_num[temp[0]] += 1
    pre_num[temp[1]] += 1
print(pre_num)
print(gold_num)
print(wrong)
print('预测错的文档',total, total / len(gold))
#10,5  271,2  352,2
# pred_record_mat = compute_struct(row[271]['event_pred'], 2)
pass

