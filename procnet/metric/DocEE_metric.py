from typing import List
import copy
from procnet.metric.basic_metric import BasicMetric
from procnet.data_preparer.basic_preparer import BasicPreparer
import time
from procnet.dee import dee_metric
from procnet.utils.util_structure import UtilStructure
import torch


class DocEEMetric(BasicMetric):
    def __init__(self,
                 preparer: BasicPreparer,):
        super(DocEEMetric, self).__init__(preparer=preparer)
        self.event_schema_index = preparer.event_schema_index
        self.event_type_type_to_index = preparer.event_type_type_to_index
        self.event_type_index_to_type = preparer.event_type_index_to_type
        self.event_role_relation_to_index = preparer.event_role_relation_to_index
        self.event_role_index_to_relation = preparer.event_role_index_to_relation
        self.seq_BIO_index_to_tag = preparer.seq_BIO_index_to_tag
        self.event_schema = preparer.SCHEMA
        self.event_null_type_index = preparer.event_type_type_to_index['Null']
        self.event_null_relation_index = preparer.event_role_relation_to_index['Null']

    def the_score_fn(self, results: List[dict]):
        # loss
        total_num = len(results)
        mean_loss = sum([x['loss'] for x in results]) / total_num
        loss_to_print = "Loss = {:.4f}, ".format(mean_loss)
        # bio
        bio_ans = [x['BIO_ans'] for x in results]
        bio_pred = [x['BIO_pred'] for x in results]
        bio_to_print, bio_score_results = self.bio_score_fn(bio_ans=bio_ans, bio_pred=bio_pred)
        # event
        if 'event_ans' in results[0] and 'event_pred' in results[0]:
            # events_ans: [[{'EventType': 3, (104, 105, 106): 4, 'entity': 7}, {}], [{}, {}]]
            # events_pred: [[{'EventType': [0.1, 0.3, 0.5], '(3, 4, 103)': [0.2, 0.4 0.2]}, {}, {}], [], []]
            event_ans_all = [x['event_ans'] for x in results]
            event_pred_all = [x['event_pred'] for x in results]
            event_ans_single = []
            event_pred_single = []
            event_ans_multi = []
            event_pred_multi = []
            assert len(event_ans_all) == len(event_pred_all)
            for i in range(len(event_ans_all)):
                ea = event_ans_all[i]
                ep = event_pred_all[i]
                if len(ea) <= 1:
                    event_ans_single.append(ea)
                    event_pred_single.append(ep)
                else:
                    event_ans_multi.append(ea)
                    event_pred_multi.append(ep)

            all_dee_to_print = []
            all_dee_score_results = []
            for event_ans, event_pred in zip([event_ans_all, event_ans_single, event_ans_multi], [event_pred_all, event_pred_single, event_pred_multi]):
                dee_to_print, dee_score_results = self.dee_score_fn(event_ans, event_pred)
                all_dee_to_print.append(dee_to_print)
                all_dee_score_results.append(dee_score_results)
            # dee_to_print = "All event:" + all_dee_to_print[0] + "\nSingle Event:" + all_dee_to_print[1] + "\nMulti Event:" + all_dee_to_print[2]
            dee_to_print = all_dee_to_print[0]
            dee_score_results = {'all_event': all_dee_score_results[0],
                                 'single_event': all_dee_score_results[1],
                                 'multi_event': all_dee_score_results[2],
                                 }
        else:
            dee_to_print, dee_score_results = "", {}

        to_print = loss_to_print + '\n' + dee_to_print
        final_score_results = {
                               'loss': mean_loss,
                               'bio': bio_score_results,
                               'event': dee_score_results,
                               }
        return to_print, final_score_results

    def dee_score_fn(self, events_ans: List[List[dict]], events_pred: List[List[dict]]):
        start_time = time.time()
        type_to_index: dict = copy.deepcopy(self.event_type_type_to_index)
        type_to_index.pop('Null')
        type_to_index = {k: v - 1 for k, v in type_to_index.items()}
        index_to_type = copy.deepcopy(self.event_type_index_to_type)
        index_to_type = index_to_type[1:]
        index_to_type = {i: index_to_type[i] for i in range(len(index_to_type))}
        role_to_index: dict = self.event_role_relation_to_index

        event_num = len(type_to_index)

        event_schema = self.event_schema
        event_type_roles_list = []
        event_type_list = []
        for i in range(event_num):
            event_type = index_to_type[i]
            event_type_list.append(event_type)
            roles = event_schema[event_type]
            event_type_roles_list.append((event_type, roles))

        gold_record_mat_list = []
        for event_ans in events_ans:
            gold_record_mat = [[] for _ in range(event_num)]
            for e_ans in event_ans:
                event_type = self.event_type_index_to_type[e_ans['EventType']]
                event_type_id = type_to_index[event_type]

                roles_tuple = []
                for i in range(len(event_schema[event_type])):
                    role_name = event_schema[event_type][i]
                    role_index = role_to_index[role_name]
                   
                    signal = 1
                    for key in e_ans:
                        if key == 'EventType':
                            continue

                        if role_index in e_ans[key]:
                            roles_tuple.append(key)
                            break

                        signal += 1
                    
                    if signal == len(e_ans):
                        roles_tuple.append(None)

                roles_tuple = tuple(roles_tuple)
                gold_record_mat[event_type_id].append(roles_tuple)
            gold_record_mat_list.append(gold_record_mat)

        pred_record_mat_list = []
        copy_num = []
        for event_pred in events_pred:
            pred_record_mat = [[] for _ in range(event_num)]

            if len(event_pred) == 0:
                pred_record_mat_list.append(pred_record_mat)
                continue

            pre_num = len(event_pred['type'])
            for event_index in range(pre_num):
                event_type = UtilStructure.find_max_number_index(event_pred['type'][event_index])
                event_type = self.event_type_index_to_type[event_type]
                if event_type == 'Null':
                    continue
                event_type_id = type_to_index[event_type]

                roles_dict = []
                for one_event_rel in event_pred['rel'][event_index]:
                    max_p, index = UtilStructure.find_max_and_number_index(one_event_rel)
                    if index == len(one_event_rel) - 1:
                        roles_dict.append(None)
                    else:
                        roles_dict.append(event_pred['index'][index])

                roles_tuple = []
                for i in range(len(event_schema[event_type])):
                    role_name = event_schema[event_type][i]
                    role_index = role_to_index[role_name]
                    roles_tuple.append(roles_dict[role_index])

                roles_tuple = tuple(roles_tuple)
                pred_record_mat[event_type_id].append(roles_tuple)

            # 去重
            before_num = 0
            after_num = 0
            for index, one_class in enumerate(pred_record_mat):
                temp = set()
                for one_event in one_class:
                    temp.add(one_event)
                before_num += len(pred_record_mat[index])
                pred_record_mat[index] = list(temp)
                after_num += len(pred_record_mat[index])

            copy_num.append(before_num - after_num)
            pred_record_mat_list.append(pred_record_mat)

        score_results = dee_metric.measure_event_table_filling(pred_record_mat_list, gold_record_mat_list, event_type_roles_list, event_type_list)

        used_time = (time.time() - start_time) / 60
        score_results['used_time'] = used_time

        to_print = "dee_metric: Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f}, ".format(
            score_results['micro_precision'], score_results['micro_recall'], score_results['micro_f1'],
        )
        if len(events_ans) >= 1000:
            torch.save({'pred':pred_record_mat_list, 'gold':gold_record_mat_list, 'copy_num':copy_num}, 'all.pt')
        return to_print, score_results
