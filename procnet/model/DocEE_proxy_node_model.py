import torch
from torch import nn
import torch.nn.functional as F
from procnet.model.basic_model import BasicModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from procnet.data_example.DocEEexample import DocEEDocumentExample
from typing import Dict
from transformers import PreTrainedModel
from torch_geometric.nn import FiLMConv
from procnet.conf.DocEE_conf import DocEEConfig
from procnet.utils.util_structure import UtilStructure
from torch_scatter import scatter_add
# from sklearn.cluster import KMeans
# import numpy as np
from torch.utils.data import RandomSampler
import random
import numpy as np
import copy


def pairwise_cosine(data1, data2):
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)

    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    cosine_dis = cosine.sum(dim=-1).squeeze()
    return cosine_dis


class DocEEBasicModel(BasicModel):
    def __init__(self,
                 preparer: BasicPreparer,):
        super().__init__()
        self.seq_BIO_index_to_tag = preparer.seq_BIO_index_to_tag
        self.seq_BIO_tag_to_index = preparer.seq_BIO_tag_to_index
        self.pad_token_id = preparer.get_auto_tokenizer().pad_token_id

        self.b_tag_index, self.i_tag_index, self.one_b_tag_index, self.one_i_tag_index = self.init_bio_tag_index_information()
        self.language_model: PreTrainedModel = None

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

    def get_bio_positions(self, bio_res: list, input_prob: bool, binary_mode: bool = False, input_id_int=None, ignore_padding_token: bool = True) -> list:
        if binary_mode:
            if input_prob:
                for x in bio_res:
                    sum_b = 0
                    for i in self.b_tag_index:
                        sum_b += x[i]
                    sum_i = 0
                    for i in self.i_tag_index:
                        sum_i += x[i]
                    x[self.one_b_tag_index] = sum_b
                    x[self.one_i_tag_index] = sum_i
        if input_prob:
            bio_index = []
            for x in bio_res:
                index = UtilStructure.find_max_number_index(x)
                bio_index.append(index)
        else:
            bio_index = bio_res

        if ignore_padding_token:
            final_index = []
            before_has_pad = False
            for i in range(len(bio_index)):
                if input_id_int is not None:
                    if input_id_int[i] != self.pad_token_id:
                        final_index.append(bio_index[i])
                        if before_has_pad:
                            raise Exception('get_bio_positions has padding token before other token!')
                    else:
                        before_has_pad = True
                else:
                    if bio_index[i] != -100:
                        final_index.append(bio_index[i])
                        if before_has_pad:
                            raise Exception('get_bio_positions has padding token before other token!')
                    else:
                        before_has_pad = True
        else:
            final_index = []
            for i in range(len(bio_index)):
                if input_id_int is not None:
                    if input_id_int[i] != self.pad_token_id:
                        final_index.append(bio_index[i])
                    else:
                        final_index.append(self.seq_BIO_tag_to_index['O'])
                else:
                    if bio_index[i] != -100:
                        final_index.append(bio_index[i])
                    else:
                        final_index.append(self.seq_BIO_tag_to_index['O'])

        if binary_mode:
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

class DocEEGNNModelHN(nn.Module):
    def __init__(self, node_size, num_relations, dropout_ratio):
        super().__init__()
        self.node_size = node_size
        self.dropout_ratio = dropout_ratio
        self.gcn1 = FiLMConv(in_channels=node_size,
                             out_channels=node_size,
                             num_relations=num_relations,
                             )
        self.linear1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.node_size, self.node_size),
            nn.Dropout(self.dropout_ratio),
        )

    def forward(self, x, edge_index, edge_type):
        x = self.gcn1(x=x, edge_index=edge_index, edge_type=edge_type)
        x = self.linear1(x)
        return x


class DocEEProxyNodeModel(DocEEBasicModel):
    def __init__(self,
                 config: DocEEConfig,
                 preparer: DocEEPreparer,
                 ):
        super().__init__(preparer=preparer)
        self.config = config
        self.device = config.device

        edge_types = ["M-self", "M-M", "C-M", "S-M"]
        self.edge_type_table = {edge_types[i]: i for i in range(len(edge_types))}

        self.dropout_ratio = 0.15
        self.temperature = config.temperature
        self.node_size = config.node_size
        self.null_bio_index = preparer.seq_BIO_tag_to_index['O']
        assert self.null_bio_index == 0
        self.null_event_type_index = preparer.event_type_type_to_index['Null']
        # self.null_event_relation_index = preparer.event_role_relation_to_index['Null']
        self.seq_BIO_index_to_tag = preparer.seq_BIO_index_to_tag
        self.event_type_index_to_type = preparer.event_type_index_to_type
        self.event_type_index_to_type_no_null = [x for x in self.event_type_index_to_type if x != 'Null']
        self.seq_bio_index_to_cate_no_null = preparer.seq_bio_index_to_cate[1:]
        self.num_BIO_tags = len(preparer.seq_BIO_index_to_tag)

        self.pos_bio_ratio_total = preparer.pos_bio_ratio_total
        self.neg_bio_ratio_total = preparer.neg_bio_ratio_total

        self.num_proxy_slot = config.proxy_slot_num
        self.num_BIO_tag = len(preparer.seq_BIO_index_to_tag)
        self.num_event_type = len(preparer.event_type_type_to_index)
        self.num_event_relation = len(preparer.event_role_relation_to_index)

        self.mid_BIO_tag = self.num_BIO_tag // 2 + 1
        for i in range(self.mid_BIO_tag - 1):
            assert self.seq_BIO_index_to_tag[1 + i][:-2] == self.seq_BIO_index_to_tag[self.mid_BIO_tag + i][:-2]
            assert self.seq_bio_index_to_cate_no_null[i] == self.seq_BIO_index_to_tag[1 + i][:-2]

        self.language_model = self.new_bert_model(model_name=config.model_name)
        self.lm_size = self.language_model.config.hidden_size

        # (proxy_slot_num, proxy_size)
        self.lm_bio_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.lm_size, self.lm_size // 4),
            nn.LayerNorm(self.lm_size // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.lm_size // 4, self.num_BIO_tag),
        )
        self.lm_hidden_linear = nn.Sequential(
            nn.Linear(self.lm_size + 1, self.node_size),
            nn.LayerNorm(self.node_size),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )
        self.lm_cls_hidden_linear = nn.Sequential(
            nn.Linear(self.lm_size + 1, self.node_size),
            nn.LayerNorm(self.node_size),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.lm_span_hidden_linear = nn.Sequential(
            nn.Linear(self.lm_size, self.node_size),
            nn.LayerNorm(self.node_size),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )
        self.proxy_linear = nn.Sequential(
            nn.Linear(self.node_size * 2, self.node_size),
            nn.LayerNorm(self.node_size),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
        )
        # self.proxy_attention2 = nn.MultiheadAttention(embed_dim=self.node_size, num_heads=8, dropout=self.dropout_ratio, batch_first=True)
        self.proxy_attention = nn.MultiheadAttention(embed_dim=self.node_size, num_heads=8, dropout=self.dropout_ratio, batch_first=True)

        self.gcn = DocEEGNNModelHN(node_size=self.node_size, num_relations=len(self.edge_type_table), dropout_ratio=self.dropout_ratio)
        self.span_span_attention = nn.MultiheadAttention(embed_dim=self.node_size, num_heads=8, dropout=self.dropout_ratio, batch_first=True)
        self.proxy_slot_event_type_linear = nn.Sequential(
            nn.Linear(self.node_size, self.node_size // 4),
            nn.LayerNorm(self.node_size // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size // 4, self.num_event_type)
        )

        self.proxy_span_attention = nn.MultiheadAttention(embed_dim=self.node_size, num_heads=8, dropout=self.dropout_ratio, batch_first=True)
        self.proxy_rel_attention = nn.MultiheadAttention(embed_dim=self.node_size, num_heads=8, dropout=self.dropout_ratio, batch_first=True)
        self.span_proxy_slot_relation_linear = nn.Sequential(
            nn.Linear(self.node_size + self.node_size, self.node_size // 4),
            nn.LayerNorm(self.node_size // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size // 4, self.num_event_relation)
        )
        self.proxy_exit_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size * 2, 2),
        )
        self.span_span_relation_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size * 2, 2),
        )
        self.rel_pri_linear = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.node_size, 3),
        )

        self.ce_none_reduction_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.ce_normal_loss_fn = nn.CrossEntropyLoss()
        self.ce_sum_reduction_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.mse_loss_fn = nn.MSELoss()

        self.all_event_num=preparer.all_event_num

        pos = torch.arange(7500, dtype=torch.float, device=self.device)
        self.pos = self.compute_pos_embedding(pos)

    def forward(self,
                example: DocEEDocumentExample,
                bios_ids=None,
                use_mix_bio: bool = True,
                ):
        # save = []
        inputs_ids = example.input_ids
        events_labels = example.events_label

        # --- setup ---
        sentence_num = len(inputs_ids)
        input_ids_int = [x.detach().cpu().numpy().tolist() for x in inputs_ids]
        # --- entity record init ---
        loss_bio, position_times, BIO_pred, lm_text_state, span_num = self.pre_entity(sentence_num, inputs_ids, bios_ids, input_ids_int, use_mix_bio)
        records = {'BIO_pred': BIO_pred,'loss_bio': loss_bio.item(),}
        if span_num == 0:
            loss = torch.FloatTensor([0]).to(self.device)
            loss += loss_bio
            if self.gradient_accumulation_steps is not None:
                loss = loss / self.gradient_accumulation_steps
            records.update({'loss': loss.item(), 'event_pred': [], 'error_report': "NodeSpanZero",})
            return loss, records

        # --- init the graph ---
        span_rep, node_span_to_index, text_embedding, cls_embedding = self.compute_span_rep(sentence_num, input_ids_int, position_times, lm_text_state)
        mention_rep, mention_to_index = self.compute_mention_rep(span_rep, node_span_to_index)
        
        proxy = self.init_proxy(mention_rep, cls_embedding, span_rep, text_embedding, example.sentence_cluster_label)
        # span_span_relation_loss  = self.compute_span_span_loss(mention_pair_s, mention_to_index, events_labels)
        # proxy = self.compute_proxy_embed(proxy, span_rep, text_embedding, example.sentence_cluster_label)

        # save.append(cluster_label)
        event_type_logit, proxy_span_relation_logit, proxy_span_tensor, predict_events = self.pre_event(proxy, span_rep, cls_embedding, node_span_to_index)
        records.update({'event_pred': predict_events,})

        loss = torch.FloatTensor([0]).to(self.device)
        if self.training == True:
            # 计算标签
            events_type_labels_tensors_list = []
            events_horizontal_role_labels_tensors_list = []
            gold_cluster_label = []
            used_entity = torch.zeros((len(mention_to_index)), dtype=torch.bool)
            entity_label = []
            for index, event_label in enumerate(events_labels):
                event_type_label_tensor = torch.LongTensor([event_label['EventType']])
                event_relation_label_tensor = torch.zeros((mention_rep.shape[0]), dtype=torch.bool)
                events_horizontal_role_label_tensor = torch.ones((self.num_event_relation,), dtype=torch.long) * -100
                one_cluster_label = torch.zeros((len(mention_to_index)), dtype=torch.bool)
                one_entity_label = torch.zeros((len(mention_to_index), self.num_event_relation), dtype=torch.bool)
                for k, v in event_label.items():
                    if k == 'EventType':
                        continue
                    if k not in mention_to_index:
                        continue
                    for item in v:
                        event_relation_label_tensor[mention_to_index[k]] = True
                        events_horizontal_role_label_tensor[item] = mention_to_index[k]
                        one_entity_label[mention_to_index[k], item] = True
                    one_cluster_label[mention_to_index[k]] = True
                    used_entity[mention_to_index[k]] = True
                
                events_type_labels_tensors_list.append(event_type_label_tensor)
                # 如果这个角色为none，则预测标签是最后一个实体数字(cls)
                events_horizontal_role_label_tensor[example.none_role_label[index]] = mention_rep.shape[0]
                events_horizontal_role_labels_tensors_list.append(events_horizontal_role_label_tensor)
                gold_cluster_label.append(one_cluster_label.unsqueeze(0))
                entity_label.append(one_entity_label.unsqueeze(0))

            gold_cluster_label = torch.cat(gold_cluster_label, dim=0)
            entity_label = torch.cat(entity_label)
            events_type_labels_tensors = torch.cat(events_type_labels_tensors_list, dim=0).to(self.device)
            events_horizontal_role_labels_tensors = torch.cat(events_horizontal_role_labels_tensors_list).reshape(-1, self.num_event_relation).to(self.device)
            
            # 计算null对应proxy的损失
            num_all_proxy_slot = proxy.shape[0]
            null_event_type_label = torch.LongTensor([self.null_event_type_index]).to(self.device).expand(num_all_proxy_slot)
            null_event_type_losses = self.ce_none_reduction_loss_fn(event_type_logit, null_event_type_label)
            
            null_event_role_label = torch.LongTensor([mention_rep.shape[0]]).to(self.device).expand(num_all_proxy_slot * self.num_event_relation)
            null_event_role_losses = self.ce_none_reduction_loss_fn(proxy_span_relation_logit.reshape(-1, proxy_span_relation_logit.shape[2]), null_event_role_label).view(num_all_proxy_slot, self.num_event_relation)
            null_event_role_losses = torch.mean(null_event_role_losses, dim=-1)

            null_loss_matrix = null_event_type_losses +  null_event_role_losses
           
            # 标签对齐
            mention_times = torch.sum(gold_cluster_label, dim=0)
            mention_times_rate = 1 / mention_times
            mention_times_rate[mention_times == 0] = 0
            score = mention_times_rate.unsqueeze(0).repeat(gold_cluster_label.shape[0], 1)
            score[~gold_cluster_label] = -1
            max_index = torch.argmax(score, dim=1)

            proxy_index = torch.arange(0, proxy.shape[0])
            proxy_to_index = {}
            for index, item in enumerate(score):
                key = tuple(proxy_index[item == item[max_index[index]]].numpy().tolist())
                if key in proxy_to_index:
                    proxy_to_index[key].append(index)
                else:
                    proxy_to_index[key] = [index]

            # 计算损失
            event_loss = torch.FloatTensor([0]).to(self.device)
            null_loss = torch.FloatTensor([0]).to(self.device)
            other_proxy = torch.ones(null_loss_matrix.shape[0], device=self.device, dtype=torch.bool)
            used_proxy = -torch.ones(null_loss_matrix.shape[0], device=self.device, dtype=torch.long)
            evnet_proxy = torch.zeros((len(events_labels), proxy_span_tensor.shape[1], proxy_span_tensor.shape[2]), device=self.device)
            not_match_event = []
            for proxy_index, event_index in proxy_to_index.items():
                # 删除先前用过的proxy
                temp = []
                for item in proxy_index:
                    if used_proxy[item] != 1:
                        temp.append(item)
                proxy_index = tuple(temp)

                if len(proxy_index) == 0:
                    not_match_event.extend(event_index)
                    continue
                if len(proxy_index) < len(event_index):
                    event_index = event_index[:len(proxy_index)]
                    not_match_event.extend(event_index[len(proxy_index):])

                event_index_tensor = torch.tensor(event_index, device=self.device)
                event_num = len(event_index)
                
                proxy_index_tensor = torch.tensor(proxy_index, device=self.device)
                proxy_num = len(proxy_index)

                pad_num = proxy_num - event_num
                other_proxy[proxy_index_tensor] = False

                # 类型损失
                type_label =events_type_labels_tensors[event_index_tensor].unsqueeze(1).expand(-1, proxy_num)
                pre_type_logit = event_type_logit[proxy_index_tensor].unsqueeze(0).expand(event_num, -1, -1)

                event_type_losses = self.ce_none_reduction_loss_fn(pre_type_logit.reshape(-1, self.num_event_type), type_label.reshape(-1)).view(event_num, proxy_num)
                # 角色填充损失
                events_horizontal_role_label = events_horizontal_role_labels_tensors[event_index_tensor].reshape(event_num, self.num_event_relation).unsqueeze(1).expand(-1, proxy_num, -1)
                event_horizontal_role_logit = proxy_span_relation_logit[proxy_index_tensor].reshape(proxy_num, self.num_event_relation, -1).unsqueeze(0).expand(event_num, -1, -1, -1)
                event_horizontal_role_losses = self.ce_none_reduction_loss_fn(event_horizontal_role_logit.reshape(-1, event_horizontal_role_logit.shape[-1]), events_horizontal_role_label.reshape(-1)).view(event_num, proxy_num, self.num_event_relation)
                not_ignore_mask = event_horizontal_role_losses != 0
                not_ignore_num = torch.sum(not_ignore_mask, dim=-1)
                event_horizontal_role_losses = torch.sum(event_horizontal_role_losses, dim=-1) / not_ignore_num

                event_loss_matrix = event_type_losses + event_horizontal_role_losses
                if pad_num > 0 :
                    event_loss_matrix = torch.cat([event_loss_matrix, null_loss_matrix[proxy_index_tensor].unsqueeze(0).expand(pad_num, -1)])
                order_dict, min_order_loss = self.event_ordering(event_loss_matrix.detach().cpu().numpy(), maximize=False)

                for k, v in order_dict.items():
                    if k < event_num:
                        event_loss += event_loss_matrix[k, v]
                        used_proxy[proxy_index[v]] = 1
                        evnet_proxy[event_index[k]] = proxy_span_tensor[proxy_index[v]]
                    else:
                        null_loss += event_loss_matrix[k, v]
            
            # 匹配未匹配的
            if len(not_match_event) != 0:
                not_match_event = torch.tensor(not_match_event)
                other_proxy_index = torch.arange(0, other_proxy.shape[0])[other_proxy]
                
                proxy_num = other_proxy_index.shape[0]
                event_num = not_match_event.shape[0]
                pad_num = proxy_num - event_num

                type_label =events_type_labels_tensors[not_match_event].unsqueeze(1).expand(-1, proxy_num)
                pre_type_logit = event_type_logit[other_proxy_index].unsqueeze(0).expand(event_num, -1, -1)
                event_type_losses = self.ce_none_reduction_loss_fn(pre_type_logit.reshape(-1, self.num_event_type), type_label.reshape(-1)).view(event_num, proxy_num)

                events_horizontal_role_label = events_horizontal_role_labels_tensors[not_match_event].reshape(event_num, self.num_event_relation).unsqueeze(1).expand(-1, proxy_num, -1)
                event_horizontal_role_logit = proxy_span_relation_logit[other_proxy_index].reshape(proxy_num, self.num_event_relation, -1).unsqueeze(0).expand(event_num, -1, -1, -1)
                event_horizontal_role_losses = self.ce_none_reduction_loss_fn(event_horizontal_role_logit.reshape(-1, event_horizontal_role_logit.shape[-1]), events_horizontal_role_label.reshape(-1)).view(event_num, proxy_num, self.num_event_relation)
                not_ignore_mask = event_horizontal_role_losses != 0
                not_ignore_num = torch.sum(not_ignore_mask, dim=-1)
                event_horizontal_role_losses = torch.sum(event_horizontal_role_losses, dim=-1) / not_ignore_num

                event_loss_matrix = event_type_losses + event_horizontal_role_losses
                if pad_num > 0 :
                    event_loss_matrix = torch.cat([event_loss_matrix, null_loss_matrix[other_proxy_index].unsqueeze(0).expand(pad_num, -1)])
                order_dict, min_order_loss = self.event_ordering(event_loss_matrix.detach().cpu().numpy(), maximize=False)

                for k, v in order_dict.items():
                    if k < event_num:
                        event_loss += event_loss_matrix[k, v]
                        used_proxy[other_proxy_index[v]] = 1
                        other_proxy[other_proxy_index[v]] = False

                        evnet_proxy[not_match_event[k]] = proxy_span_tensor[other_proxy_index[v]]
                    else:
                        null_loss += event_loss_matrix[k, v]
            
            # 其他无用proxy
            not_used_loss = null_loss_matrix[other_proxy]
            if not_used_loss.shape[0] != 0:
                null_loss += torch.sum(not_used_loss)
            
            pos_rate = len(events_labels) / proxy.shape[0]
            null_rate = 1 - pos_rate
            # mean_event_type_loss = (event_type_loss * null_rate + null_type_loss * pos_rate) / proxy.shape[0]
            # mean_event_role_loss = (event_role_loss * null_rate + null_role_loss * pos_rate) / proxy.shape[0]
            mean_event_loss = event_loss * null_rate + null_loss * pos_rate

            # event_exit_loss = self.compute_event_exit_loss(used_proxy, text_embedding, cls_embedding, proxy, proxy_span_tensor, events_labels)
            contrastive_loss = self.compute_role_contrastive_loss(evnet_proxy, entity_label, used_entity)
            # span_span_loss = self.compute_span_span_loss(mention_rep, mention_to_index, events_labels)
            role_pri_loss = self.compute_role_pri_loss(mention_times, proxy)

            # loss += mean_event_loss + loss_bio + event_exit_loss + 0.1 * contrastive_loss
            loss += mean_event_loss + 0.01 * loss_bio + 0.01 * role_pri_loss + 0.01 * contrastive_loss
            """
            A_gold = events_type_labels_tensors
            A_pre = torch.argmax(event_type_logit, dim=-1)
            proxy_to_index
            print(torch.sum(A_gold != 0).item(), torch.sum(A_pre != 0).item())
            """

            # print('Event:{:.4f}  Bio:{:.4f}  Total:{:.4f}'
            #       .format(mean_event_loss.item(), loss_bio.item(), 
            #               loss.item()), end=' ')

            if self.gradient_accumulation_steps is not None:
                loss = loss / self.gradient_accumulation_steps

        records.update({'loss': loss.item(),'error_report': '',})
        return loss, records
    
    def pre_entity(self, sentence_num, inputs_ids, bios_ids, input_ids_int, use_mix_bio):
        bio_probs = []
        position_times = []
        lm_text_state = []
        loss_bio = torch.FloatTensor([0]).to(self.device)
        span_num = 0
        # --- sequence labeling ---
        for time_step in range(sentence_num):
            # (1, seq_length, )
            input_ids = inputs_ids[time_step].unsqueeze(0).to(self.device)
            # (1, seq_length, )
            bio_ids = bios_ids[time_step].unsqueeze(0) if bios_ids is not None else None
            # cpu [101, 102, 103]
            input_id_int = input_ids_int[time_step]
            # --- all sentence to LM for BIO ---
            lm_res: BaseModelOutputWithPoolingAndCrossAttentions = self.language_model(input_ids=input_ids)
            # (1, seq_length, lm_size)
            lm_last_hidden_states = lm_res.last_hidden_state
            lm_text_state.append(lm_last_hidden_states.squeeze(0))

            # (1, seq_length, bio_tags_size)
            lm_logit = self.lm_bio_linear(lm_last_hidden_states)
            if bio_ids is None:
                one_loss_bio = torch.FloatTensor([0]).to(self.device)
            else:
                raw_loss_bio = self.ce_none_reduction_loss_fn(lm_logit.view(-1, self.num_BIO_tag), bio_ids.view(-1, ))
                bio_is_o = bio_ids.squeeze(0) == self.null_bio_index
                bio_not_o = bio_ids.squeeze(0) != self.null_bio_index
                loss_o = torch.sum(raw_loss_bio * bio_is_o)
                loss_bi = torch.sum(raw_loss_bio * bio_not_o)
                one_loss_bio = loss_o * self.pos_bio_ratio_total + loss_bi * self.neg_bio_ratio_total
                # one_loss_bio = one_loss_bio * 0.01
            loss_bio += one_loss_bio

            # (1, seq_length, bio_tags_size)
            bio_prob = F.softmax(lm_logit, dim=2)
            # cpu probability of bio.  [[[0.1, 0.9], [0.5, 0.4 ]], ]
            bio_result = bio_prob.squeeze(0).detach().cpu().numpy().tolist()
            # cpu positions. [[[start, end], [start, end]], ]
            bio_probs.append(bio_prob.squeeze(0).detach().cpu())
            pred_position = self.get_bio_positions(bio_res=bio_result, input_id_int=input_id_int, input_prob=True, binary_mode=True, ignore_padding_token=False)

            # --- bio post process, bio tag and bio position---
            if bios_ids is None:
                ans_position = []
            else:
                # cpu bio ids [0, 0, 1, 5, 6]
                bio_ids = bio_ids.squeeze(0).detach().cpu().numpy().tolist()
                # cpu positions. [[start, end], [start, end]]
                ans_position = self.get_bio_positions(bio_res=bio_ids, input_id_int=input_id_int, input_prob=False, binary_mode=False, ignore_padding_token=False)

            # --- train the model on the predicted span ---
            if bio_ids is not None:
                if use_mix_bio:
                    position = pred_position + ans_position
                else:
                    position = ans_position
            else:
                position = pred_position
            position_times.append(position)
            span_num += len(position)

        BIO_pred = torch.cat(bio_probs, dim=0).view(-1, self.num_BIO_tags).detach().cpu().numpy().tolist()

        return loss_bio, position_times, BIO_pred, lm_text_state, span_num

    def compute_span_rep(self, sentence_num, input_ids_int, position_times, lm_text_state):
        node_span_to_index = {}

        span_rep = []
        text_embedding = []
        cls = []
        current_index = 0
        current_length = 0
        for time_step in range(sentence_num):
            input_id_int = input_ids_int[time_step]
            position = position_times[time_step]
            lm_hidden_state = lm_text_state[time_step]

            lm_hidden_state = self.lm_span_hidden_linear(lm_hidden_state + self.pos[current_length:current_length + lm_hidden_state.shape[0]])
            cls.append(lm_hidden_state[0])
            current_length += lm_hidden_state.shape[0]
            text_embedding.append(lm_hidden_state)

            for pos in position:
                span = tuple(input_id_int[pos[0]:pos[1]])
                # (span_length, node_size)
                span_hidden_state = lm_hidden_state[pos[0]:pos[1]]
                # (1, node_size)
                span_state = torch.mean(span_hidden_state, dim=0, keepdim=True)
                # --- span node ---
                span_rep.append(span_state)
                if span not in node_span_to_index:
                    node_span_to_index[span] = [current_index]
                else:
                    node_span_to_index[span].append(current_index)
                current_index += 1

        span_rep = torch.cat(span_rep, dim=0)
        # text_embedding = torch.cat(text_embedding)
        cls = torch.cat(cls).reshape(-1, self.node_size)

        return span_rep, node_span_to_index, text_embedding, cls
    
    def compute_mention_rep(self, span_rep, node_span_to_index):
        # lm_span_hidden_logit = self.lm_span_hidden_linear(torch.cat([span_rep, pos_embeddings[:span_rep.shape[0]].to(self.device)], dim=1))
        mention_rep = []
        mention_to_index = {}
        index = 0
        for key, value in node_span_to_index.items():
            mention_rep.append(torch.mean(span_rep[value], dim=0, keepdim=True))
            mention_to_index[key] = index
            index += 1
        mention_rep = torch.cat(mention_rep)
        return mention_rep, mention_to_index

    def init_proxy(self, mention_rep, cls_embedding, span_rep, lm_hidden_state_times, sentence_cluster_label):
        cls_proxy, _ = self.proxy_attention(query=mention_rep, key=cls_embedding, value=cls_embedding)
        proxy = self.proxy_linear(torch.cat([cls_proxy, mention_rep], dim=1))

        return  proxy
    
    def pre_event(self, proxy, span_rep, cls_embedding, node_span_to_index):
        cls_span = torch.cat([cls_embedding, span_rep], dim=0)
        cls_span_tensor, _ = self.span_span_attention(query=cls_span, key=cls_span, value=cls_span)

        cls_part = cls_span_tensor[:cls_embedding.shape[0]]
        span_part = cls_span_tensor[cls_embedding.shape[0]:]

        # --- event relation logit --
        span_num = len(node_span_to_index)
        max_individual_span_num = max([len(v) for k, v in node_span_to_index.items()] + [cls_embedding.shape[0]])
        span_tensor_index_to_span = list(node_span_to_index.keys())
        span_tensor_span_to_index = {span_tensor_index_to_span[i]: i for i in range(span_num)}
        span_tensor = torch.zeros((span_num + 1, max_individual_span_num, self.node_size), dtype=torch.float, device=self.device)
        span_tensor_mask = torch.ones((span_num + 1, max_individual_span_num), dtype=torch.bool, device=self.device)
        for span, tensor_index in span_tensor_span_to_index.items():
            count = -1
            for span_state_index in node_span_to_index[span]:
                count += 1
                span_state = span_part[span_state_index]
                span_tensor[tensor_index, count] = span_state
                span_tensor_mask[tensor_index, count] = False
        for i in range(cls_embedding.shape[0]):
            span_tensor[-1, i] = cls_part[i]
            span_tensor_mask[-1, i] = False

        proxy_slot_expand = proxy.unsqueeze(0).expand(span_num + 1, proxy.shape[0], self.node_size)
        span_tensor, _ = self.proxy_span_attention(query=proxy_slot_expand, key=span_tensor, value=span_tensor, key_padding_mask=span_tensor_mask)
        proxy_span_tensor = torch.cat([proxy_slot_expand, span_tensor], dim=2).transpose(0, 1)
        proxy_span_relation_logit = self.span_proxy_slot_relation_linear(proxy_span_tensor).transpose(1, 2)

        # --- event type logit ---
        event_type_logit = self.proxy_slot_event_type_linear(proxy)
        
        # --- pack probability result ---
        event_type_prob = event_type_logit.detach().cpu().numpy().tolist()
        event_relation_prob = proxy_span_relation_logit.detach().cpu().numpy().tolist()
        predict_events = {'type': event_type_prob, 'rel': event_relation_prob, 'index': span_tensor_index_to_span}

        return event_type_logit, proxy_span_relation_logit, proxy_span_tensor, predict_events
    
    def compute_role_contrastive_loss(self, evnet_proxy, entity_label, used_entity):
        loss = torch.FloatTensor([0]).to(self.device)
        entity_label = entity_label.to(self.device)
        used_entity = used_entity.to(self.device)
        event_num = entity_label.shape[0]

        if event_num != 1:
            all_same_mask = torch.sum(entity_label.unsqueeze(0).expand(event_num, -1, -1, -1) != entity_label.unsqueeze(1).expand(-1, event_num, -1, -1), dim=-1) == 0
            evnet_proxy = evnet_proxy[:, :-1, :] + 1e-20
            num = 0
            for i in range(event_num):
                for j in range(i):  
                    num += 1
                    event_A = evnet_proxy[i]
                    event_B = evnet_proxy[j]

                    same = all_same_mask[i, j] & used_entity
                    diff = ~all_same_mask[i, j] & used_entity

                    same_role_A = event_A[same] 
                    diff_role_A = event_A[diff] 
                    same_role_B = event_B[same]
                    diff_role_B = event_B[diff] 

                    if same_role_A.shape[0] == 0:
                        same_sim = torch.exp(torch.ones(1, device=self.device) / self.temperature)
                    elif same_role_A.shape[0] == 1:
                        same_sim = torch.exp(pairwise_cosine(same_role_A, same_role_B) / self.temperature)
                    else:
                        mask = torch.triu(torch.ones((same_role_A.shape[0], same_role_A.shape[0]), dtype=torch.bool, device=self.device), diagonal=1)
                        same_sim = torch.sum(torch.exp(pairwise_cosine(same_role_A, same_role_B)[mask] / self.temperature))

                    if diff_role_A.shape[0] == 0:
                        diff_sim = torch.exp(-torch.ones(1, device=self.device) / self.temperature)
                    elif diff_role_A.shape[0] == 1:
                        diff_sim = torch.exp(pairwise_cosine(diff_role_A, diff_role_B) / self.temperature)
                    else:
                        mask = torch.triu(torch.ones((diff_role_A.shape[0], diff_role_A.shape[0]), dtype=torch.bool, device=self.device), diagonal=1)
                        diff_sim = torch.sum(torch.exp(pairwise_cosine(diff_role_A, diff_role_B)[mask] / self.temperature))

                    loss -= torch.log(same_sim / (same_sim + diff_sim))

            loss = loss / num

        return loss

    def compute_role_pri_loss(self, mention_times, proxy):
        mention_times[mention_times > 1] = 2
        pri_logit = self.rel_pri_linear(proxy)
        loss = self.ce_normal_loss_fn(pri_logit, mention_times.to(self.device))
        return loss

    def compute_pos_embedding(self, pos):
        dim = self.lm_size
        base = torch.tensor(10000, device=self.device)

        indices = torch.arange(0, dim // 2, dtype=torch.float)
        indices = torch.pow(base, (-2 * indices / dim).to(self.device))
        pos_embeddings = torch.einsum('...,d->...d', pos, indices)
        pos_embeddings = torch.stack([torch.sin(pos_embeddings), torch.cos(pos_embeddings)], axis=-1)
        pos_embeddings = torch.flatten(pos_embeddings, -2)

        return pos_embeddings
    