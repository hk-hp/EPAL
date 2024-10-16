import torch
import torch
A = torch.ones(5, 512, dtype=torch.long)
B = A.T
C = A @ B
C = C.float()
pre_cluster_label = []
gold_re_label = []
gold_cluster_label= []
pre_re_label = []
pre_re_score = []
pre_eve_type = []
A_event_type_losses = []
A_event_role_loss=[]
A_total_event_num_loss = []
A_span_span_relation_loss= []
A_role_compare_loss = []
A_loss = []
dic = []
for i in range(10, 21):
    file = torch.load(str(i))
    pre_cluster_label.append(file[0])
    gold_re_label.append(file[1])
    pre_eve_type.append(file[2])
    pre_re_label.append(file[3])
    pre_re_score.append(file[4][0])
    A_event_type_losses.append(file[5])
    A_event_role_loss.append(file[6])
    A_total_event_num_loss.append(file[7])
    A_span_span_relation_loss.append(file[8])
    A_role_compare_loss.append(file[9])
    A_loss.append(file[10])
    dic.append(file[11])
    gold_cluster_label.append(file[12])
pass