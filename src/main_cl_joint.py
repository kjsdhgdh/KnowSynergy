import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from cl_model import KnowSynergyCL, BioEncoder, HgnnEncoder, Decoder
from sklearn.model_selection import KFold
import os
import glob
import sys
import logging
import argparse
import warnings
import random
from drug_util import GraphDataset, collate
from utils import metrics_graph, set_seed_all
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from process_data import getData

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
sys.path.append('..')

def load_data(dataset):
    cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy, DDI= getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)
    DDI = torch.from_numpy(DDI).to(device)
    threshold = 30
    for row in synergy:
        row[3] = 1 if row[3] >= threshold else 0

    drug_sim_matrix, cline_sim_matrix = get_sim_mat(drug_smiles_fea, np.array(gene_data, dtype='float32'))

    return drug_fea, cline_fea, synergy, drug_sim_matrix, cline_sim_matrix, DDI

def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    # -----split synergy into 5CV,test set
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    # --CV set
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    # --test set
    synergy_test = np.concatenate((np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    # np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(synergy_test).to(device)
    return synergy_cv_data, test_ind, test_label

def get_sim_mat(drug_fea, cline_fea):
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(cline_fea))
    return torch.from_numpy(drug_sim_matrix).type(torch.FloatTensor).to(device), torch.from_numpy(
        cline_sim_matrix).type(torch.FloatTensor).to(device)

def drop_hyperedges(edge_data, ratio, seed=None):
    hyperedge_num = int(len(edge_data))
    permute_num = int(hyperedge_num * ratio)
    if seed is not None:
        np.random.seed(seed)
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_data_aug=np.delete(edge_data,edge_remove_index,axis=0)
    synergy_edge = edge_data_aug.reshape(1, -1)
    index_num = np.expand_dims(np.arange(len(edge_data_aug)), axis=-1)
    synergy_num = np.concatenate((index_num, index_num, index_num), axis=1)
    synergy_num = np.array(synergy_num).reshape(1, -1)
    synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
    synergy_graph= torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)
    return synergy_graph

def train_cl(drug_set, cline_set, edge_data, index, label_train, alpha):
    synergy_graph = drop_hyperedges(edge_data, ratio=0)
    synergy_graph1 = drop_hyperedges(edge_data, ratio=args.aug_ratio, seed=123)
    synergy_graph2 = drop_hyperedges(edge_data, ratio=args.aug_ratio, seed=231)
    cl_model.train()
    cl_optimizer.zero_grad()
    for batch, (drug, cline) in enumerate(zip(drug_set, cline_set)):
        drug = drug.to(device)
        pred, project_embed = cl_model(drug.x, drug.edge_index, drug.batch, cline[0], synergy_graph, 
                                       index[:, 0], index[:, 1], index[:, 2])
        _, project_embed1 = cl_model(drug.x, drug.edge_index, drug.batch, cline[0], synergy_graph1, 
                                         index[:, 0], index[:, 1], index[:, 2])
        _, project_embed2 = cl_model(drug.x, drug.edge_index, drug.batch, cline[0], synergy_graph2, 
                                         index[:, 0], index[:, 1], index[:, 2])
        
        input_list = [project_embed, project_embed1, project_embed2]
        input1, input2 = random.choices(input_list, k=2)

        drug1,cline1 = input1[:drug_num], input1[drug_num:]
        drug2,cline2 = input2[:drug_num], input2[drug_num:]

        cl_loss = cl_model.loss(drug1,cline1,drug2,cline2,DDI, cline_sim_mat)   
        pred_loss = loss_func(pred, label_train)

        loss_train = (1 - alpha) * pred_loss + alpha * cl_loss
        loss_train.backward()

        cl_optimizer.step()

        auc_train, aupr_train, f1_train, acc_train = metrics_graph(label_train.cpu().detach().numpy(),pred.cpu().detach().numpy())

        # logger.info('fold: {:d}, '.format(fold_num) + 'cl_loss: {:.6f}, pred_loss: {:.6f}'.format(cl_loss.item(), pred_loss.item()))
        return [auc_train, aupr_train, f1_train, acc_train], loss_train, cl_loss, pred_loss

def test_cl(drug_set, cline_set, synergy_adj, index, label):
    cl_model.eval()
    with torch.no_grad():
        for batch, (drug, cline) in enumerate(zip(drug_set, cline_set)):
            drug = drug.to(device)
            pred, _ = cl_model(drug.x, drug.edge_index, drug.batch, cline[0], synergy_adj,
                                            index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        auc_test, aupr_test, f1_test, acc_test = metrics_graph(label.cpu().detach().numpy(),
                                                               pred.cpu().detach().numpy())
        return [auc_test, aupr_test, f1_test, acc_test], loss.item(), pred.cpu().detach().numpy()

if __name__ == '__main__':
    # python main_v4.py --cv_mode=2 --save=result_cls_debug --epochs=1
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--exp', type=str, default="KnowSynerge")
    parser.add_argument('--cv_mode', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="ALMANAC")# or ONEIL
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default="cls_result")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    args = parser.parse_args()

    dataset_name = args.dataset
    seed = args.seed
    cv_mode = args.cv_mode
    weight_decay = args.weight_decay
    epochs = args.epochs
    learning_rate = args.lr
    alpha = args.alpha
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    set_seed_all(seed)

    if dataset_name == 'ALMANAC':
        drug_num = 87 
        cline_num = 55 
    else:
        drug_num = 38
        cline_num = 32
    #-----Dataset load 
    drug_feature, cline_feature, synergy_data, drug_sim_mat, cline_sim_mat, DDI = load_data(dataset_name)
  
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                                collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
    cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),
                                batch_size=len(cline_feature), shuffle=False)


    dir = '{}/cv_mode_{}/'.format(args.save,cv_mode)
    if not os.path.exists(dir):
        os.makedirs(dir)      
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)    
    path = dir + dataset_name + '_' + str(cv_mode) + '_'
    logger.info(args)

    # ---cl_model_build
    logger.info('KnowSynerge')
    cl_model = KnowSynergyCL(BioEncoder(dim_drug=75, dim_cellline=cline_feature.shape[-1], output=100),
                                HgnnEncoder(in_channels=100, out_channels=256),Decoder(in_channels=768),
                                    projection_dim=256, tau=0.5
                                ).to(device)
    cl_optimizer = torch.optim.Adam(cl_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    loss_func = torch.nn.BCELoss()
    #------Test cv
    synergy_cv, index_test, label_test = data_split(synergy_data)
    
    best_val_metric = [0, 0, 0, 0]
    best_val_test_metric = [0, 0, 0, 0]
    best_test_metric = [0, 0, 0, 0]
    best_test_val_metric = [0, 0, 0, 0]
    best_val_epoch = 0
    best_test_epoch = 0
    logger.info('Metric: AUC, AUPR, F1, ACC')
    for epoch in range(1,epochs+1):
        # ---cv
        if cv_mode == 1:
            cv_data = synergy_cv
        elif cv_mode == 2:
            cv_data = np.unique(synergy_cv[:, 2])  # cline_level
        else:
            cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T  # drug pairs_level
        # ---5CV
        final_test_metric = np.zeros(4)
        final_val_metric = np.zeros(4)
        fold_num = 0
        train_loss_all = 0
        cl_loss_all = 0
        pred_loss_all = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, validation_index in kf.split(cv_data):
            if cv_mode == 1:  # normal_level
                synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
            elif cv_mode == 2:  # cell line_level
                train_name, test_name = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
                synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
            else:  # drug pairs_level
                pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array(
                    [j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_validation = np.array(
                    [j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            # np.savetxt(path + str(epoch) + 'val_' + str(fold_num) + '_true.txt', synergy_validation[:, 3])
            label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
            label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
            index_train = torch.from_numpy(synergy_train).to(device)
            index_validation = torch.from_numpy(synergy_validation).to(device)
            edge_data = synergy_train[synergy_train[:, 3] == 1, 0:3]
            synergy_graph = drop_hyperedges(edge_data, ratio=0)
            
            #----contrastive learning
            train_metric, train_loss, cl_loss, pred_loss = train_cl(drug_set, cline_set, edge_data,index_train, label_train, alpha)
            val_metric, _, y_val_pred = test_cl(drug_set, cline_set, synergy_graph,index_validation, label_validation)
            test_metric, _, y_test_pred = test_cl(drug_set, cline_set, synergy_graph, index_test, label_test)
            train_loss_all += train_loss
            cl_loss_all += cl_loss
            pred_loss_all += pred_loss
            final_test_metric += test_metric
            final_val_metric += val_metric
            fold_num = fold_num + 1
        final_test_metric /= 5
        final_val_metric /= 5
        train_loss_all /=5
        cl_loss_all /=5
        pred_loss_all /=5

        if final_test_metric[0] > best_test_metric[0]:
            best_test_epoch = epoch
            best_test_metric = final_test_metric
            best_test_val_metric = final_val_metric
        if final_val_metric[0] > best_val_metric[0]:
            best_val_epoch = epoch
            best_val_metric = final_val_metric
            best_val_test_metric = final_test_metric
        logger.info('Epoch:{:03d}, train_loss:{:.6f}, cl_loss:{:.6f}, pred_loss:{:.6f}, '.format(epoch, train_loss_all, cl_loss_all, pred_loss_all)+'val: {:.6f} '.format(final_val_metric[0])+
            '{:.6f} '.format(final_val_metric[1])+
            '{:.6f} '.format(final_val_metric[2])+ '{:.6f}'.format(final_val_metric[3]))
    logger.info('best_val_epoch: {}, best_val_metric: {}, best_val_test_metric: {}'.format(best_val_epoch, ' '.join(str(x) for x in best_val_metric), ' '.join(str(x) for x in best_val_test_metric)))
