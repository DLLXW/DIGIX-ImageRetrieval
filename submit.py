#
import time
import os
import pdb
import numpy as np
import torch
import torch.nn as nn
from utils import retriDatasetInfer
from config import Config
import cv2
from torch.utils import data
from cnn_finetune import make_model
from sklearn import preprocessing
from modeling import LandmarkNet
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_dis(query_fea, gallery_fea):
        query_fea = query_fea.transpose(1, 0)
        inner_dot = gallery_fea.mm(query_fea)
        dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
        dis = dis - 2 * inner_dot
        dis = dis.transpose(1, 0)
        return dis

def return_index(query_fea, gallery_fea):
    dis = cal_dis(query_fea, gallery_fea)
    sorted_index = torch.argsort(dis, dim=1)
    if top_k != 0:
        sorted_index = sorted_index[:, :top_k]
    return dis, sorted_index
#
def get_featurs(model, testloader,batchsize=10):
    features = None
    cnt = 0
    begin_time=time.time()
    total_iters=len(testloader)
    data_size = total_iters*batchsize
    print("datasize: ", data_size)
    cnt_iter=0
    image_names=[]
    for i, values in enumerate(testloader):
        data,img_path=values
        img_path=[per.split('/')[-1] for per in img_path]
        image_names+=img_path
        cnt_iter+=1
        data = data.to(device)
        output=model.module.xcep_net(data)
        output=output.view(output.size(0), -1)
        #
        feature = output.data.cpu().numpy()
        feature=preprocessing.normalize(feature,norm='l2')
        #
        output_arc,_= model(data)
        feature_arc = output_arc.data.cpu().numpy()
        feature_arc=preprocessing.normalize(feature_arc,norm='l2')
        feature=np.concatenate((feature,feature_arc),axis=1)
        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))
        current_time = time.time()
        time_perIter = (current_time-begin_time)/cnt_iter
        eta=time_perIter*(total_iters-cnt_iter)/60
        print("iter:{}/{},  ETA:{}m".format(cnt_iter,total_iters,eta))
    return features,image_names


if __name__ == '__main__':
    begin_time=time.time()
    opt=Config()
    model  = LandmarkNet(n_classes=opt.num_classes,
                     model_name=model_name,
                     pooling=opt.pooling,
                     loss_module=opt.loss_module,
                    fc_dim=opt.fc_dim)
    device=torch.device(opt.device)
    model.to(device)
    model = nn.DataParallel(model)
    best_weight = torch.load(opt.test_model_dir)
    model.load_state_dict(best_weight)
    model.eval()
    #
    for mode in ['qeuery','gallary']:
        if mode=='gallery':
            image_datasets = retriDatasetInfer(opt.gallery_dir, input_size=opt.input_size)
            dataset_loaders = torch.utils.data.DataLoader(image_datasets,
                                                      batch_size=opt.test_batch_size,
                                                      shuffle=False, num_workers=4)
            gallery_frt,gallery_names=get_featurs(model, dataset_loaders,opt.test_batch_size)
        else:
            image_datasets = retriDatasetInfer(opt.query_dir, input_size=opt.input_size)
            dataset_loaders = torch.utils.data.DataLoader(image_datasets,
                                                      batch_size=opt.test_batch_size,
                                                      shuffle=False, num_workers=4)
            qeuery_frt,qeuery_names=get_featurs(model, dataset_loaders,opt.test_batch_size)
            
    print('get features finished..............')
    print('begin searching images............')
    query_frt=torch.from_numpy(query_frt)
    gallery_frt=torch.from_numpy(gallery_frt)
    _,sorted_index=return_index(query_frt, gallery_frt)
    sorted_index=sorted_index.numpy().tolist()
    csv_file = open('submission.csv', 'w')
    
    for i in range(len_query):
        query=query_frt[i]
        q_name=query_names[i]
        str_w=''
        for k in range(top_k):
            if k<9:
                str_w+=gallery_names[sorted_index[i][k]]+','
            else:
                str_w += gallery_names[sorted_index[i][k]]
        csv_file.write(q_name+','+'{'+str_w+'}'+'\n')
    print('time spend:', time.time() - begin_time)
