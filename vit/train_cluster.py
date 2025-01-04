import numpy as np

import gc
import os
import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim

import hdf5storage as h5

from ViT import HyBrainViT

import random
import time
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score
#from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
#import matplotlib.pyplot as plt

#from tqdm import tqdm

from imblearn.over_sampling import RandomOverSampler


##############################################
#Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

ros = RandomOverSampler(random_state=seed)
#rus = RandomUnderSampler(sampling_strategy={0:1000, 3:1000},random_state=seed)
#sm = SMOTE(random_state=42)

##############################################
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


#FUNCTIONS
def neighborhood_band(x_train, band, band_patch, patch):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)

    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

def normal(data):
    data_normalized = np.zeros(data.shape)
    for i in range(data.shape[0]):
        input_max = np.max(data[i,:])
        input_min = np.min(data[i,:])
        data_normalized[i,:] = (data[i,:]-input_min)/(input_max-input_min)
	
    return data_normalized

def substituteLabel(labels):
    labels[labels==101] = 1 #normal
    labels[labels==102] = 1 #white-tissue -> normal
    labels[labels==200] = 2 #tumour GBM
    labels[labels==221] = 2 #tumour astroglial
    labels[labels==302] = 3 #venous blood
    labels[labels==301] = 3 #arterial blood
    labels[labels==320] = 4 #dura mater
    labels[labels==331] = 0 #skull

    return labels

def getTrainValTestDataPoints(hsi_label):
    classes_points = {}
    for i in range(4):
        ith_class_points = np.argwhere(hsi_label==(i+1))
        classes_points[i] = ith_class_points

    #min_points_class = np.min([len(points) for _, points in classes_points.items() if len(points) > 0], axis=0)

    train_set = {}
    #val_set = {}
    test_set = {}
    for i in range(4):
        #train_end = int(0.8 * min_points_class)
        train_end = int(0.8 * len(classes_points[i]))
        #val_end = train_end + int(0.2 * min_points_class)

        np.random.shuffle(classes_points[i])

        train_set[i] = classes_points[i][:train_end]
        #val_set[i] = classes_points[i][train_end:val_end]
        test_set[i] = classes_points[i][train_end:]

    return train_set, test_set

def mirror_hsi(height,width,band,data_normalized,patch):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)

    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=data_normalized
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=data_normalized[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=data_normalized[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    return mirror_hsi

def getPatch(mirror_image, hsi_labels, points, i, patch):
    x = points[i][0]
    y = points[i][1]

    ptc = mirror_image[x:(x + patch), y:(y + patch), :]
    label = hsi_labels[x,y]

    return ptc, label

def splitData(mirror_image, band, train_set_points, test_set_points, hsi_labels, patch):
    train_list = [x for _, val in train_set_points.items() for x in val]
    #val_list = [x for _, val in val_set_points.items() for x in val]
    test_list = [x for _, val in test_set_points.items() for x in val]

    train_data = np.zeros((len(train_list), patch, patch, band), dtype=float)
    #val_data = np.zeros((len(val_list), patch, patch, band), dtype=float)
    test_data = np.zeros((len(test_list), patch, patch, band), dtype=float)
    
    train_labels = np.zeros(len(train_list), dtype=int)
    #val_labels = np.zeros(len(val_list), dtype=int)
    test_labels = np.zeros(len(test_list), dtype=int)

    for j in range(len(train_list)):
        train_data[j,:,:,:], train_labels[j] = getPatch(mirror_image, hsi_labels, train_list, j, patch)
    #for j in range(len(val_list)):
    #    val_data[j,:,:,:], val_labels[j] = getPatch(mirror_image, hsi_labels, val_list, j, patch)
    for j in range(len(test_list)):
        test_data[j,:,:,:], test_labels[j] = getPatch(mirror_image, hsi_labels, test_list, j, patch)

    return train_data, train_labels, test_data, test_labels

def processImage(imgDataName, imgLabelName, image_patch):
    hsi_dataset= h5.loadmat(imgDataName)
    hsi_data = hsi_dataset.get('preProcessedImage')
    hsi_data = normal(hsi_data)

    hsi_dataset_labels = h5.loadmat(imgLabelName)
    hsi_labels = hsi_dataset_labels.get('groundTruthMap')
    hsi_labels = substituteLabel(hsi_labels)

    train_set_points, test_set_points = getTrainValTestDataPoints(hsi_labels)
    mirrored_image = mirror_hsi(hsi_data.shape[0],hsi_data.shape[1],hsi_data.shape[2],hsi_data,image_patch)

    return splitData(mirrored_image, hsi_data.shape[2], train_set_points, test_set_points, hsi_labels, image_patch)

def onlyPatchData(mirror_image, band, hsi_data_points, hsi_labels, patch):
    o_data = np.zeros((len(hsi_data_points), patch, patch, band), dtype=float)
    o_labels = np.zeros(len(hsi_data_points), dtype=int)

    for j in range(len(hsi_data_points)):
        o_data[j,:,:,:], o_labels[j] = getPatch(mirror_image, hsi_labels, hsi_data_points, j, patch)

    return o_data, o_labels
"""
def getImage(data, height, width):
    colorIndex = {0:[0,255,0],1:[255,0,0],2:[0,0,255],3:[125,0,255]};

    npimg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            npimg[y, x, :] = colorIndex.get(data[y][x], [0, 0, 0])
    return npimg

def printImage(data, height, width):
    data_reshaped = np.reshape(data, (height, width))
    npimg = getImage(data_reshaped, height, width)                                      

    plt.figure()
    plt.imshow(npimg)
    plt.show()

def calcMetrics(test_all_labels, test_all_preds_argmax, test_all_preds_softmax):
    acc_value = accuracy_score(test_all_labels, test_all_preds_argmax, normalize=True)

    matrix = confusion_matrix(test_all_labels, test_all_preds_argmax)
    per_class_accuracy = matrix.diagonal()/matrix.sum(axis=1)

    roc_value = roc_auc_score(test_all_labels, test_all_preds_softmax, multi_class='ovr', average=None)
    roc_value_overall = roc_auc_score(test_all_labels, test_all_preds_softmax, multi_class='ovo', average='weighted')

    precision, recall, fscore, support = precision_recall_fscore_support(test_all_labels, test_all_preds_argmax, beta=1.0, average=None)

    print(f'Accuracy: {acc_value}')
    print(f'Per class accuracy: {per_class_accuracy}')
    print(f'ROC per class: {roc_value}')
    print(f'ROC overall: {roc_value_overall}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F-score: {fscore}')
    print(f'Support: {support}')

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_predictions(
            test_all_labels,
            test_all_preds_argmax,
            display_labels=['normal', 'tumor', 'blood','duraMater'],
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
        plt.show()
"""
##############################################
#BASE SETTINGS
batch_size = 64
epochs = 500
band_patch = 1

##############################################

IDX =  int(os.environ['SLURM_PROCID']) % 12 # CHANGE ACCORDING TO GPUs available

parameter_combinations = [
	(beta1, beta2, embedDim, nBlocks, nHeads, image_patch, is_over)
	for beta1 in [0.85, 0.9]
	for beta2 in [0.9, 0.999]
	for embedDim in [64, 128, 256]
	for nBlocks in [1, 2, 3, 4, 5]
	for nHeads in [1, 2, 4, 8, 16]
    for image_patch in [1, 3, 5, 7]
    for is_over in [False]
]

parser = argparse.ArgumentParser("Training script for HyBrainViT")
parser.add_argument('--num_gpus', default=1, help='GPUs number')
args = parser.parse_args()

# Distribute parameter combinations across different processes
num_gpus = int(args.num_gpus)  # Define the number of GPUs available
combinations_per_gpu = len(parameter_combinations) // num_gpus

start_idx = IDX * combinations_per_gpu
end_idx = start_idx + combinations_per_gpu if IDX < num_gpus-1 else None
process_combinations = parameter_combinations[start_idx:end_idx]

device = torch.device(f"cuda:{IDX}")
print("Training on", device)

##############################################

image_list = [('ID0018img.mat', 'ID0018gt.mat'),
                ('ID0025img.mat', 'ID0025gt.mat'),
                ('ID0029img.mat', 'ID0029gt.mat'),
                ('ID0030img.mat', 'ID0030gt.mat'),
                ('ID0033img.mat', 'ID0033gt.mat'),
                ('ID0034img.mat', 'ID0034gt.mat'),
                ('ID0035img.mat', 'ID0035gt.mat'),
                ('ID0038img.mat', 'ID0038gt.mat'),
                ('ID0047C1img.mat', 'ID0047C1gt.mat'), 
                ('ID0047C2img.mat', 'ID0047C2gt.mat'),
                ('ID0050img.mat', 'ID0050gt.mat'),
                ('ID0051img.mat', 'ID0051gt.mat'),
                ('ID0056img.mat', 'ID0056gt.mat'),
                ('ID0065img.mat', 'ID0065gt.mat'),
                ('ID0067img.mat', 'ID0067gt.mat'),
                ('ID0070img.mat', 'ID0070gt.mat'),
                ('ID0072img.mat', 'ID0072gt.mat'),
                ('ID0075img.mat', 'ID0075gt.mat'),
                ('ID0084img.mat', 'ID0084gt.mat')
            ]

"""
image_list = [('ID0038img.mat', 'ID0038gt.mat'),('ID0065img.mat', 'ID0065gt.mat'),
              ('ID0047C1img.mat', 'ID0047C1gt.mat'), ('ID0047C2img.mat', 'ID0047C2gt.mat'),('ID0050img.mat', 'ID0050gt.mat')]
"""

#image_list = [('ID0038img.mat', 'ID0038gt.mat')]

##############################################
stratified_splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)

test_n=0
for parameters in process_combinations:

    train_data, train_labels, test_data, test_labels = processImage(image_list[0][0], image_list[0][1], parameters[5])
    for img in image_list[1:]:
        tdata, tlabels, tst_data, tst_labels = processImage(img[0], img[1], parameters[5])
        
        train_data=np.concatenate((train_data, tdata), axis=0)
        train_labels=np.concatenate((train_labels, tlabels), axis=0)
        test_data=np.concatenate((test_data, tst_data), axis=0)
        test_labels=np.concatenate((test_labels, tst_labels), axis=0)

    train_data = neighborhood_band(train_data, 25, band_patch, parameters[5])
    test_data = neighborhood_band(test_data, 25, band_patch, parameters[5]) #NEVER USED (FOR LATER)

    indices = list(range(train_data.shape[0]))

    acc_values_folds = []
    roc_values_folds = []
    roc_class_folds = []
    timings = []

	##############################################
    for fold, (train_indices, val_indices) in enumerate(stratified_splitter.split(indices, train_labels)):
        if(parameters[6]):
            tdata, tlabels = ros.fit_resample(np.reshape(train_data[train_indices],(train_data[train_indices].shape[0],train_data.shape[1]*train_data.shape[2])), train_labels[train_indices]-1)
            tdata = np.reshape(tdata,(tdata.shape[0],train_data.shape[1],train_data.shape[2]))
            tdata = torch.from_numpy(tdata.transpose(0,2,1)).type(torch.FloatTensor)
            tlabels = torch.from_numpy(tlabels).type(torch.LongTensor)
        else:
            tdata = torch.from_numpy(train_data[train_indices].transpose(0,2,1)).type(torch.FloatTensor)
            tlabels = torch.from_numpy(train_labels[train_indices]-1).type(torch.LongTensor)

        vdata = torch.from_numpy(train_data[val_indices].transpose(0,2,1)).type(torch.FloatTensor)
        vlabels = torch.from_numpy(train_labels[val_indices]-1).type(torch.LongTensor)

        #weights = compute_class_weight(class_weight="auto", classes=[0,1,2,3], y=(train_labels[train_indices]-1))
        #weights = np.array([1.0, 2.0, 1.0, 1.0])
        #loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).type(torch.FloatTensor)).cuda()
        loss_fn = nn.CrossEntropyLoss().cuda()

        trainLoader=DataLoader(TensorDataset(tdata,tlabels),batch_size=batch_size,shuffle=True)
        valLoader=DataLoader(TensorDataset(vdata,vlabels),batch_size=batch_size,shuffle=True)

        net = HyBrainViT(image_size=parameters[5], near_band=band_patch, nBlocks=parameters[3], mlp_dim=4, mode='CAF', numHeads=parameters[4], embedDim=parameters[2], numClasses=4, dropout=0)
        net.to(device)

        optimizer = optim.AdamW(net.parameters(), lr=1e-4, betas=(parameters[0], parameters[1]), weight_decay=5e-5)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//10, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

        print("\n"+str(IDX)+str(test_n))
        print("Parameters: ", parameters,"\n")

        #TRAINING
        avg_train_loss =[]
        avg_val_loss = []
        roc_val = []
        current_patience = 0
        best_val_loss = float('inf')

        start_time = time.time()
        early_stopper = EarlyStopper(patience=10, min_delta=0.01)
        for epoch in range(epochs):
            
            #EPOCH TRAINING
            net.train()
            running_loss = 0
            for i, data in enumerate(trainLoader, 0):
                tr_inputs, tr_labels = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.long)

                optimizer.zero_grad()

                output = net(tr_inputs)
                loss = loss_fn(output, tr_labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            avg_train_loss.append(running_loss / len(trainLoader))

            #EPOCH VALIDATION
            net.eval()
            val_loss = 0.0
            val_all_preds_softmax = []
            val_all_preds_argmax = []
            val_all_labels = []
            with torch.no_grad():
                for _, val_data in enumerate(valLoader, 0):
                    val_inputs, val_labels = val_data[0].to(device, dtype=torch.float32), val_data[1].to(device, dtype=torch.long)
                    val_outputs = net(val_inputs)
                    val_loss += loss_fn(val_outputs, val_labels).item()

                    val_all_preds_softmax.extend((torch.softmax(val_outputs,1)).cpu().numpy())
                    val_all_preds_argmax.extend((torch.argmax(val_outputs,1)).cpu().numpy())
                    val_all_labels.extend(val_labels.cpu().numpy())

            roc_val.append(roc_auc_score(val_all_labels, val_all_preds_softmax, multi_class='ovo', average='weighted'))
            avg_val_loss.append(val_loss / len(valLoader))

            if early_stopper.early_stop(val_loss / len(valLoader)):             
                break
            
            scheduler.step(val_loss / len(valLoader))
            
            #scheduler.step()
        end_time = time.time()
        timings.append((end_time - start_time))
        
        acc_values_folds.append(accuracy_score(val_all_labels, val_all_preds_argmax, normalize=True))
        roc_values_folds.append(roc_auc_score(val_all_labels, val_all_preds_softmax, multi_class='ovo', average='weighted'))
        roc_class_folds.append(roc_auc_score(val_all_labels, val_all_preds_softmax, multi_class='ovr', average=None))

        torch.save(net.state_dict(), f'results/models/best_model_test_{IDX}_{test_n}_{fold}.pth')

        np.save(f'results/losses/train_loss_test_{IDX}_{test_n}_{fold}.npy',np.array(avg_train_loss))
        np.save(f'results/losses/val_loss_test_{IDX}_{test_n}_{fold}.npy',np.array(avg_val_loss))
        np.save(f'results/losses/val_roc_test_{IDX}_{test_n}_{fold}.npy',np.array(roc_val))

        del net
        gc.collect()
        torch.cuda.empty_cache()

    with open(f'results/data{IDX}.csv' , mode='a', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow([test_n, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4],parameters[5],parameters[6],
                np.mean(acc_values_folds), acc_values_folds[0], acc_values_folds[1], acc_values_folds[2], acc_values_folds[3], acc_values_folds[4], 
                np.mean(roc_values_folds), roc_values_folds[0], roc_values_folds[1], roc_values_folds[2], roc_values_folds[3], roc_values_folds[4],
                roc_class_folds[0][0], roc_class_folds[0][1], roc_class_folds[0][2], roc_class_folds[0][3],
                roc_class_folds[1][0], roc_class_folds[1][1], roc_class_folds[1][2], roc_class_folds[1][3],
                roc_class_folds[2][0], roc_class_folds[2][1], roc_class_folds[2][2], roc_class_folds[2][3],
                roc_class_folds[3][0], roc_class_folds[3][1], roc_class_folds[3][2], roc_class_folds[3][3],
                roc_class_folds[4][0], roc_class_folds[4][1], roc_class_folds[4][2], roc_class_folds[4][3],
                np.mean(timings), timings[0], timings[1], timings[2], timings[3], timings[4]])
        
    test_n+=1