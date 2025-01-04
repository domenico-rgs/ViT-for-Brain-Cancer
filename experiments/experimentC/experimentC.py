import numpy as np

import gc

import torch
import torch.nn as nn
import torch.optim as optim

import csv

import hdf5storage as h5

from ViT import HyBrainViT

import random
import time
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

from imblearn.metrics import specificity_score


import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import Counter

##############################################
#Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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

##############################################
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

def getTrainValTestDataPoints(hsi_data, hsi_label):
    classes_points = {}
    for i in range(4):
        ith_class_points = np.argwhere(hsi_label==(i+1))
        classes_points[i] = ith_class_points

    min_points_class = np.min([len(points) for _, points in classes_points.items() if len(points) > 0], axis=0)

    train_set = {}
    val_set = {}
    test_set = {}
    for i in range(4):
        #train_end = int(0.6 * min_points_class)
        #val_end = train_end + int(0.2 * min_points_class)
        train_end = int(0.6 * len(classes_points[i]))
        val_end = train_end + int(0.2 * len(classes_points[i]))

        np.random.shuffle(classes_points[i])

        train_set[i] = classes_points[i][:train_end]
        val_set[i] = classes_points[i][train_end:val_end]
        test_set[i] = classes_points[i][val_end:]

    return train_set, val_set, test_set

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

def splitData(mirror_image, band, train_set_points, val_set_points, test_set_points, hsi_labels, patch):
    train_list = [x for _, val in train_set_points.items() for x in val]
    val_list = [x for _, val in val_set_points.items() for x in val]
    test_list = [x for _, val in test_set_points.items() for x in val]

    train_data = np.zeros((len(train_list), patch, patch, band), dtype=float)
    val_data = np.zeros((len(val_list), patch, patch, band), dtype=float)
    test_data = np.zeros((len(test_list), patch, patch, band), dtype=float)
    
    train_labels = np.zeros(len(train_list), dtype=int)
    val_labels = np.zeros(len(val_list), dtype=int)
    test_labels = np.zeros(len(test_list), dtype=int)

    for j in range(len(train_list)):
        train_data[j,:,:,:], train_labels[j] = getPatch(mirror_image, hsi_labels, train_list, j, patch)
    for j in range(len(val_list)):
        val_data[j,:,:,:], val_labels[j] = getPatch(mirror_image, hsi_labels, val_list, j, patch)
    for j in range(len(test_list)):
        test_data[j,:,:,:], test_labels[j] = getPatch(mirror_image, hsi_labels, test_list, j, patch)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def processImage(imgDataName, imgLabelName, image_patch):
    hsi_dataset= h5.loadmat(imgDataName)
    hsi_data = hsi_dataset.get('preProcessedImage')
    hsi_data = normal(hsi_data)

    hsi_dataset_labels = h5.loadmat(imgLabelName)
    hsi_labels = hsi_dataset_labels.get('groundTruthMap')
    hsi_labels = substituteLabel(hsi_labels)

    train_set_points, val_set_points, test_set_points = getTrainValTestDataPoints(hsi_data, hsi_labels)
    mirrored_image = mirror_hsi(hsi_data.shape[0],hsi_data.shape[1],hsi_data.shape[2],hsi_data,image_patch)

    return splitData(mirrored_image, hsi_data.shape[2], train_set_points, val_set_points, test_set_points, hsi_labels, image_patch)

def onlyPatchData(mirror_image, band, hsi_data_points, hsi_labels, patch):
    o_data = np.zeros((len(hsi_data_points), patch, patch, band), dtype=float)
    o_labels = np.zeros(len(hsi_data_points), dtype=int)

    for j in range(len(hsi_data_points)):
        o_data[j,:,:,:], o_labels[j] = getPatch(mirror_image, hsi_labels, hsi_data_points, j, patch)

    return o_data, o_labels

def getImage(data, height, width):
    colorIndex = {0:[0,255,0],1:[255,0,0],2:[0,0,255],3:[125,0,255]};

    npimg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            npimg[y, x, :] = colorIndex.get(data[y][x], [0, 0, 0])
    return npimg

def printImage(img,data, height, width, rgb_img):
    data_reshaped = np.reshape(data, (height, width))
    npimg = getImage(data_reshaped, height, width)                                      

    plt.figure()
    plt.imshow(npimg)
    plt.imshow(rgb_img, alpha=0.7)
    plt.axis('off')
    plt.savefig(f"{img}_single.png", format="png", dpi=600)
    #plt.show()

def calcMetrics(test_all_labels, test_all_preds_argmax, test_all_preds_softmax):
    acc_value = accuracy_score(test_all_labels, test_all_preds_argmax, normalize=True)

    matrix = confusion_matrix(test_all_labels, test_all_preds_argmax, labels=[0,1,2,3])
    per_class_accuracy = matrix.diagonal()/matrix.sum(axis=1)

    specificity = specificity_score(test_all_labels, test_all_preds_argmax, average=None, labels=[0,1,2,3])
    try:
        roc_value = roc_auc_score(test_all_labels, test_all_preds_softmax, multi_class='ovr', average=None, labels=[0,1,2,3])
    except ValueError:
        roc_value = 0
    #roc_value_overall = roc_auc_score(test_all_labels, test_all_preds_softmax, multi_class='ovo', average='weighted')

    precision, recall, fscore, support = precision_recall_fscore_support(test_all_labels, test_all_preds_argmax, beta=1.0, average='weighted', labels=[0,1,2,3])

    with open(f'results/data.csv' , mode='a', newline='') as file_csv:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow([img[0], acc_value, per_class_accuracy, precision, recall, fscore, specificity, roc_value, support])

    """
    print(f'Accuracy: {acc_value}')
    print(f'Per class accuracy: {per_class_accuracy}')
    #print(f'ROC per class: {roc_value}')
    #print(f'ROC overall: {roc_value_overall}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F-score: {fscore}')
    print(f'Support: {support}')

    np.set_printoptions(precision=2)
    """

    # Plot non-normalized confusion matrix
    """
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

batch_size = 64
epochs = 200
device = torch.device("cuda:0")
image_patch = 7
band_patch = 1

"""
image_list = [('ID0038img.mat', 'ID0038gt.mat'),('ID0065img.mat', 'ID0065gt.mat'),
              ('ID0047C1img.mat', 'ID0047C1gt.mat'), ('ID0047C2img.mat', 'ID0047C2gt.mat'),
              ('ID0050img.mat', 'ID0050gt.mat')]
"""
image_list = [
                ('ID0067img.mat', 'ID0067gt.mat'),
                ('ID0018img.mat', 'ID0018gt.mat'),
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
                ('ID0070img.mat', 'ID0070gt.mat'),
                ('ID0072img.mat', 'ID0072gt.mat'),
                ('ID0075img.mat', 'ID0075gt.mat'),
                ('ID0084img.mat', 'ID0084gt.mat')
            ]

# image_list = [
#                 ('ID0067img.mat', 'ID0067gt.mat')
#             ]

for img in image_list:

    train_data, train_labels, val_data, val_labels, test_data, test_labels = processImage(img[0], img[1], image_patch)


    #train_data = train_data.reshape(train_data.shape[0], image_patch*image_patch, 25)
    #val_data = val_data.reshape(val_data.shape[0], image_patch*image_patch, 25)
    #test_data = test_data.reshape(test_data.shape[0], image_patch*image_patch, 25)

    train_data = neighborhood_band(train_data, 25, band_patch, image_patch)
    val_data = neighborhood_band(val_data, 25, band_patch, image_patch)
    test_data = neighborhood_band(test_data, 25, band_patch, image_patch)

    train_data=torch.from_numpy(train_data.transpose(0,2,1)).type(torch.FloatTensor)
    train_labels=torch.from_numpy(train_labels).type(torch.LongTensor)

    val_data=torch.from_numpy(val_data.transpose(0,2,1)).type(torch.FloatTensor)
    val_labels=torch.from_numpy(val_labels).type(torch.LongTensor)

    test_data=torch.from_numpy(test_data.transpose(0,2,1)).type(torch.FloatTensor)
    test_labels=torch.from_numpy(test_labels).type(torch.LongTensor)

    trainLoader=DataLoader(TensorDataset(train_data,train_labels-1),batch_size=batch_size,shuffle=True)
    valLoader=DataLoader(TensorDataset(val_data,val_labels-1),batch_size=batch_size,shuffle=True)
    testLoader=DataLoader(TensorDataset(test_data,test_labels-1),batch_size=batch_size,shuffle=False)

    net = HyBrainViT(patchSize=image_patch, nBlocks=5, mlp_dim=4, mode='CAF', numHeads=16, embedDim=64, numClasses=4, dropout=0.1)
    net.to(device)

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//20, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

    ##############################################
    #TRAINING

    avg_train_loss =[]
    avg_val_loss = []
    roc_val = []
    current_patience = 0
    best_val_loss = float('inf')
    #early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        
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

        #roc_val.append(roc_auc_score(val_all_labels, val_all_preds_softmax, multi_class='ovo', average='weighted'))
        avg_val_loss.append(val_loss / len(valLoader))

        if(avg_val_loss[-1] < best_val_loss):
            best_val_loss = avg_val_loss[-1]
            torch.save(net.state_dict(), f'results/models/best_model_test_{img[0]}_hs1.pth')
        
        scheduler.step(val_loss / len(valLoader))

    #torch.save(net.state_dict(), f'results/models/best_model_test{img[0]}.pth')

    np.save(f'results/losses/train_loss_test{img[0]}.npy',np.array(avg_train_loss))
    np.save(f'results/losses/val_loss_test{img[0]}.npy',np.array(avg_val_loss))
    #np.save(f'val_roc_test.npy',np.array(roc_val))

    end_time = time.time()
    print(f'Total time {img[0]}: {end_time-start_time}')
    ##############################################
    #TESTING
    net = HyBrainViT(patchSize=image_patch, nBlocks=5, mlp_dim=4, mode='CAF', numHeads=16, embedDim=64, numClasses=4, dropout=0.1)
    net.load_state_dict(torch.load(f'results/models/best_model_test_{img[0]}_hs1.pth'))
    net.to(device) #7 16

    net.eval()
    test_all_preds_softmax = []
    test_all_preds_argmax = []
    test_all_labels = []
    with torch.no_grad():
        for _, test_data in enumerate(testLoader, 0):
            test_inputs, test_labels = test_data[0].to(device, dtype=torch.float32), test_data[1].to(device, dtype=torch.long)
            test_outputs = net(test_inputs)

            test_all_preds_softmax.extend((torch.softmax(test_outputs,1)).cpu().numpy())
            test_all_preds_argmax.extend((torch.argmax(test_outputs,1)).cpu().numpy())
            test_all_labels.extend(test_labels.cpu().numpy())

    calcMetrics(test_all_labels, test_all_preds_argmax, test_all_preds_softmax)

    ##############################################
    #NEW IMG
    hsi_dataset= h5.loadmat(img[0])
    hsi_data = hsi_dataset.get('preProcessedImage')
    hsi_data = normal(hsi_data)

    hsi_dataset_labels = h5.loadmat(img[1])
    hsi_labels = hsi_dataset_labels.get('groundTruthMap')
    hsi_labels = substituteLabel(hsi_labels)

    rgb_img = hsi_dataset_labels.get('dataResults')[0,0][7]

    mirrored_image = mirror_hsi(hsi_data.shape[0],hsi_data.shape[1],hsi_data.shape[2],hsi_data,image_patch)
    o_data, o_labels = onlyPatchData(mirrored_image, 25, [point for point, _ in np.ndenumerate(hsi_labels)], hsi_labels, image_patch)
    o_data = neighborhood_band(o_data, 25, band_patch, image_patch)

    newLoader=DataLoader(TensorDataset(torch.from_numpy(o_data.transpose(0,2,1)).type(torch.FloatTensor),torch.from_numpy(o_labels).type(torch.LongTensor)),batch_size=batch_size,shuffle=False)

    net.eval()
    new_all_preds_softmax = []
    new_all_preds_argmax = []
    new_all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for _, new_data in enumerate(newLoader, 0):
            new_inputs, new_labels = new_data[0].to(device, dtype=torch.float32), new_data[1].to(device, dtype=torch.long)
            new_outputs = net(new_inputs)

            new_all_preds_softmax.extend((torch.softmax(new_outputs,1)).cpu().numpy())
            new_all_preds_argmax.extend((torch.argmax(new_outputs,1)).cpu().numpy())
            new_all_labels.extend(new_labels.cpu().numpy())

    end_time = time.time()
    print(f'{img}\nTotal time: {end_time-start_time}')
    printImage(img[0],new_all_preds_argmax, hsi_data.shape[0],hsi_data.shape[1], rgb_img)
    #calcMetrics(new_all_labels, new_all_preds_argmax, new_all_preds_softmax)

    del net
    gc.collect()
    torch.cuda.empty_cache()