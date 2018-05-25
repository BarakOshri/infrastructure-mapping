
# from __future__ import print_function, division
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from utils import addis as util
import ast
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

np.set_printoptions(threshold=np.nan)

satellite = 'l8'
filetail = ".0.npy"
len_dataset = 7022
test_urban = int(sys.argv[2])
test_rural = int(sys.argv[3])
test_country = 0
test_unique = 1
data_dir = sys.argv[1]
argv = sys.argv
# columns = util.balanced_binary_features
columns = [ argv[4]]
hold_out = ['CotedIvoire']
column_weights = [1 for _ in range(len(columns))] # How much to weigh each column in the loss function

num_examples = 1000
train_test_split = 0.8
continuous = False
lr = 1e-4 # was 0.01 for binary
momentum = 0.5 # was 0.4 for binary
detailed_metrics_for = 20
batch_size = 128
num_workers = 4
num_epochs = 20
weight_decay =1e-3

use_five_bands = True

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # all_results = open(satellite + '_results.csv', 'w')
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_train_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                dataloders = dataloaders_train
                current_dataset = dataset_train
            else:
                model.train(False)  # Set model to evaluate mode
                dataloders = dataloaders_test
                current_dataset = dataset_test

            dataset_size = len(current_dataset)

            running_loss = 0.0
            running_corrects = torch.zeros(len(columns))
            running_preds = None
            running_labels = None
            running_scores = None

            # Iterate over data.
            for data in dataloders:
                # get the inputs
                inputs = data['image']
                if continuous: 
                    labels = data['labels'].type(torch.FloatTensor)
                else:
                    labels = data['labels'].type(torch.LongTensor)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if use_five_bands:
                    convolved = convolver(inputs)
                    outputs = model(convolved)
                else:
                    outputs = model(inputs)
                scores = sigmoider(outputs)
                preds = torch.round(scores).data
                scores = scores.data
                # outputs = outputs.type(torch.cuda.LongTensor)
                loss = criterion(outputs.squeeze(), labels.type(torch.cuda.FloatTensor).squeeze())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                if not continuous:
                    running_corrects += torch.sum((preds == labels.data.type(torch.cuda.FloatTensor)).type(torch.FloatTensor), 0)
                    if not continuous and epoch >= num_epochs - detailed_metrics_for: 
                        if running_preds is None: 
                            running_preds = preds.cpu().numpy()
                            running_labels = labels.data.cpu().numpy()
                            running_scores = scores.cpu().numpy()
                        else: 
                            running_preds = np.vstack((running_preds, preds.cpu().numpy()))
                            running_labels = np.vstack((running_labels, labels.data.cpu().numpy()))
                            running_scores = np.vstack((running_scores, scores.cpu().numpy()))

                # print (preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.numpy() / dataset_size
            if not continuous and epoch >= num_epochs - detailed_metrics_for:
                print ('%s Loss: %.4f') % (phase, epoch_loss)
                for i, column in enumerate(columns):
                    column_labels = running_labels[:, i]
                    column_preds = running_preds[:, i]
                    column_scores = running_scores[:, i]

                    epoch_f1 = f1_score(column_labels, column_preds)
                    epoch_precision = precision_score(column_labels, column_preds)
                    epoch_recall = recall_score(column_labels, column_preds)
                    roc_score = roc_auc_score(column_labels, column_scores)
                    print ('%s Acc: %.4f F1: %.4f Precision: %.4f Recall: %.4f ROC_score: %.4f') % (column, epoch_acc[i], epoch_f1, epoch_precision, epoch_recall, roc_score)
                    print ('Balance: %.4f' % current_dataset.balance[i])
                    false_positive_index = np.argmin(column_labels - column_scores)
                    false_negative_index = np.argmax(column_labels - column_scores)
                    true_positive_index = np.argmin((column_labels - column_scores) + 2*(1-column_labels))
                    true_negative_index = np.argmax((column_labels - column_scores) - 2*column_labels)

                    false_positive_sat_index = current_dataset.indices[false_positive_index] + 1, column_scores[false_positive_index], column_labels[false_positive_index]
                    false_negative_sat_index = current_dataset.indices[false_negative_index] + 1, column_scores[false_negative_index], column_labels[false_negative_index]
                    true_positive_sat_index = current_dataset.indices[true_positive_index] + 1, column_scores[true_positive_index], column_labels[true_positive_index]
                    true_negative_sat_index = current_dataset.indices[true_negative_index] + 1, column_scores[true_negative_index], column_labels[true_negative_index]

                    print "False positive id, score, label: %d, %.4f, %d" % false_positive_sat_index
                    print "False negative id, score, label: %d, %.4f, %d" % false_negative_sat_index
                    print "True positive id, score, label: %d, %.4f, %d" % true_positive_sat_index
                    print "True negative id, score, label: %d, %.4f, %d" % true_negative_sat_index
                    print ""

                # print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                #             phase, epoch_loss, epoch_acc, epoch_f1))
            else:
                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #     phase, epoch_loss, epoch_acc))
                print ('%s Loss: %.4f') % (phase, epoch_loss)
                for i, column in enumerate(columns):
                    epoch_f1 = f1_score(running_labels[:, i], running_preds[:, i])
                    print ('%s Acc: %.4f') % (column, epoch_acc)
    
            # all_results.write(','.join([str(epoch), phase, str(epoch_loss), str(epoch_acc)]) + '\n')
            # deep copy the model
            if phase == 'val' and np.mean(epoch_acc) > best_acc:
                best_acc = np.mean(epoch_acc)
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_train_acc, best_acc



class AddisDataset(Dataset):
    """Addis dataset."""
    def __init__(self, indices, csv_file, root_dir, columns, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the numpy files.
            column (string): Variable to predict
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)[columns].values[indices] # TODO: lol indexing is jank rn will change
        self.root_dir = root_dir
        self.transform = transform
        self.indices = indices

        if not continuous:
            self.balance = np.sum(self.data, axis=0) / float(len(self.data))
            # print "The following are balanced"
            # for i in range(len(self.balance)):
            #     if self.balance[i] >= 0.05 and self.balance[i] <= 0.95: print "'%s'," % columns[i]
            # print "The above are balanced"


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, satellite + '_median_addis_multiband_224x224_%d.npy' % (self.indices[idx]))
        # image = np.load(img_name)[:, :, :3]
        if use_five_bands: image = np.load(img_name)
        else: image = np.load(img_name)[:, :, :3]
        labels = self.data[idx]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': labels, 'id': indices[idx]}

        return sample

class AfroDataset(Dataset):
    """Afrobarometer dataset."""
    def __init__(self, indices, csv_file, root_dir, columns, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the numpy files.
            column (string): Variable to predict
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)[columns].values[indices] # TODO: lol indexing is jank rn will change
        self.root_dir = root_dir
        self.transform = transform
        self.indices = indices

        if not continuous:
            self.balance = np.sum(self.data, axis=0) / float(len(self.data))
            # print "The following are balanced"
            # for i in range(len(self.balance)):
            #     if self.balance[i] >= 0.05 and self.balance[i] <= 0.95: print "'%s'," % columns[i]
            # print "The above are balanced"


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #inx = np.random.choice(self.indices)
        #img_name = os.path.join(self.root_dir, satellite + '_median_afro_multiband_224x224_%d.npy' % (inx))
        img_name = os.path.join(self.root_dir, satellite + '_median_afro_multiband_224x224_%d.npy' % (self.indices[idx]))
        # image = np.load(img_name)[:, :, :3]
        if use_five_bands: image = np.load(img_name)
        else: image = np.load(img_name)[:, :, :3][:,:,::-1].copy()
        labels = self.data[idx]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': labels, 'id': indices[idx]}

        return sample

####### Initialize Data

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406, 0.45, 0.45], [0.229, 0.224, 0.225, 0.225, 0.225])
    ])

indices = np.arange(len_dataset)
# np.random.shuffle(indices)
def get_clean_indices():
    indices = []
    log = []
    with open(argv[5],'r') as f:
        missing = ast.literal_eval(f.read())
    for i in range(0, len_dataset-1):
        if i < 6700:
            if i+1 not in missing:
                indices.append(i)
        else:
            indices.append(i)
    return np.array(indices)

indices = get_clean_indices()

def country_urban_indices(indices, len_dataset):
    data_country = pd.read_csv(argv[6])['country'][np.arange(len_dataset)].values
    data_urban =  pd.read_csv(argv[6])['urban'][np.arange(len_dataset)].values
    data_unique =  pd.read_csv(argv[6])['uniquegeocode'][np.arange(len_dataset)].values
    countries = {}
    urban = []
    rural = []
    unique =[]
    for i in indices:
        country = data_country[i]
        if country not in countries:
            countries[country] = []
        countries[country].append(i)
        urban_val = data_urban[i]
        if urban_val:
            urban.append(i)
        else:
            rural.append(i)
        if data_unique[i]:
            unique.append(i)
    return (countries, urban, rural, unique)

countries, urban, rural, unique = country_urban_indices(indices, len_dataset)
num_examples = len(indices)
train_indices = []
test_indices = []
if test_country:
    for country in countries:
        if not country in hold_out:
            train_indices = train_indices + countries[country]
        else:
            test_indices = test_indices+countries[country]
else:
    np.random.seed(1)
    np.random.shuffle(indices)
    split_point = int(num_examples*train_test_split)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:num_examples]
temp = test_indices
temp_train = train_indices
if test_unique:
    test_indices = []
    train_indices = []
    for i in temp:
        if i in unique:
            test_indices.append(i)
        else:
            train_indices.append(i)
    for i in temp_train:
        train_indices.append(i)
final_test_indices = []
if test_urban:
    for i in test_indices:
        if i in urban:
            final_test_indices.append(i)

            
if test_rural:
    for i in test_indices:
        if i in rural:
            final_test_indices.append(i)


print len(train_indices)
test_indices = final_test_indices;
print len(test_indices)
dataset_train = AfroDataset(train_indices, argv[6],
                                    root_dir=data_dir,
                                    columns=columns,
                                    transform=data_transforms)
dataset_test = AfroDataset(test_indices, argv[6],
                                    root_dir=data_dir,
                                    columns=columns,
                                    transform=data_transforms)

for i, column in enumerate(columns):
    print "Balance %s train: %f, test: %f" % (column, dataset_train.balance[i], dataset_test.balance[i])

dataloaders_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloaders_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

use_gpu = torch.cuda.is_available()

######## Train Model

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
convolver = nn.Conv2d(6, 3, 1)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
if continuous:
    model_ft.fc = nn.Linear(num_ftrs, 1)
else:
    model_ft.fc = nn.Linear(num_ftrs, len(columns))

sigmoider = nn.Sigmoid()

if use_gpu:
    model_ft = model_ft.cuda()
    convolver = convolver.cuda()

if not continuous:
    assert(len(columns) == len(column_weights))
    #column_weights = np.minimum(dataset_train.balance, 1-dataset_train.balance)
    criterion = nn.BCEWithLogitsLoss(weight=torch.cuda.FloatTensor(column_weights))
if continuous:
    criterion = nn.MSELoss(size_average=True)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay = weight_decay)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
