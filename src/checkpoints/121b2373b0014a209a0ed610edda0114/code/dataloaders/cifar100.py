# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from PIL import Image
import os
import os.path
import sys


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import datasets, transforms

from utils import *
#-------------------------------------------------Imports necessary libraries for dataset processing, system operations, and others.---------------------------------------------------

# defined a subclass of the cifar10 dataset from torchvision
class iCIFAR10(datasets.CIFAR10):

    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None, target_transform=None, download=True):
        # Initializes the custom dataset. it has various parameters, including where data stored, which classes to consider, memory for past tasks, and more

        super(iCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform, download=True) # super calls the base cifar10 class

    # set up class mapping for the dataset
        self.train = train  # training set or test set, This line assigns the value of the train argument to the instance variable self.train.
        if not isinstance(classes, list):
            classes = [classes] # These lines check if the classes argument is not of type list. If it's not a list (for example, if it's a single integer or string),
            # it wraps classes in a list. This is done to ensure that subsequent operations on classes can always assume it's a list, making the code more consistent and avoiding potential errors.


# This line creates a dictionary where the keys are class labels or identifiers (from the classes list) and the values are corresponding integer indices.
        # For instance, if classes = ['cat', 'dog'], the resulting self.class_mapping would be {'cat': 0, 'dog': 1}.
        # This provides a way to map class labels to unique integer indices.
        self.class_mapping = {c: i for i, c in enumerate(classes)} # This creates a dictionary to map the original class labels to new labels starting from 0 and incrementing.
        self.class_indices = {} # This initializes an empty dictionary called self.class_indices.


#This loop goes through each class in the classes list.
        # For each class, it uses self.class_mapping to get its integer index and then initializes an empty list in the self.
        # class_indices dictionary with that index as the key.
        # The purpose seems to be to prepare a structure where indices of samples belonging to each class can be stored later.
        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []

# Train/Test Data Preparation
# The next few lines handle filtering of the CIFAR10 data based on the provided classes.
# For training data, it also incorporates memory samples from previous tasks if available.
        if self.train: # If the dataset is for training, it does the following:
            train_data = [] # This initializes an empty list to store the training data.
            train_labels = [] # This initializes an empty list to store the training labels.
            train_tt = []  # task module labels
            # This initializes an empty list to store the task module labels.
            train_td = []  # disctiminator labels
            # This initializes an empty list to store the discriminator labels.

            for i in range(len(self.data)): # This loop goes through each sample in the dataset.
                if self.targets[i] in classes: # If the sample's class is in the classes list, it does the following:
                    train_data.append(self.data[i]) # It adds the sample to the train_data list.
                    train_labels.append(self.class_mapping[self.targets[i]]) # It adds the sample's class label to the train_labels list.
                    train_tt.append(task_num)
                    train_td.append(task_num+1)
                    self.class_indices[self.class_mapping[self.targets[i]]].append(i)

# This nested loop goes through each task and its stored memory samples.
            # If the label of a memory sample is within the range of the current task's memory classes, the sample's data and metadata are appended to the training lists.
            if memory_classes:
                for task_id in range(task_num): # This loop goes through each task ID from 0 to task_num - 1.
                    for i in range(len(memory[task_id]['x'])): # This loop goes through each sample in the memory for the current task.
                        if memory[task_id]['y'][i] in range(len(memory_classes[task_id])):
                            train_data.append(memory[task_id]['x'][i])
                            train_labels.append(memory[task_id]['y'][i])
                            train_tt.append(memory[task_id]['tt'][i])
                            train_td.append(memory[task_id]['td'][i])

# This segment converts the list of training data into a numpy array and then assigns the organized data and labels to the instance variables.
            self.train_data = np.array(train_data)
            self.train_labels = train_labels
            self.train_tt = train_tt
            self.train_td = train_td

# for Testing Data
        if not self.train:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            # These lines get the filename of the test data and open the corresponding file.

        # Loads the test data from the file. There's a version-specific handling: for Python 2, it uses the default loading method, and for Python 3, it specifies an encoding.
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']

        # This segment is similar to the one for training data, except that it doesn't consider memory samples.
            #the test data and labels from the loaded entry.
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:

                self.test_labels = entry['fine_labels']
            fo.close() # This closes the file.
            self.test_data = self.test_data.reshape((10000, 3, 32, 32)) # Reshapes the test data and then transposes it to have a different order of dimensions.
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        # This organizes the test data, filters the data based on classes of interest, and updates the test data and metadata lists.
            test_data = []
            test_labels = []
            test_tt = []  # task module labels
            test_td = []  # disctiminator labels
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i])
                    test_labels.append(self.class_mapping[self.test_labels[i]])
                    test_tt.append(task_num)
                    test_td.append(task_num + 1)
                    self.class_indices[self.class_mapping[self.test_labels[i]]].append(i)

            self.test_data = np.array(test_data)
            self.test_labels = test_labels
            self.test_tt = test_tt
            self.test_td = test_td

# Data Retrieval 数据检索
# Returns the image, target class, and associated task labels for the given index.
    def __getitem__(self, index):
        if self.train:
            img, target, tt, td = self.train_data[index], self.train_labels[index], self.train_tt[index], self.train_td[index]
        else:
            img, target, tt, td = self.test_data[index], self.test_labels[index], self.test_tt[index], self.test_td[index]
# Returns the data (image, target class, and task-specific labels) for a given index.
        # Data is first converted into a PIL Image for consistent handling with other datasets.
        # Then, any transformations specified during initialization are applied.
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image （Python Image Library）
        try:
            img = Image.fromarray(img)
        except:
            pass

        try:
            if self.transform is not None:
                img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            pass

        return img, target, tt, td



# Returns the number of samples in the dataset (either training or testing)
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


# This is the central class for the incremental learning dataset setup.
class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args): # It initializes the dataset generator using configurations provided via args.
        super(DatasetGen, self).__init__()

        self.seed = args.seed # 通过设置随机种子，可以控制随机过程的可重复性，保证实验或模型训练的结果可以重现，并且方便进行模型比较和调试
        self.batch_size=args.batch_size # This sets the batch size, which is the number of samples processed together in one forward/backward pass.
        self.pc_valid=args.pc_valid # this set the percentage of the dataset that should be reserved for validation
        self.root = args.data_dir # this sets the directory where the dataset is or will be stored
        self.latent_dim = args.latent_dim # this sets the size of the latent dimension

        self.num_tasks = args.ntasks # 20 tasks
        self.num_classes = 100 # the total number of classes in the dataset

        self.num_samples = args.samples # sets the number of samples to be stored in memory for each task


        self.inputsize = [3,32,32] # define the size of the input images, which is 3*
        mean=[x/255 for x in [125.3,123.0,113.9]] #normalize
        std=[x/255 for x in [63.0,62.1,66.7]]

# Creates a composition of transformations applied to the dataset: converting data to a tensor and then normalizing using the previously calculated mean and standard deviation.
        self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# This creates a list of tasks, where each task has an associated number of classes. For instance, if there are 20 tasks for the CIFAR-100 dataset, each task would typically have 5 classes.
        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]  # 20 classes per task (100/20)

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = args.workers # This sets the number of worker processes for data loading.
        self.pin_memory = True  # When set to True, data loaders will copy tensors into CUDA pinned memory before returning them. It can lead to speed improvements when using GPUs.

        np.random.seed(self.seed)
        # np.random.permutation(self.num_classes) creates a random permutation of class indices.
        # np.split divides this permutation into a specified number of chunks (self.num_tasks).
        # This results in a list of arrays, where each array contains a unique set of class indices representing a task.
        task_ids = np.split(np.random.permutation(self.num_classes),self.num_tasks)   # This permutation ensures that classes are randomly assigned to tasks.
        self.task_ids = [list(arr) for arr in task_ids]  # Converts the arrays from the previous step into lists for easier handling in Python.

# Initializes dictionaries to store training sets, test sets, and training splits for validation for each task.
        self.train_set = {}
        self.test_set = {}
        self.train_split = {}

# self.task_memory is a dictionary storing memory samples for each task
# This initializes a nested dictionary structure for the memory mechanism.
# Each task has its own memory, and for each task, there are four lists: x for data, y for labels, and tt and td for additional metadata.
        self.task_memory = { i: {name: [] for name in ['x', 'y', 'tt', 'td']} for i in range(self.num_tasks) }
        # for i in range(self.num_tasks):
        #     self.task_memory[i] = {}
        #     self.task_memory[i]['x'] = []
        #     self.task_memory[i]['y'] = []
        #     self.task_memory[i]['tt'] = []
        #     self.task_memory[i]['td'] = []

        self.use_memory = args.use_memory


# It's responsible for creating and returning data loaders for a specific task.
    # It sets up training, validation, and test loaders and can incorporate memory samples from previous tasks if needed.

    def get(self, task_id): # It generates and returns data loaders for the specified task.

        self.dataloaders[task_id] = {}  # This initializes an empty dictionary for the current task_id within the dataloaders attribute. This dictionary will store data loaders for training, validation, and testing datasets for the current task.
        sys.stdout.flush()  #This flushes the standard output buffer. It ensures that any output that might be buffered is actually written to the console immediately.


# If we're at the first task (task_id == 0), it sets memory_classes and memory to None. This is because for the first task, there are no previous tasks, hence no memory.
        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory # For tasks other than the first, it sets memory_classes to all task IDs, and memory to the stored memory, which is a record of samples from previous tasks.

        # This creates an instance of the iCIFAR100 dataset for training data. It sets the dataset's root directory, specifies the classes for the current task, incorporates memory samples from previous tasks if available, specifies the current task number, and applies a transformation.
        self.train_set[task_id] = iCIFAR100(root=self.root, classes=self.task_ids[task_id], memory_classes=memory_classes,
                                            memory=memory, task_num=task_id, train=True, download=True, transform=self.transformation)
        # This creates a similar instance but for testing data. Note that memory is not considered for the test set.
        self.test_set[task_id] = iCIFAR100(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
                                           memory=None, task_num=task_id, train=False,
                                     download=True, transform=self.transformation)




# This calculates how many samples should be reserved for validation based on a percentage (self.pc_valid). pc_valid = 0.15 by default. This means that 15% of the training data is used for validation.
        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split]) # This splits the training set into actual training data (train_split) and validation data (valid_split) using the split value calculated in the previous step.

        self.train_split[task_id] = train_split # Stores the training split for the current task for potential future use.

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True) # These three lines set up the data loaders for training, validation, and testing datasets. These loaders allow the model to fetch data in batches during training and testing.
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=int(self.batch_size * self.pc_valid),
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory,shuffle=True)

# Stores the data loaders and a name for the current task within the dataloaders dictionary.
        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'CIFAR100-{}-{}'.format(task_id,self.task_ids[task_id])

# Prints out the sizes of the training, validation, combined training + validation, and test datasets.
        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

# If the flag use_memory is set to 'yes' and the number of samples to store in memory is greater than 0, it updates the memory with samples from the current task.
        if self.use_memory == 'yes' and self.num_samples > 0 :
            self.update_memory(task_id)

        return self.dataloaders # Returns the dataloaders dictionary which contains the data loaders for training, validation, and testing for the current task.



    def update_memory(self, task_id): # Updates memory samples for the given task, selecting a subset of samples from the current task.

        num_samples_per_class = self.num_samples // len(self.task_ids[task_id])
        # This line calculates the number of samples to be stored in memory for each class in the current task.
        # It divides the total number of samples (self.num_samples) by the number of classes in the current task.
        mem_class_mapping = {i: i for i, c in enumerate(self.task_ids[task_id])} # 每一步 self.task_ids[task_id]) 释放一小批数据用来学习 这个函数的基本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置。
        # This creates a dictionary that maps class numbers in the current task to a new set of numbers starting from 0. This re-indexing is often useful for working with memory in incremental learning.

        # Looping over each class in the current task
        for i in range(len(self.task_ids[task_id])):
            # Getting all samples for this class
            data_loader = torch.utils.data.DataLoader(self.train_split[task_id], batch_size=1,
                                                        num_workers=self.num_workers,
                                                        pin_memory=self.pin_memory) # This line creates a data loader for the training data of the current task. This loader will fetch samples one by one (batch_size=1).
            # Randomly choosing num_samples_per_class for this
# This line randomly selects indices of samples. The number of indices selected is the same as the previously computed num_samples_per_class. This ensures that a balanced number of samples is taken from each class.
            randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_class]

            # Adding the selected samples to memory
            for ind in randind:
                self.task_memory[task_id]['x'].append(data_loader.dataset[ind][0]) # This line appends the image (or data) of the selected sample to the memory for the current task.
                self.task_memory[task_id]['y'].append(mem_class_mapping[i]) # This appends the class label of the selected sample to the memory. It uses the mem_class_mapping to re-index the class labels.
                self.task_memory[task_id]['tt'].append(data_loader.dataset[ind][2]) # These lines append additional metadata of the selected sample to the memory.
                self.task_memory[task_id]['td'].append(data_loader.dataset[ind][3])

        print ('Memory updated by adding {} images'.format(len(self.task_memory[task_id]['x'])))

# After each task, a subset of samples is chosen to be stored in memory.
# This function handles that process.
# It takes a balanced sample (i.e., equal from each class in the task) to ensure fair representation.
