# 1. Title: Adaptive Memory Networks: Learning Online from Non-Stationary Data Streams in Class-Incremental Scenarios
#### Presentation link: https://drive.google.com/file/d/1TZ0ChImedXlzyq4qJeFOTW6V7oeg3cxD/view?usp=drive_link
## 2. Introduction 	
### Motivation and Goal
When training data arrives sequentially in the online multi-task learning method, due to the differences in data distribution between the source domain data and the target domain data, although Transfer Learning technology can be used to transfer knowledge from the source domain to the target domain, it can also lead to catastrophic forgetting and other issues. Catastrophic forgetting of data can significantly degrade the performance of deep models. Inspired by the structure of human memory, we propose an Adaptive Memory Networks for Class-Incremental Learning model based on memory replay to address the problem of catastrophic forgetting. Human memory consists of short-term memory and long-term memory. The primary component of long-term memory helps humans remember past experiences and facts. Additionally, memory not only holds specific memories but also common memories that shared similarities between tasks . 

In this project,  we represent memory as common feature and specific feature within neural networks to mitigate catastrophic forgetting. And hypothesize that common feature are significantly less prone to forgetting and propose a novel hybrid incremental learning framework that learns a disjoint representation for task-invariant(common feature) and task-specific features(specific feature) required to solve a sequence of tasks. Intuitively, specific feature also recalls input-output relationships (facts) from previous tasks, which is achieved by jointly rehearsing previous samples and learning the current task through a replay-based approach. Furthermore, common feature aims to capture long-term task-relevant information across task sequences to regulate the learning of the current task, thereby preserving task-specific weight realizations (experiences) in high task-specific layers. In this work, we implement a concrete instantiation of the proposed task memory by generating instance replays. We intend to conduct extensive experiments on the benchmark dataset 20-class CIFAR-100 to verify that our proposed method can outperform previous approaches and achieve significant improvements by retaining information from samples and tasks.

## 3. Datasets--20-Split CIFAR100 
We will evaluate our approach on the commonly used benchmark datasets for T -split class-incremental learning where the entire dataset is divided into T disjoint susbsets or tasks.
And we want to demonstrate the experiments on sequentially learning single datasets such as 20-Split CIFAR100, which is incrementally learn CIFAR100 in 5 classes at a time in 20 tasks
### Training datasets
	*CIFAR100: CIFAR100 dataset will be auto-downloaded in future project code impeletation.
	*Structure of data directory
	data
	├── cifar100
	│   └── cifar-100-python
	│       ├── train
	│       ├── test
	│       ├── meta  This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names
	│       └── file.txt~

#### The CIFAR-100 dataset description
This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
Here is the list of classes in the CIFAR-100:

		Superclass 	Classes
		aquatic mammals 	beaver, dolphin, otter, seal, whale
		fish 	aquarium fish, flatfish, ray, shark, trout
		flowers 	orchids, poppies, roses, sunflowers, tulips
		food containers 	bottles, bowls, cans, cups, plates
		fruit and vegetables 	apples, mushrooms, oranges, pears, sweet peppers
		household electrical devices 	clock, computer keyboard, lamp, telephone, television
		household furniture 	bed, chair, couch, table, wardrobe
		insects 	bee, beetle, butterfly, caterpillar, cockroach
		large carnivores 	bear, leopard, lion, tiger, wolf
		large man-made outdoor things 	bridge, castle, house, road, skyscraper
		large natural outdoor scenes 	cloud, forest, mountain, plain, sea
		large omnivores and herbivores 	camel, cattle, chimpanzee, elephant, kangaroo
		medium-sized mammals 	fox, porcupine, possum, raccoon, skunk
		non-insect invertebrates 	crab, lobster, snail, spider, worm
		people 	baby, boy, girl, man, woman
		reptiles 	crocodile, dinosaur, lizard, snake, turtle
		small mammals 	hamster, mouse, rabbit, shrew, squirrel
		trees 	maple, oak, palm, pine, willow
		vehicles 1 	bicycle, bus, motorcycle, pickup truck, train
		vehicles 2 	lawn-mower, rocket, streetcar, tank, tractor
		
Binary version
The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as test_batch.bin. Each of these files is formatted as follows:

<1 x label><3072 x pixel>
...
<1 x label><3072 x pixel>

In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.

Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.

There is another file, called batches.meta.txt. This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. It is merely a list of the 10 class names, one per row. The class name on row i corresponds to numeric label i. 


You can also manually download the dataset through the link below
https://www.cs.toronto.edu/~kriz/cifar.html

##  The size of the dataset
Training set size: 2125 images of 32 X 32
Validation set size: 375 images of 32 X 32
Train + Val set size: 2500 images of 32 X 32
Test set size: 500 images of 32 X 32

## Baseline Performance
AVG ACC: 73.0560%
AVG BWT: 0.00% ( There are no changes compared with previous performance )



## 4. Plan of Work and Time Schedule
	23/10/29 project 1 presentaltion
	23/11/05 Project 1 disscussion
	23/12/03 Project 2 presentation
	23/12/10 Project 2 disscussion
	23/12/17 Result evaluation for your proposed project (report)










