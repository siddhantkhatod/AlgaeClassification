# AlgaeClassifier
Classifies microscopic images of algae with 25 classes


### Overview

This repo contains multiple image classification models (see models folder) trained on microscopic images of algae.


### How to Use

The AlgaeClassification.ipynb script loads and classifies algae images. In the script, change:

- The path of [model_path] to the path to the desired classification model you want to use (see models folder)
- The path of [image_directory] to the path of the folder containing the images you want to classify
- The path of [results_directory] to where you would like the classification results to be saved

Run the script, and it will output classification labels and predictions associated with each image name as well as output an excel file containing this information to the [results_directory] path.


### Model Descriptions

There are multiple trained models you can call via the AlgaeClassifier.ipynb script.

- **algaeclassification_cnn**: Custom-built CNN trained, validated, and tested on the entire algae dataset (all 853 images); kernel size: 3x3; layers: 3; epochs: 30; batch size: 32
- **algaeclassification_yolo**: YOLOv5 model finetuned with 20-shot learning; epochs: 10
- **algaeclassification_maskrcnn**: Mask R-CNN model finetuned with 20-shot learning; epochs: 10
- **algaeclassification_ssd**: SSD model finetuned with 20-shot learning; epochs: 10


### Algae Classes

The algae classification model can identify the following 25 classes:
- Actinoptychus
- Bacillaria
- Biddulphia
- Centric Diatom
- Ciliate
- Coscinodiscus
- Cylindrotheca
- Dactyliosolen
- Diatom
- Dinoflagellate
- Entomoneis
- Euglenoid
- Fragilidium
- Hemiaulus
- Heterosigma Akashiwo
- Lyrella
- Navicula
- Nitzschia
- Odontella
- Paralia
- Pennate Diatom
- Pleurosigma
- Prorocentrum
- Tintinnid
- Tripos Hircus


# Abstract

Harmful algae can produce toxic effects on people, fish, shellfish, marine mammals, and birds. Detecting, identifying, and counting algae in water samples under a microscope is difficult, tedious, and time consuming and requires experience and expert knowledge–causing a bottleneck in algae research. An automatic deep learning algae classification model would significantly improve algae detection methods. In this project, I explored the ability to use pre-trained object detection models for algae identification. Three pre-trained models (YOLO, Mask R-CNN, and SSD) were assessed for their ability to identify harmful algae cells in microscopic images after zero-shot, single-shot, and multi-shot learning. In addition, a custom convolutional neural network (CNN) was built and trained on labeled microscopic images of algae. The classification performance of the four models was quantified and compared to determine the best method for creating a robust and highly accurate algae classification model using minimal training data and to determine if transfer learning from the pre-trained models onto algae data was feasible. The ground truth data used to train, validate, and test these models consisted of 853 images spanning 25 algae classes. Overall, results showed that the CNN performed best for classifying algae. The CNN outperformed all three pre-trained models, suggesting that pre-trained models cannot transfer their learning enough after training on up to 20 examples to achieve high performance. The CNN was able to achieve 100% accuracy, precision, and recall with three convolutional layers, 3x3 kernels, 30 epochs, and a batch size of 64. The pre-trained models performed best when trained on 20 samples. Though the pre-trained models performed well after training on 20 samples in terms of accuracy, their precision and recall were low. Mask R-CNN had the highest accuracy of the pre-trained models (95.11%), followed by YOLO (94.77%), and SSD (91.12%). In conclusion, pre-trained models were not able to transfer their learning to algae classification with high enough performance after training on 20 examples. A custom CNN is the ideal model for algae classification.


# I. Introduction

Harmful algae can produce toxic effects on people, fish, shellfish, marine mammals, and birds [1]. Global research initiatives aim to detect harmful algae blooms and assess the effects of such harmful algae on the environment and ecosystem. One of the main methods with which this is accomplished is via collecting water samples and viewing them under a microscope to identify and count any harmful algae that may be present. However, this method is tedious, time-consuming, and requires a highly trained individual to detect and label the algae cells. One of the major bottlenecks when analyzing algae in water samples is counting and labeling the algae. Water samples can contain hundreds or thousands of algal cells, many of which look alike and are hard to classify without extensive knowledge and experience. Current practices involve taking pictures of each algal cell with a microscope for each water sample (tens to thousands of samples can be collected per project), labeling each algae image, and then counting how many algae were found in the samples. In addition, algae cells can be difficult to classify because one type of algae can look very different depending on the angle it is viewed at (Figure 1) and some species look very similar (Figure 2). In addition, there are hundreds of algae types, and it can be time consuming to research images of each type of algae and confirm its label.

<img width="600" alt="Screenshot 2025-04-30 at 9 54 33 AM" src="https://github.com/user-attachments/assets/f097c857-baf8-46af-8af1-0ef11cedb7a9" />


**Figure 1. Example of one algae species that can look very different.**

<img width="600" alt="Screenshot 2025-04-30 at 10 13 57 AM" src="https://github.com/user-attachments/assets/45c7b330-855c-420e-a071-21e847844551" />


**Figure 2. Example of two different algae species that look similar.**

An automatic deep learning (DL)-based algae classification model would significantly improve algae detection methods–both through detection accuracy and time saved. Using such a classification model would reduce the analysis bottleneck of counting and labeling algal cells and would reduce the dependency on labeling experts–freeing up their time for more additional research tasks. Users would be able to collect water samples, process them under a microscope (i.e., take pictures), pass them through a DL algae classification model, and obtain detailed information on the type (i.e., class) and number of algae in each sample. What previously took minutes or hours per sample would now take seconds with the power of DL. The overall goal of this project was to determine if pre-trained object detection models could be used to identify algae in microscopic images or whether a custom classification model should be built. The hypothesis was that the pre-trained models would be able to identify algae after a small amount of training with high accuracy (>95%) and would outperform the custom CNN since they can leverage transfer learning. I explored several DL methods for algae detection. Using the research findings allows us to draw conclusions on whether it is better to develop an algae classification model from scratch or using pretrained models. Results of this analysis were used to build a DL algae classification model that can be leveraged in research efforts. The contributions of this work include a customized DL algae classification model, a labeled algae dataset, and benchmark results.


# II. Description
Counting and labeling algae cells in water samples is a tedious, time consuming, and often difficult process. It can be difficult to distinguish the various types of algae species without expert knowledge and extensive experience, making this process even more time-consuming and difficult. A DL classification model could reduce this research bottleneck by automatically detecting, labeling, and counting algae in a sample.

In this project, four DL classification models were trained and assessed for their ability to accurately and reliably detect and classify (i.e., label) algae in microscopic images. First, a custom convolutional neural network (CNN) was built, trained, and tuned on microscopic images of algae. Various CNN architectures were assessed and the model hyperparameters were tuned specifically to classify algae. Since collecting and labeling images of algae for training and developing such a model would be a time-consuming and potentially costly task, the remaining assessed models were pre-trained models. Pre-trained detection/classification models are trained on thousands or millions of images spanning many classes. In theory, their learning could be transferred to images of algae with minor training on algae images labeled with new classes (i.e., species of algae). The pre-trained models included YOLO [2], Mask R-CNN [3], and SSD [4]. They were trained with zero-shot, single-shot, and multi-shot learning. Zero-shot learning enables a model to recognize new classes without prior exposure by leveraging semantic relationships. The model is tested on a new class of data without training on an example. Singleshot learning trains a model with only one example per class, making efficient use of limited data. Multishot learning uses multiple examples per class, providing robustness by capturing greater intra-class variability. Multi-shot learning in this project included training on five, 10, and 20 samples. The classification accuracy of the four models was quantified and compared to determine the best method for creating a robust and highly accurate algae classification model using minimal training data. A final network architecture and baseline results are described below.

## A. Data
The ground truth dataset used in this project was comprised of 28 seawater samples. Each sample was viewed under a microscope and images of any algal cells were captured (Figure 3).

<img width="600" alt="Screenshot 2025-04-30 at 10 04 52 AM" src="https://github.com/user-attachments/assets/3905a7fe-0ebb-417d-bb6b-46554498e415" />

**Figure 3. Example of algae cell captured in microscopic image.**

A total of 1,142 images were captured from these seawater samples. The data were then cleaned, meaning any blurry/non-visible images or images not containing algae (e.g., non-algal cells, debris) were discarded. After cleaning, 853 images remained in the dataset. These images were then labeled with the following 25 algae classes:
- Actinoptychus
- Bacillaria
- Biddulphia
- Centric Diatom
- Ciliate
- Coscinodiscus
- Cylindrotheca
- Dactyliosolen
- Diatom
- Dinoflagellate
- Entomoneis
- Euglenoid
- Fragilidium
- Hemiaulus
- Heterosigma Akashiwo
- Lyrella
- Navicula
- Nitzschia
- Odontella
- Paralia
- Pennate Diatom
- Pleurosigma
- Prorocentrum
- Tintinnid
- Tripos Hircus

The cleaned and labeled data were divided into training (70%), validation (15%), and testing (15%) datasets.


## B. CNN Hyperparameter Tuning

Various hyperparameters were assessed for the CNN, including kernel size, number of convolutional layers, number of epochs, and batch size. Overall, 16 different architectures were assessed (Table 1). The custom-built CNNs were trained on the entire training dataset and validated on the validation dataset. The model was then fed algae images from the testing dataset, and its ability to correctly label the algae were assessed.

**Table 1. CNN Hyperparameter Tuning.**
| CNN Name  | Kernel Size | Layers | Epochs | Batch Size |  
| --------- | ----------- | ------ | ------ | ---------- |
|    CNN1   |     3x3     |   3    |   30   |     32     |
|    CNN2   |     3x3     |   3    |   30   |     64     |
|    CNN3   |     3x3     |   3    |   50   |     32     |
|    CNN4   |     3x3     |   3    |   50   |     64     |
|    CNN5   |     3x3     |   5    |   30   |     32     |
|    CNN6   |     3x3     |   5    |   30   |     64     |
|    CNN7   |     3x3     |   5    |   50   |     32     |
|    CNN8   |     3x3     |   5    |   50   |     64     |
|    CNN9   |     5x5     |   3    |   30   |     32     |
|    CNN10  |     5x5     |   3    |   30   |     64     |
|    CNN11  |     5x5     |   3    |   50   |     32     |
|    CNN12  |     5x5     |   3    |   50   |     64     |
|    CNN13  |     5x5     |   5    |   30   |     32     |
|    CNN14  |     5x5     |   5    |   30   |     64     |
|    CNN15  |     5x5     |   5    |   50   |     32     |
|    CNN16  |     5x5     |   5    |   50   |     64     |


## C. Pre-Trained Model Training

The pre-trained models (YOLO, Mask R-CNN, SSD) were assessed for their ability to classify algae after zero-, single-, and multi-shot learning. During zeroshot learning, the models were not given any additional training. During single-shot learning, the models were trained on one image of each class of algae from the training dataset. During multi-shot learning, the models were trained on five, 10, and 20 images of each class of algae from the training dataset. The pre-trained models were then evaluated for their ability to classify algae with the testing dataset.

## D. Evaluation Metrics

All four models were evaluated for their classification accuracy with the following metrics:
- Accuracy = TP + TNTP + TN + FP + FN
- Precision = TPTP + FP
- Recall = TPTP + FN
where TP: true positive; TN: true negative; FP: false positive; FN: false negative.

# III. Evaluation 

The results from the CNN hyperparameter tuning and model assessments are outlined in detail below.

## A. CNN Results
The CNN model was assessed with various hyperparameters, as previously outlined in Table 1. Overall, CNN2, CNN4, and CNN10 were the best performing models with accuracies, precisions, and recalls of 100%. Results from all 16 CNNs are outlined in Table 2.

**Table 2. CNN performance results.**
| CNN Name  | Accuracy  | Precision |  Recall   | 
| --------- | --------- | --------- | --------- |
|    CNN1   |  98.96%   |   90.90%  |   82.35%  |    
|    CNN2   |**100.00%**|**100.00%**|**100.00%**|  
|    CNN3   |  99.86%   |   98.81%  |   97.65%  |  
|    CNN4   |**100.00%**|**100.00%**|**100.00%**|  
|    CNN5   |  98.21%   |   89.83%  |   62.35%  |
|    CNN6   |  98.48%   |   98.33%  |   66.67%  |
|    CNN7   |  99.44%   |   95.06%  |   90.59%  |
|    CNN8   |  99.62%   |   95.24%  |   95.24%  |
|    CNN9   |  96.94%   |   69.23%  |   42.35%  |
|    CNN10  |**100.00%**|**100.00%**|**100.00%**| 
|    CNN11  |  96.24%   |   60.87%  |   16.47%  |
|    CNN12  |  99.62%   |   95.24%  |   99.62%  |
|    CNN13  |  96.00%   |   0.00%   |   0.00%   |
|    CNN14  |  96.00%   |   0.00%   |   0.00%   |
|    CNN15  |  96.00%   |   0.00%   |   0.00%   |
|    CNN16  |  96.00%   |   0.00%   |   0.00%   |

Of the top performing models, CNN2 was the fastest and least computationally expensive and was therefore selected as the best CNN architecture for algae classification in this project. The training/validation loss and training/validation accuracy of CNN2 can be seen in Figures 4 and 5, respectively. Figure 4 suggests that the model did not overfit since validation loss continues to decline.

<img width="600" alt="Screenshot 2025-04-30 at 10 20 36 AM" src="https://github.com/user-attachments/assets/5ee0f43c-f980-4534-af09-9238330b65fa" />

**Figure 4. Training (loss) and validation (val_loss) loss of the CNN2 model.** 

<img width="600" alt="5" src="https://github.com/user-attachments/assets/6dd0bf14-3cf2-4c1c-8c12-1f2d52b87f24" />


**Figure 5. Training (accuracy) and validation (val_accuracy) accuracy of the CNN2 model.**


## B. YOLO Results
The YOLOv5 [5] implementation was selected as the backbone YOLO architecture (Figure 6) since it has models that specifically support classification rather than object detection. 

<img width="487" alt="6" src="https://github.com/user-attachments/assets/55b5982f-e4c5-4f50-956a-8efa819a1b2f" />
**Figure 6. YOLO architecture [2].**

Initial exploratory assessment revealed that the YOLO validation loss plateaued after approximately 10 epochs, so the model was trained with 10 epochs during the single- and multishot learning. Results from the YOLO training can be found in Table 3.

**Table 3. YOLO performance results.**
| Training  | Accuracy  | Precision |  Recall   | 
| --------- | --------- | --------- | --------- |
| Zero-shot |   0.00%   |    0.00%  |   0.00%   |    
|Single-shot|  92.69%   |   8.64%   |   8.64%   |  
|  5-shot   |  93.58%   |   19.75%  |   19.75%  |  
|  10-shot  |  92.79%   |   9.88%   |   9.88%   |  
|  20-shot  |**94.77%** |**34.57%** |**34.57%** |


Overall, multi-shot training with 20 samples produced the best results. Zero-shot learning had 0% accuracy, precision, and recall (Figure 7). However, those results were not that surprising since the algae classes are very specific. YOLO was not trained on this kind of data and had no reference to what a specific algae species was. Model performance increased with the number of training samples, with the exception of 10-shot learning. Improved performance with more training examples is not that surprising of a result; however, it is interesting that 10-shot learning decreased performance when compared to 5-shot learning.

<img width="506" alt="7" src="https://github.com/user-attachments/assets/27db859c-911d-4737-9410-8801588dd25b" />


**Figure 7. Example of classification results with zero-shot learning. The YOLO model was not able to identify the algae as a specific species or even identify it as algae, since it had not been trained on any images containing algae classes.**


## C. Mask R-CNN Results

Initial exploratory assessment revealed that the Mask R-CNN (Figure 8) validation loss plateaued after approximately 10 epochs, so the model was trained with 10 epochs during the single- and multi-shot learning. 

<img width="501" alt="8" src="https://github.com/user-attachments/assets/bbaa7644-d3d5-490a-b695-453e7da57033" />


**Figure 8. Mask R-CNN architecture [3].**

Results from the Mask R-CNN training can be found in Table 4.

**Table 4. Mask R-CNN performance results.**
| Training  | Accuracy  | Precision |  Recall   | 
| --------- | --------- | --------- | --------- |
| Zero-shot |   0.00%   |    0.00%  |   0.00%   |    
|Single-shot|  91.63%   |   11.24%  |   8.19%   |  
|  5-shot   |  91.52%   |   14.29%  |   18.62%  |  
|  10-shot  |  93.29%   |   24.19%  |   22.91%  |  
|  20-shot  |**95.11%** |**42.10%** |**34.38%** |

Overall, multi-shot training with 20 samples produced the best results. Zero-shot learning had 0% accuracy, precision, and recall (Figure 9). However, those results were not that surprising since the algae classes are very specific. Mask R-CNN was not trained on this kind of data and had no reference to what a specific algae species was. Model performance increased with the number of training samples.


<img width="504" alt="9" src="https://github.com/user-attachments/assets/80ce74e0-9fe8-4ced-9ad1-f4ef827fd090" />

**Figure 9. Example of classification results with zeroshot learning. The Mask R-CNN model was not able to identify the algae as a specific species or even identify it as algae, since it had not been trained on any images containing algae classes.**


## D. SSD Results

Initial exploratory assessment revealed that the SSD (Figure 10) validation loss plateaued after approximately 10 epochs, so the model was trained with 10 epochs during the single- and multi-shot learning. 

<img width="505" alt="10" src="https://github.com/user-attachments/assets/d8850afd-8c91-4815-a903-decd9022c2c3" />
**Figure 10. SSD architecture [4].**

Results from the SSD training can be found in Table 5.

**Table 5. SSD performance results.**
| Training  | Accuracy  | Precision |  Recall   | 
| --------- | --------- | --------- | --------- |
| Zero-shot |   0.00%   |    0.00%  |   0.00%   |    
|Single-shot|   45.49%  |   15.69%  |  15.69%   |    
|  5-shot   |  72.36%   |   18.26%  |   18.27%  |  
|  10-shot  |  89.69%   | **34.42%**|   42.21%  |  
|  20-shot  |**91.12%** |  32.69%   |**56.59%** |

Overall, multi-shot training with 20 samples produced the best results. Zero-shot learning had 0% accuracy, precision, and recall (Figure 11). However, those results were not that surprising since the algae classes are very specific. SSD was not trained on this kind of data and had no reference to what a specific algae species was. Model performance increased with the number of training samples, similarly to Mask RCNN.


<img width="492" alt="11" src="https://github.com/user-attachments/assets/97f2502f-d65a-4560-a96f-72ca52e388d5" />

**Figure 11. Example of classification results with zero-shot learning. The SSD model was not able to identify the algae as a specific species or even identify it as algae, since it had not been trained on any images containing algae classes.**

## E. Model Comparisons
The CNN2 had the highest accuracy of all the models assessed (Table 6). Of the pre-trained models, Mask R-CNN trained with 20 algae images performed the best. Mask R-CNN had the highest accuracy and precision of the pre-trained models, and SSD had the highest recall.

**Table 6. Results of top performing models.**
|   Model   | Accuracy  | Precision |  Recall   | 
| --------- | --------- | --------- | --------- |
|    CNN    |**100.00%**|**100.00%**|**100.00%**|
|   YOLO    |   94.77%  |   34.57%  |  34.57%   |    
|Mask R-CNN |  95.11%   |   42.10%  |   34.38%  |  
|    SSD    |  91.12%   |   32.69   |   56.59%  |  


# IV. RELATED WORK

The development of DL models for algae classification, particularly through the use of pretrained object detection models and custom CNNs, intersects with broader research trends in automatic image classification, transfer learning, and the use of pre-trained models and small example datasets.

While a custom-build classification model, such as the CNN developed in this project, has the possibility to achieve high performance on the very particular task of algae classification, it can require a massive representative dataset to train and fine-tune. Pretrained models, on the other hand, offer a promising avenue for the development of accurate and robust classification tools with the use of limited training data. However, these models can be limited by the specificity of the training data and the nature of the transfer learning.

Further research into optimizing these models for specific types of biological data, combined with advancements in transfer learning methodologies, could enhance their applicability and accuracy in complex image classification tasks. Representative works related to this project include studies on transfer learning applications and specialized image classification systems for biological samples.

## A. Transfer Learning for Image Classification

Researchers have extensively explored transfer learning, where a model developed for one task is repurposed for a different but related task. This approach is well-suited for domains where labeled data are scarce or expensive to obtain. Notably, transfer learning has been leveraged successfully in medical image analysis, as demonstrated by Shin et al., who used pre-trained CNNs to detect thoracoabdominal lymph nodes and interstitial lung disease with significant success [6]. However, the efficacy of transfer learning heavily depends on the similarity between the source and target tasks. A common drawback is that the transferred features might not be optimal for new tasks that significantly deviate from the original data domain.

## B. Algae Classification Models

Several models have been developed for algae classification. For example, Ai et. al. showed 89.6% and 86.0% accuracy in identifying harmful algae blooms with a random forest and long short-term memory model, respectively [7]. Though these results are promising, it should be noted that these models were classifying algae blooms and not individual cells. Zhou et. al. showed mean average precision of algae detection on six classes of cells in a variety of DL models (e.g., Faster R-CNN, SSD, YOLOv3) ranging from 24.6% to 70.9% [8]. These models were trained on a much larger dataset (>2,000 images) than the one used in this project; however, the dataset was limited to only 6 classes. In addition, the precision of these models is low to average, suggesting that transfer learning via these models may not be as good as the custom-built CNN described in this project. Hawezi trained a CNN on 2,749 microscopic algae images and leveraged transfer learning via an AlexNet and MobileNetv2 [9]. However, this work does not outline quantitative metrics on the performance of the model and only categorizes the algae into four broad categories (green vegetative, red vegetative, green cyst, red cyst) rather than identifying individual algae species.

These models offered improvements over manual methods but were not accurate or specialized enough to act as public algae classification tools. More research in this area is needed, which this project aims to address.


## C. Pre-Trained Models in Biological Imaging.

The use of pre-trained models has been explored in different contexts of biological imaging [10]. These models generally excel when fine-tuned with a substantial amount of domain-specific data. While these models demonstrate high accuracy in specific settings, their adaptation to new domains (e.g., algae classification) requires careful tuning and sufficient training data, often necessitating adjustments to the network architecture or training process to improve specificity and sensitivity.


# V. SUMMARY AND CONCLUSIONS

Four DL models were assessed for their ability to classify algae in microscopic images: CNN, YOLO, Mask R-CNN, and SSD. The CNN was designed, trained, validated, and tuned specific to algae images. Pre-trained models (YOLO, Mask R-CNN, SSD) were trained with zero-, single-, and multi-shot learning to determine if transfer learning could be leveraged to build a classification model with minimal algae data.

Overall, this research showed that a custom-built CNN performed best for classifying algae in microscopic images. The CNN unexpectedly outperformed all three pre-trained models, which were hypothesized to perform better than the CNN since they could theoretically transfer their pretrained learning onto the algae images.

Multiple CNN architectures were assessed, and three achieved 100% accuracy, precision, and recall. CNN2 was the smallest and least computationally expensive of these three models but still achieved the same performance. Therefore, it is considered the top choice for an algae classification architecture of the ones assessed in this project. The pre-trained models were expected to outperform the CNN since they can leverage transfer learning. However, they did not perform as well as the CNN. While they were fairly accurate, their precision and recall were low. However, it is possible that these models could perform better with more training data (e.g., training on the entire ground truth dataset). In addition, it is possible that the pre-trained models generalize better than the CNN to other microscopic images (that may be similar to the ground truth dataset, but captured under different conditions) since they can leverage their inherent knowledge and semantic understanding.


## A. Conclusions

In conclusion, a CNN trained on algae imagery was the best performing classification model and outperformed the pre-trained models. The CNN was able to achieve 100% accuracy, precision, and recall with three convolutional layers, 3x3 kernels, 30 epochs, and a batch size of 64. Though the pretrained models performed well after training on 20 samples in terms of accuracy, their precision and recall were low. It seemed that these models were not able to transfer their learning to algae classification well when trained with up to 20 images. However, it is possible that, with more training, these models can perform as well as the CNN. In addition, it is possible that the pre-trained models would generalize well to other microscopic algae images.


## B. Future Work

Results of this project left several additional research questions that would be interesting to assess in future work. First, it would be interesting to train, validate, and test all four models on more data. While the CNN performed surprisingly well with a ground truth dataset only containing 853 images, 853 images is not a large dataset when developing DL models, particularly since this data was split between 25 classes. It would be interesting to characterize and compare model performances after training on a larger and more variable dataset.

Second, it would be interesting to compare the performance of the CNN to the pre-trained models if they were trained on the entire algae ground truth dataset. While the pre-trained models did not perform as well as the CNN, their performance improved with additional training samples (e.g., 20-shot trained models outperformed single-shot trained models). It is possible that the pre-trained models would perform as well as the CNN.

Third, all four models could be tested on additional data, particularly algae data captured by others at different labs and under different conditions. While microscopic images will all be captured similarly, variations in environmental conditions, sample processing, microscopes, and image capture could affect the performance of the classification models. It is possible that the model performance would decrease when tested on a larger and more variable algae dataset. In addition, it would be interesting to see if the pre-trained models generalize better to additional algae data than the CNN since they can leverage their inherent weights and knowledge.


# REFERENCES
[1] https://www.noaa.gov/what-is-harmful-algalbloom

[2] https://arxiv.org/pdf/1506.02640.pdf

[3] https://arxiv.org/pdf/1703.06870.pdf

[4] https://arxiv.org/pdf/1512.02325.pdf

[5] https://github.com/ultralytics/yolov5

[6] https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-022-00793-7

[7] https://www.sciencedirect.com/science/article/pii/S0043135423001458

[8] https://arxiv.org/pdf/2211.07546.pdf

[9] https://iwaponline.com/wqrj/article/45/4/413/39738/Algae-based-Biomonitoring-Predicting-Diatom

[10] https://www.sciencedirect.com/science/article/pii/S0002944021002613




