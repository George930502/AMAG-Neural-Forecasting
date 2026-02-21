# NSF HDR SCIENTIFIC MODELING OUT OF DISTRIBUTION: NEURAL FORECASTING

## Intro

Understanding the mechanisms of neural activity is important for diagnosing neurological disorders at early stages, devising effective treatment plans, and helping patients regain movement abilities [1,2,3]. Among the various ways to study these mechanisms, analyzing the dynamics of neural activity offers a unique perspective on how neurons interact to perform specific functions. Such dynamic properties also pave the way for decoding signals into observable behaviors.

However, most existing approaches to learning neural dynamics focus on modeling concurrent (i.e., immediate) neural activity [4,5], with comparatively little attention paid to predicting future neural dynamics. Predicting the future neural dynamics is challenging, particularly when the observed activity is incomplete, and additional day-to-day or hour-to-hour drifts in the recording array add further variability.

Prior work [6] addressed a simplified scenario by estimating future neural activity using training and testing data collected on the same day to avoid the complexities of day-to-day drifts. In this challenge, we extend that dataset to explore the more difficult task of predicting future neural activity across multiple days, capturing the additional variability introduced by these drifts.

## Problem setting: Neural Forecasting

We forecast the activations of a cluster of neurons given previous signals from the same cluster. This targets the critical problem of brain-artificial neuron interfaces, and these models can be used in brain-chip interfaces for artificial limb control, amongst many others.

## Challenge target (Important)

**Learning the Neural Dynamics through Prediction:**

Neural activities are recorded in the form of multivariate time series. Previous studies investigated neural dynamics using neural activities in a fixed time window [4,7]. We challenge participants to propose methods to measure the changes in neural dynamics from recorded neural activity, such that the trained model can predict future activities given past neural activities.

**Generalization of Predicting Neural Activity in Unseen Sessions:**

Validating the trained model on a new recording session poses an additional challenge due to changes in the recorded neuron sets and changes in the status of the recording technique. Here, we encourage participants to propose methods that have good generalization ability to a new session. The ability to predict neural activities in a new session has great potential for building future low-latency daily-use BCIs.

## Datasets

The motor neural activity forecasting dataset includes recorded neural signals from two monkeys performing reaching activities, Monkey A and Monkey B, using
μ
μECoG arrays. Recorded neural signals are in the form of multivariate continuous time series, with variables corresponding to recording electrodes in the recording array.

The dataset includes all 239 electrodes from Monkey A and 87 electrodes specifically from the M1 region of Monkey B.

### Dataset format

The dataset provided follows the shape:

Neural_data: N *T* C *F ( Sampe_size* Time_steps *Channel* Feature )

Sampe_size: varies depending on the dataset. The exact number is summarized in the next section

Time_steps: Each sample will have 20 time steps recorded. The model is expected to take the first 10 steps as input and predict the following 10 steps.

Channel: The number of electrodes, which depends on the Monkey. 239 electrodes from Monkey A and 87 electrodes from Monkey B

Feature: There are nine features provided. The first feature ([0]) is the final prediction we want the model to take as input and predict. All the remaining features ([1:]) are the decomposition of the original feature in different frequency bands.

### Training dataset

We provide:

* 985 training samples for Monkey A (affi) (see dataset/train/train_data_affi.npz)
* 700 training samples for Monkey B (beignet) (see dataset/train/train_data_beignet.npz)

Additional sample records from different dates were provided:

* 162 training sample records from Monkey A (see dataset/train/train_data_affi_2024-03-20_private.npz)
* 82 + 76 training sample records from Monkey B (see dataset/train/train_data_beignet_2022-06-01_private.npz and dataset/train/train_data_beignet_2022-06-02_private.npz)

### Testing data

A hold-out dataset is used to evaluate model performance on Codabench.  

* 122 + 162 samples from Monkey A (see dataset/test/test_data_affi.npz and dataset/test/test_data_affi_2024-03-20_private.npz)
* 87 + 82 + 76 samples from Monkey B (see dataset/test/test_data_beignet.npz and dataset/test/test_data_beignet_2022-06-01_private.npz and dataset/test/test_data_beignet_2022-06-02_private.npz)

### Final secret dataset

Another set of secret datasets will be used to evaluate the final ranking of the competition.

## Starting kit and example submission

A Google Colab notebook is provided for the participant to explore and train new ML models. The participant is encouraged to copy and make modifications for their training.

Models must be trained on up-to-date versions of TensorFlow/PyTorch/Scikit-learn/etc. An example of the intended format of a submission is:

```python
import torch
import os

class Model:
 def __init__(self, monkey_name=""):
 # You could include a constructor to initialize your model here, but all calls will be made to the load method
 self.monkey_name = monkey_name
 if self.monkey_name == 'beignet':
 # Load setting for beignet models 
 self.input_size = 89
 elif self.monkey_name == 'affi':
 # Load setting for affi models 
 self.input_size = 239
 else:
 raise ValueError(f'No such a monkey: {self.monkey_name}')
            
 def predict(self, X):
 # This method should accept an input of any size (of the given input format) and return predictions appropriately with the shape (Sample_size * Time_steps * Channel)

 return # Something

 def load(self):
 # This method should load your pre-trained model from wherever you have it saved
 path = "model.pth"
 if self.monkey_name == 'beignet':
 path = "model_beignet.pth"
 elif self.monkey_name == 'affi':
 path = "model_affi.pth"
 else:
 raise ValueError(f'No such a monkey: {self.monkey_name}')
 self.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), path), weights_only=True))
 
```

The essential functions are predict and load, the former should return an array of predicted probabilities in the shape of (Sample_size *20* Channel). The latter should load a pre-trained model; any auxiliary files necessary, such as "config.json" and "model.weights.h5", should also be included in the submission. The submission will be a zipped file (or files). The only required file to be included in any submission is one of the above formats, named "model.py". There is no restriction on library usage.

### Expected model inputs

The model will take an numpy array of shape (Sampe_size *20* Channel * Feature)

Only the first 10 steps have meaningful value. The last 10 steps are masked and repeat the 10th step's values.

### Expected model outputs & target

The model will be expected to return a numpy array of shape (Sample_size *20* Channel), which should contain the first 10 steps and the predicted next 10 steps.

Only the first feature is the target for the prediction. All the remaining features ([1:]) are the decomposition of the original feature into different frequency bands.

## Evaluation

Models will be evaluated on a combined metric of Mean squared error (MSE). MSE measures absolute discrepancy between predicted neural signals and the recorded neural signals. The goal is to minimize the total MSE across all 5 test datasets.

## Reference

[1] A Bolu Ajiboye, Francis R Willett, Daniel R Young, William D Memberg, Brian A Murphy, Jonathan P Miller, Benjamin L Walter, Jennifer A Sweet, Harry A Hoyen, Michael W Keith, et al. Restoration of reaching and grasping movements through brain-controlled muscle stimulation in a person with tetraplegia: a proof-of-concept demonstration. The Lancet, 389(10081):1821–1830, 2017.

[2] Jacques J Vidal. Toward direct brain-computer communication. Annual review of Biophysics and Bioengineering, 2(1):157–180, 1973.

[3] Jonathan R Wolpaw, Niels Birbaumer, Dennis J McFarland, Gert Pfurtscheller, and Theresa M Vaughan. Brain–computer interfaces for communication and control. Clinical neurophysiology, 113(6):767–791, 2002.

[4] Trung Le and Eli Shlizerman. Stndt: Modeling neural population activity with spatiotemporal transformers. Advances in Neural Information Processing Systems, 35:17926–17939, 2022.

[5] Chethan Pandarinath, Daniel J O'Shea, Jasmine Collins, Rafal Jozefowicz, Sergey D Stavisky, Jonathan C Kao, Eric M Trautmann, Matthew T Kaufman, Stephen I Ryu, Leigh R Hochberg, et al. Inferring single-trial neural population dynamics using sequential auto-encoders. Nature methods, 15(10):805–815, 2018.

[6] Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, and Eli Shlizerman. Amag: Additive, multiplicative, and adaptive graph neural network for forecasting neuron activity. Advances in Neural Information Processing Systems, 36:8988–9014, 2023.

[7] Joel Ye and Chethan Pandarinath. Representation learning for neural population activity with neural data transformers. arXiv preprint arXiv:2108.01210, 2021.
