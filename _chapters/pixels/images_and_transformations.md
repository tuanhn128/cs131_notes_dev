---
title: ML for CV
keywords: (insert comma-separated keywords here)
order: 2 # Lecture number for 2020
---


# Lecture #15: ML for CV: A Brief Overview

KK Barrows, Nandini Naidu, Tuan Nguyen, AJ Rossman, Xuchen Wei

- [Why CV Needs ML](#first-big-topic)
	- [How Humans Interpret the World](#subtopic-1-1)
	- [Understanding Pixels](#subtopic-1-2)
	- [Key Considerations when using ML](#subtopic-1-3)
- [ML Foundations](#second-big-topic)
	- [Data](#subtopic-2-1)
	- [Types of ML problemsn](#subtopic-2-2)
	- [ML Models](#subtopic-2-3)
- [ML Applications in CV](#third-big-topic)
	- [ML Problems in CV](#subtopic-3-1)
	- [Tasks](#subtopic-3-2)
	- [Applications](#subtopic-3-3)

<a name='first-big-topic'></a>
## 1 Why CV Needs ML

Throughout this quarter, it has been shown how computer vision can accurately and effectively make

sense of the world around us using geometry, linear algebra, or other various methods and algorithms

(ex: Canny edge detector). So, the question must be asked: why would machine learning techniques

be helpful in solving computer vision problems?

<a name='subtopic-1-1'></a>
### 1.1 How Humans Interpret the World

First, it is important to understand how humans see the world around us. This might give us a clue as

to why ML could be useful in the ﬁeld of computer vision.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/optical illusion.png">
  <div class="figcaption">Figure 1: An optical illusion where the shapes in the picture appear to be moving.</div>
</div>

Using the example of the picture above, it is apparent that we interpret images using more than just

our eyes. In fact, we see more with our brain than we do with our eyes. We use our brains to better

interpret and recognize the world around us. We can apply our past memories and experiences to

understand the content of an image. In other words, we learn how to see! Using ML with CV would

allow us to do the same thing: apply past knowledge and data to help better understand a current

image.

<a name='subtopic-1-2'></a>
### 1.2 Understanding Pixels

Computers cannot "see" in the same way that we do. Images for computers are simply pixel arrays

that contain RGB values. For many years, computer scientists had trouble programming computers

to make sense of these numbers to understand the context of an image. In the real world, images are

often noisy and imperfect. This makes it hard to apply strict rules and conditions to solve problems in

C V.

But, with ML, we are able to solve problems without explicit programming. As mentioned in the past

section, ML allows the computer to learn by example and compress data into patterns. This way, even

in noisy and imperfect conditions, computers have enough context about the problem to correctly

interpret the scene.

<a name='subtopic-1-3'></a>
### 1.3 Key Considerations when using ML

ML can be an extremely powerful tool. In order to use ML in an effective and safe manner, it is

important to keep these considerations in mind:

\- Every good ML model starts with great data. It is important to ﬁnd reliable, trustworthy data that is

accurate and minimizes noise. This is covered in more detail later.

\- The cost of creating ML models is often high (labels, compute, time, societal, etc).

\- Be cautious not to simply try and brute-force a solution. Many times, if you are not getting the

solution you want, it can be tempting to just randomly change things until you seemingly ﬁnd a

solution that ﬁts. Take the time to understand the problem you are trying to solve and the algorithm

you are trying to use.

\- It is important to be aware of coder biases when creating ML models. ML carries a heavy ethical

burden and the decisions these algorithms make can sometimes be in life or death situations (ex:

self-driving cars)

<a name='second-big-topic'></a>
## 2 ML Foundations

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/ml_equation.png">
  <div class="figcaption">Figure 2: A generalized equation describing ML.</div>
</div>



This equation formally describes what ML is. The overall goal is to create a model that accurately

predicts our outcome given a set of data. \\( \theta^* \\) is the parameter that holds the best model parameters, i.e.

the settings that make our model most accurate.

In short, \\(\theta^*\\) is the \\(\textit{argmin}_\theta \\) over the expectation of our data distribution of the loss function, so it’s the

parameter that minimizes the expectation of the loss function. The loss function compares our model

to the ground truth. We get this by optimizing over a dataset, which is sampled from the distribution,

D. The lower the expectation of the loss function, the more accurate our model is.




<a name='subtopic-2-1'></a>
### 2.1 Data

As previously discussed, understanding how data comes into this equation will summarize much of

what we need to know about machine learning. In Figure 1, the expectation, random variables, data

distribution, dataset size, and sample data terms interact with our dataset.

D represents the data source, the entire collection of real world data that we want to model. The

expectation is the integral over the probability distribution of D. More speciﬁcally, this is a distribution

over (x, y) ∈ D where x and y are random variables which represent x as the input and y as the output.

We want to ﬁnd a relationship between x and y, or rather a function f(x) = y that approximates this

relationship.

The expectation is an integral over the whole distribution of D, or an inﬁnite sum. We cannot deﬁne

this entire distribution because it is from the real world. This also means that we cannot compute

the expectation. However, we can sample from the world. We take n samples from the real-world

distribution D and compute a sum over all \\( (x_i, y_i) \\) where 1 ≤ i ≤ n. In other words, we are creating

a dataset of size n made up of sample data from the real world, i.e. our distribution D. We can then

approximate our expectation, our inﬁnite sum, with a ﬁnite sum.

To build our dataset of size n, we begin with our unknown distribution D. We sample (x , y ) ∼ D

to build the dataset. In ML, we must assume that the data we sample is independent and identically

distributed (i.i.d.), meaning the \\( (x_i, y_i) \\) pairs have no correlation other than the fact that they are

from the same dataset. This is a fundamental assumption in ML, however, this is not always a fair

assumption. Seeing as humans run data collection, it is entirely possible that we bias the data at some

point along the way, from collection to output labelling.

#### 2.1.1 Datasets

In ML, we strive for generalization. We want our models to work well on generalized and unseen

data. This means that when we are building our dataset, we want it to be representative of the real

world so that it can better approximate the expectation of D. The less our dataset reﬂects the real

world distribution, the worse our approximation will be.

Generalization is a challenge because we only have a *ﬁnite* sample \\( \hat{D}=\{(x_i, y_i)\sim D, i=1:n\} \\).

This brings us to an important question: how big should the dataset be? Sometimes we have cues,

like convergence bounds or scaling laws, that tell us how big n should be. Convergence bounds tell

us how fast our error decreases as n increases. However, convergence bounds are not always helpful

as they are often loose. The empirical equivalent of convergence bounds is scaling laws. Scaling laws

let us forecast the performance of a model as we increase n, the number of samples in the dataset.

Often times, convergence bounds and scaling laws are either unreliable or unavaible. In this case, the

size of a dataset depends on your problem, model, data, etc. so we experiment to ﬁnd an optimal n.

While it is important that we collect enough data, we must also keep in mind quality and diversity. It

is not always better to have a bigger dataset if it is not diverse or of high quality.

#### 2.1.2 Statistical hygiene

We can estimate generalization, i.e. increase accuracy, by splitting our dataset into three sections: a

training set, a validation set, and a test set.

Training set: The training set usually contains most of our dataset, about 50-80% depending on the

size of the dataset. We run the training set through the model many times to learn and tune our model

parameters, \\( \theta^* \\).

Validation set: The validation set should be a small representative fraction of our dataset, approx-

imately 10-20%. We run the validation set a couple of times to select models and settings for the

algorithm hyper-parameters. Hyper-parameters are not part of θ∗, they are variables that we set, like

the learning rate in stochastic gradient descent or the number of layers in a deep net. The validation

set must be high quality, meaning it’s not noisy and it has good coverage, because it steers your

design decisions.

Test set: The test set should be the same size or bigger than your validation set. We only look

at the test set once because we use it to evaluate our model and need the data to be unseen. The

test set should only be seen after many rounds of training and validation. Otherwise, we may

have statistical leakage, meaning we have explicitly learned what the test set contains making our

evaluation inaccurate.

#### 2.1.3 The importance of datasets

Building datasets is generally the hardest part of ML. We build the dataset upstream of the whole

project and any mistakes made at this stage hurt a lot, a lot later. It is also computationally expensive

to build a dataset seeing as the process involves building a data pipeline, collecting data, labeling

data, auditing data, etc.

Datasets are built by humans! Decisions surrounding dataset building are some of the most important

decisions in ML.

Standardization of benchmarks is key for reproducibility. It enabled the machine learning community

to grow quickly and accelerated tremendously the path of progress. Standardization allowed people

to skip 90% of the work by borrowing datasets as well as giving us the ability to directly compare

models.

There are many datasets available on line for research. There are more than 5,000 datasets on

[paperswithcode.com.](https://paperswithcode.com/)


<a name='subtopic-2-2'></a>
### 2.2 Types of ML problems

#### 2.2.1 Supervised Learning

Supervised learning is the most common paradigm for machine learning. Supervised learning is

learning from fully labeled examples. In other words, you are given in the training data both the

inputs and the desired outputs. On supervised learning task is classiﬁcation, where we try to assign

a categorical label to an image. Another supervised learning task is regression, where we assign a

continuous value to an image. Multi-task learning involves learning multiple related tasks jointly.

Few-shot learning occurs when training set is limited to just a few examples. The most extreme case

is one-shot, where you seek to learn from only a single example.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/catordog.png">
  <div class="figcaption"> Figure 3: An example of a classiﬁcation task would be whether this image is of a dog. An example of

a regression task would be the weight of the dog. </div>
</div>


#### 2.2.2 Unsupervised Learning

Unsupervised learning learns from unlabeled data rather than labeled data. One common task of

unsupervised learning is clustering, which is the process of grouping data into "clusters." These

clusters might have some sort of underlying meaning, for example, a clustering algorithm run on

photos of animals might split the data into groups of dogs, cats, etc. However, the dataset is not labeled

with these identiﬁcations beforehand. Other types of unsupervised learning include dimensionality

reduction and outlier detection

Unlike supervised learning, which optimizes for accurate "predictions" trained on labeled data, the

goal of unsupervised learning is to parse meaning or underlying patterns from unlabeled data.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/unsupervised_learning_example.png">
  <div class="figcaption"> 
Figure 4: The output of a clustering algorithm which grouped "similar" animals together but was not

explicitly given speciﬁc labels to assign.
</div>
</div>

#### 2.2.3 Semi-Supervised Learning

Semi-supervised learning is when we have a partially labeled training set. One case of this is active-

learning where the model is ﬁrst trained on current data and then new unlabeled data is added. The

ones which it performs poorly on are then labeled, and fed back into training the model. Another case

of this is with auto-labeling, such as labeling videos where the labels are auto-propagated between

frames.

#### 2.2.4 Transfer Learning

Transfer learning involves the process of re-using learned knowledge to a new problem. For example,

we may use a model trained for one task and apply it to another task. A case of this would be an

image classiﬁcation model for whether an image is of a car that we try to generalize to trucks.

#### 2.2.5 Domain Adaptation

Domain adaptation involves learning on one (source) data distribution and aiming to generalize to

another (target) data distribution. The idea is that the two data distributions are similar in some way

so one distribution can be aligned to another. Some examples include generalizing from daytime to

nighttime or generalizing from Japan to the united States.

#### 2.2.6 Meta-Learning

Meta-learning involves learning from multiple related datasets how to quickly learn a new model on

a new dataset. Here we are aiming to ﬁnd a procedure which can quickly generate a new model for a

given dataset. This is a popular area of current ML research.

#### 2.2.7 Self Supervised Learning

Self supervised learning is the paradigm of learning from an unlabeled dataset along with some form

of "proxy supervision" which is also derived from that unlabeled dataset. This proxy supervision

refers to some sort of additional use of our dataset to alter the way our model is trained.

For example, proxy supervision could entail the process of omitting a certain part of the dataset such

as the color of each image, and then using the colored images as labels for their corresponding grey

scale versions. This type of procedure would allow for the training of a supervised learning model to

predict these newly created labels, which all were derived from the original unlabeled dataset.

#### 2.2.8 Reinforcement Learning

Reinforcement learning refers to the area of ML involving building a sequential decision-making

policy (recommending actions based on current state) in order to maximize some cumulative reward

in an environment. For example, a reinforcement learning model could be trained to look at the board

of a board game (state) and recommend an optimal move (action) for highest future win probability

(reward). This learning paradigm was famously used by Google’s DeepMind in order to beat a human

world champion in Go, a game formerly thought to be too complex for computers to succeed at the

highest level.

Reinforcement learning can be done online, in which a model is suggesting actions and then observing

and training on the results. It can also be done ofﬂine, in which a model instead trains on data from

past demonstrations.

#### 2.2.9 Deep Learning

Deep learning refers to the ﬁeld of ML which utilizes neural networks, models which consist of a

series of layers, each layer transforming the previous layer (or input) eventually resulting in the desired

output. These layers working in tandem can allow for very complex patterns and relationships in the

training data to be modeled. Deep neural networks also allow for "end to end learning" approaches

which forgo traditional intermediate processing steps for one large neural network. While Deep

Learning can be very powerful, it also comes with risks of overﬁtting, large data and computation

requirements, and ethical issues associated with very large datasets.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/e2e_learning_example.png">
  <div class="figcaption"> 

Figure 5: A diagram representing end-to-end learning compared to traditional CV approaches.
</div>
</div>

<a name='subtopic-2-3'></a>
### 2.3 ML Models

Choosing the correct ML models depends ﬁrst and foremost on the data type of both x and y. From

there there are common categories of models to choose from including classiﬁcation models like

K-NN(nearest neighbors), linear models like linear regression, clustering models like KMeans,

probabilistic models like Naive Bayes, etc. An additional consideration is that deep models seem to

have (implicit) assumptions that ﬁt many CV problems very well.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/ML_model_highlight.png">
  <div class="figcaption"> 

Figure 6: The generalized ML equation highlighting the model variable.
</div>
</div>

#### 2.3.1 Deep Learning (DL) Models

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/DL_model_pic.png">
  <div class="figcaption"> 

Figure 7: Generalized ﬂow chart of a DL Model.
</div>
</div>

Deep Learning models are essentially a composition of layers (like linear function layers or sigmoid

layers). Some examples of Deep Learning models include Neural Nets which have been around for

decades, and are coarsely inspired from biological systems. There are also Deep Nets which combine

the use of multiple layers, architectural innovations, GPU, and data in order to form a deep learning

model. In general DL models use many combinations of types of layers and architectural variations

in practice.

#### 2.3.2 MLP: Multi-Layer Perceptron

A perceptron is deﬁned as \\( f(x; w, b) = sgn(w^\top x + b \\). MLPs are fully connected nets that take on

the general form of \\( f(x;\theta) = g_{\theta 1} \circ \cdots \circ g_{\theta N}(x) \\). The layers of MLPs include linear weighting

and activation functions. For MLPs the Universal Approximation Theorem applies meaning that an

arbitrary width/depth MLPs can approximate any function. Some modern MLPs include NeRF and

MLP-Mixer.

#### 2.3.3 CNN: Convolutional Neural Network

CNNS are a composition of convolutional layers and activation functions like ReLU. CNNS learn

ﬁlters to convolve with the image. The convolutional layer is essentially a bank of ﬁlters and parameter

efﬁciency is created through shift-invariance. The ﬁrs CNN was LeNet(LeCun ’89), and the current

most popular CNN is ResNet(He’15).

#### 2.3.4 RNN: Recurrent Neural Network

RNNS operate on time series and updates the hidden state recurrently hence the name Recurrent

Neural Networks. There are many variants of RNNs including LSTM, GRU, and ConvGRU.

#### 2.3.5 GNN: Graph Neural Networks

GNNS operate on graphs. These graphs can be images social, k-NN graphs, molecules, scene graphs,

etc.

#### 2.3.6 Transformer

The transformer is a simple network architecture based solely on attention mechanisms. The core

concept is self-attention which can be described by \\( Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt()d_k})V \\). A

Vision Transormer is a variation that employs attention on patches. In addition DETR has applications

to object detection.

#### 2.3.7 Generative Models

The difference between Discriminative vs Generative models can be describes as p(y|x) vs p(x, y). A

Generative Adversarial Net(GAN) is a generative model where the quality of a generator net is judged

by a jointly trained discriminator net. Other options for Generative Models include normalizing ﬂow

models and diffusion models.

#### 2.3.8 The "Learning" part of ML: Optimization

The objective function of the optimization portion of the equation is to measure the error incurred by

predicting \\( y\hat{^} = f(x;\theta) \\). Some examples include mean squared error loss, cross-entropy loss, hinge

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/optimization_equation.png">
  <div class="figcaption"> 
Figure 8: Generalized equation for ML highlighting the optimization parameter.
</div>
</div>

loss, etc. In order to learn and increase optimization, we want to ﬁnd parameters \\( \theta ^ * \\) that minimize

the error over the training set. The main method of optimization is gradient descent. However to

solve the problem of large models we employ backpropagation. In addition computing the gradient

over the whole dataset is a problem for large datasets, so stochastic gradient descent(SGD) poses a

solution because the parameter update from gradient descent is of one random sample at a time. As a

result the modern optimization route includes distributed mini-batch stochastic gradient descent in

combination with adaptive learning rates.

#### 2.3.9 ML Theory

The main question of ML theory is how do we know we are learning a good model? The generalized

answer to this is that the success of our model depends on how well our model works on unseen data.

Often times in building our models there will be a bias-variance tradeoff. In addition, regularization

adds additional objective penalizing parts of the hypothesis space.


<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/ml_optimization_graph.png">
  <div class="figcaption"> 
Figure 9: Optimization graph for ML Models.
</div>
</div>


#### 2.3.10 DL Theory

Generalization in deep learning is complicated and not fully understood yet. The best generalization

we have is over-parametrized and overﬁt. The theory of deep learning generally runs behind practice.

#### 2.3.11 ML Engineering

The goal of ML engineering is to solidify the empirical practice of ML. This is key for reproducibility,

traceability, auditability, safety, and efﬁciency.


<a name='third-big-topic'></a>

<a name='subtopic-3-1'></a>
### 3.1 ML Problems in CV

Virtually all problems in CV have ML applications. CV as a high-level challenge involves semantic

understanding from pixel values of images and video. The training of ML models allows for this

understanding to be learned from experience rather than having to be coded explicitly.
	
<a name='subtopic-3-2'></a>

### 3.2 Tasks

#### 3.2.1 Classiﬁcation

Image classiﬁcation is the task of assigning a single label from a set of a categories to an image.

These categories could range from answering binary questions such as "is this a dog?", to more

complex questions such as "what digit is written here?". The example included in the following ﬁgure

shows a model which would be answering whether the image contains a building.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/classification.png">
  <div class="figcaption"> 

Figure 10: The result of a binary classiﬁcation model detecting the presence of a building.
</div>
</div>


#### 3.2.2 Segmentation

Image segmentation is the task of assigning each pixel in an image to a category label.
<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/segmentation.png">
  <div class="figcaption"> 

Figure 11: The result of a segmentation model applied to an image of a street.
</div>
</div>

#### 3.2.3 Detection

Object detection is the task of locating a speciﬁc object or objects in an image.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/detection.png">
  <div class="figcaption"> 

Figure 12: The result of a detection model identifying the location of a car.
</div>
</div>

#### 3.2.4 Tracking

Object tracking is the task of locating a speciﬁc object over time in videos.
<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/object_tracking.png">
  <div class="figcaption"> 

Figure 13: The result of a tracking model which tracks paths of cars moving through a street. Source:

https://miro.medium.com/max/1064/1\*7nRA-tBxznPfHKVv48CTYg.png
</div>
</div>

#### 3.2.5 Event Recognition

Event recognition is the task of detecting when a speciﬁc event has occurred in a video.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/event recognition.png">
  <div class="figcaption"> 

Figure 14: An image circling the group of people an event recognition model might be trying to

identify the behavior of.
</div>
</div>

<a name='subtopic-3-3'></a>
### 3.3 Applications

**Searching and Indexing Images** One application of ML in computer vision would be for searching

and indexing images. For example, ML can be used to aid organization of a personal photo library by

grouping photos with the same person or object together.

**Human-Robot Interaction** ML computer vision tasks are also utilized for the creation of robotic

systems which work for and with humans. Robots can include cameras in order to gain information

from their environment, which would necessitate CV methods to parse this camera data. Tasks such

as object detection, tracking, and event recognition have very straightforward applications in terms

of improving a robot’s ability to communicate or interact with humans. For example, Deniz et al.

published an example of a tracking based algorithm which was able to detect whether a human

nodded or shook their head in their paper Useful Computer Vision Techniques for Human-Robot

Interaction

**Sign Language** ML can also be applied in CV for recognizing sign language. This can aid in the

interpretation and and translation of sign language for those who do not understand it.

**Ambient Intelligence** ML computer vision can also be utilized in the realm of hospitals through

the form of ambient intelligence. Ambient intelligence can use object detection, tracking, and event

recognition in order to monitor or analyze the activity of patients even without doctors physically

present. This intelligent monitoring can be very valuable for things like improving clinical workﬂows

or evaluating everyday behavior of certain patients.

**Sport Analysis** Another ML application in CV is the analysis of sports activity. Applications

include determining if certain events like fouls or goals occurred, and also monitoring the motion of

athletes and their performance.

**Others** Beyond the topics explicitly listed here, ML computer vision also has a variety of other

applications. These other applications might involve combining CV with other ML ﬁelds such as

language processing in order to complete tasks such as visual question answering (VQA), which

involves parsing the text of a question and then parsing an associated image to answer the question.

ML computer vision can even be used in the realm of art with interesting applications such as neural

networks which transfer "styles" across pieces of art.

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/style_transfer_example.png">
  <div class="figcaption"> 

Figure 15: An image depicting the input and output of a style transfer model. Source:

https://colab.research.google.com/github/alzayats/Google\_Colab/blob/master/8\_3\_neural\_style\_transfer.ipynb

</div>
</div>


References

[1] Falcon A. Mendez J. Castrillon M. Deniz, O. Useful computer vision techniques for human-robot interaction.

pages 725–723, 2004.

[2] Albert Haque, Arnold Milstein, and Li Fei-Fei. Illuminating the dark spaces of healthcare with ambient

intelligence. Nature, 585(7824):193–202, 2020.
