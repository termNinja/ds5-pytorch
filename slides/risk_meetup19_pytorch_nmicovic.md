---
title: 'Introduction to Deep Learning through PyTorch'
aspectratio: 169
author: Nemanja Mićović
urlcolor: cyan
colorlinks: true
headerincludes: |
    \newcommand{\theimage}[1]{\begin{center}\includegraphics[width=1\textwidth,height=0.9\textheight,keepaspectratio]{#1}\end{center}}
    \newcommand{\en}[1]{(eng. \textit{#1})}
---

## MATF
\theimage{./images/matf-logo}

## MATF - moduli
\theimage{./images/matf-smerovi}

## `whoami`
\begin{columns}
    \begin{column}{0.6\textwidth}
        \begin{itemize}
            \item Teaching assistant and PhD student at Faculty of Mathematics
            \item ML/AI Research scientist at Nordeus
            \item I enjoy:

            \begin{itemize}
                \item Artificial intelligence
                \item Machine learning
                \item Education and teaching
                \item GNU/Linux and open source community
                \item Python
                \item Epic and science fantasy
                \item Video games!
                \item Krav Maga
            \end{itemize}
        \end{itemize}
    \end{column}
    \begin{column}{0.4\textwidth}  %%<--- here
        \begin{center}
            \includegraphics[scale=0.13]{./images/nmicovic}
        \end{center}
    \end{column}
\end{columns}


## Organization RISK (Računarstvo i informatika studentski klub)
\begin{center}
    \includegraphics[scale=0.2]{./images/risk}
\end{center}

- Organization for computer science that meets every 2-3 weeks
- Working on hot topics like:
    - video games development
    - artificial intelligence
    - android development
    - web programming
    - blockchain...
- Lectures come from both industry and academia

## R/SK> (org. team)
\theimage{./images/riskteam02}

## R/SK>
\begin{center}
    \includegraphics[scale=0.2]{./images/risk}
\end{center}

### How to reach us
- [instagram: \@riskmatf](https://www.instagram.com/riskmatf/) - follow us
- [web: risk.matf.bg.ac.rs](http://risk.matf.bg.ac.rs/)
- [github: \@riskmatf](https://github.com/riskmatf)
- [youtube](https://www.youtube.com/channel/UCdYWzvNHd1vUtVJYWJb9ggg) - Video of meetings are here

# Machine learning

## Machine learning
\begin{center}
    \includegraphics[scale=0.14]{./images/ml-robot}
\end{center}

- Field of artificial intelligence
- Amazing results in the last 10 years
- Very attractive, dynamic and active field
- Devoted to systems who can improve their performance through time
    - Systems that **learn**
- Field consists of:
    - Supervised learning
    - Unsupervised learning
    - Semi-supervised learning
    - Reinforcement learning

## Where is ML applied today?
\begin{columns}
    \begin{column}{0.5\textwidth}
        \begin{itemize}
            \item Autonomous driving
            \item Bioinformatics
            \item Social networks
            \item Algorithm portfolio
            \item Playing video games
            \item Image classification
            \item Recognizing harndwritting
            \item Natural language processing
            \item Generating optimization algorithms
            \item Generating images
        \end{itemize}
    \end{column}
    \begin{column}{0.5\textwidth}  %%<--- here
        \begin{itemize}
            \item Computer vision
            \item Detecting frauds
            \item Data mining
            \item Medical use
            \item Marketing and targeted marketing
            \item Robotics
            \item Economy 
            \item Speech recognition
            \item Speech synthesis
            \item Recommender systems
        \end{itemize}
    \end{column}
\end{columns}

## Machine learning and deep learning
\theimage{./images/field-of-ml}

## Deep learning
- Field of ML
- Focused on neural networks
- Some of most amazing results come from here
- Deep learning is a **subset** of machine learning
- Often distinction isn't made due to marketing purposes
- Even worse, **deep learning** is used as a synonym for **AI**

## AlphaGo
\begin{center}
    \includegraphics[scale=0.3]{./images/alphago}
\end{center}

- *Lee Sedol*, world champion in game of Go losses from system AlphaGo (Google) 2015.
- Google used 1920 CPUs and 280 GPUs (per some reports)
- Game of Go is very complex
- It wasn't expected that AI would be able to conquer it for some time

## AlphaZero
- Continuation of algorithm AlphaGo 2015.
- Trained to play against itself
    - Defeated AlphaGo in Go with 60:40
- It can also play games of chess and shogi
- Currently is considered the best AI systems for go, chess and shogi

## Autonomous driving
\begin{center}
    \includegraphics[scale=0.3]{./images/tesla}
\end{center}

- Tesla: [video](https://www.youtube.com/watch?v=tlThdr3O5Qo)
- Waymo: [video](https://www.youtube.com/watch?v=aaOB-ErYq6Y)
- Waymo taxi: [video](https://www.youtube.com/watch?v=WBkgs4u5tW0)

## Google QuickDraw
\begin{center}
    \includegraphics[scale=0.24]{./images/quickdraw}
\end{center}

- Sistem recognizing user drawn objects
- [live demo](https://quickdraw.withgoogle.com/)

## DeepFake ([terminator video](https://www.youtube.com/watch?v=AQvCmQFScMA))
\theimage{./images/deepfake01}

## Mitsuku ([online demo](https://www.pandorabots.com/mitsuku/))
\theimage{./images/mitsuku}

## Detecting objects
\begin{center}
    \includegraphics[scale=0.2]{./images/bond}
\end{center}

- Detecting objects - [video](https://www.youtube.com/watch?v=_zZe27JYi8Y)
- Detecting objects - [video](https://www.youtube.com/watch?v=QcCjmWwEUgg)
- James Bond - [video](https://www.youtube.com/watch?v=VOC3huqHrss&t=47s)

## Style transfer ([video](https://www.youtube.com/watch?v=VkyGphC8aCY))
\theimage{./images/style-transfer}

## Generating images (faces)
\begin{center}
    \includegraphics[scale=0.3]{./images/nordeus}
\end{center}

- Example shown by Nordeus at Machine learning seminar at MATF
- Which face is generated?

\begin{center}
    \includegraphics[scale=0.24]{./images/gan-faces}
\end{center}

## Writing poetry - real VS generated
\begin{columns}
    \begin{column}{0.5\textwidth}
    \emph{Sveti Jovan od zemlje na noge,}

    \emph{Sve pod njima konja privatiše,}

    \emph{Pod Stjepana grada bijeloga,}

    \emph{Pa podiže sirotinja rodila,}

    \emph{Pa pogubi pod svoje postajemo,}

    \emph{ne bi li me provizur-Mijkom.}

    \emph{Kad su bili na noge lagane.}
    \end{column}
    \begin{column}{0.5\textwidth}  %%<--- here
    \emph{Sveti Jovan otisnu jabuku,}

    \emph{Ona pade moru u dubine,}

    \emph{Tople su ga suze propanule,}

    \emph{No mu care riječ progovara:}

    \emph{"A ne plači, dragi pobratime!}

    \emph{"Ne moj mene ugrabit’ korunu,}

    \emph{"Ja ću tebe izvadit’ jabuku."}
    \end{column}
\end{columns}

# Neural networks

## Neural networks
- Universal function approximators
- Basic building blocks of many ML algorithms
- Not so new, some first versions shown in '60s and '70s of previous century
- Inspired with the way our brain works

\begin{figure}[h!]
    \centering
    \includegraphics[scale=0.44]{./images/deep_neural_network}
    \caption{Arhitektura neuronske mreže.}
\end{figure}

## Neuron of a neural network

### Terminology:
- Activation: $a_i$
- Neuron weight: $w_i$
- Bias: $b$
- Non linear function: $g$
- Output is calculated as:

$$
a_{out} = g(b + \sum_{i=1}^{N} a_i w_i)
$$

## Neuron of a Neural network 
\begin{figure}[h!]
    \centering
    \includegraphics[scale=1.3]{./images/neuron}
    \caption{Neuron illustration}
\end{figure}

## Activation function of a neural nework
- Very important to apply non linear transformation
    - Otherwise, function will stay linear
- Some popular activation functions:
	- ReLU: $g(x) = max(0, x)$
	- Sigmoid function: $g(x) = \frac{1}{1 + e^{-x}}$
	- Tangens hyperbolic $g(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
    - Lots of other ones, LeakyReLU, SinReLU, SeLU...
z
## Training neural networks

### Training neural networks
- Training is done with variants of (stochastic) gradient descent
- Weights are updated so that overall loss $L$ of model on data is **reduced**
- Stochasticity is introduced to improve training speed 
    - Reason is that gradient isn't calculated on **all** of data, but on a **subset**
    - This subset is called *batch*
- Term *epoch* is used to denote 1 run through the training data
- So, if number of epochs is 40, and batch size is 64 then:
    - 40 times we go through all the training data
    - 1 run consists of taking subsets of (at most) 64 data points

## Gradient descent

$$
w_{i+1} = w_i - \alpha \cdot \nabla{L}
$$

\begin{figure}[h!]
    \centering
	\includegraphics[scale=0.2]{./images/graddesc}
    \caption{Visualizing gradient descent}
\end{figure}

## Training neural networks

### Backpropagation
- Calculate the function gradient relative to the weights of neurons
- Is a basic building block for training neural networks
- GPUs allow for extreme parallelization of calculations 
    - They are one of the main reasons for explosion of the field
    - Also, a great excuse to buy a powerful GPU that can run modern video games :)

## Convolutional neural networks (CNN)
- Type of neural networks
- Leading approach for image classifications
- Devoted to signal processing in where there is space locality (images, audio, video)
- CNNs are constructing new attributes from the input
- Very extensively used in computer vision
- Complexity of constructed attributes increases with the depth of network
- Contain partial interpretability for their work

## Convolutional neural networks (CNN)
\begin{figure}[h!]
    \centering
	\includegraphics[scale=0.5]{./images/covnet-visual01}
    \caption{Visualizing learned filters.}
\end{figure}

## Convolutional neural networks (CNN)
\begin{figure}[h!]
    \centering
    \includegraphics[scale=0.22]{./images/covnet-visual02}
    \caption{Visualizing learned filters.}
\end{figure}

## Convolutional neural networks (CNN) - architecture

### Architecture is mostly a combination of following elements:
- Convolution layer
- Aggregation layer
- Fully connected layer

### In the last few years:
- Skip connections [@resnet]
- Inception module [@inception]

## Convolutional neural networks (CNN) - architecture

### Convolution layer:
- Detects a certain template in data
- For example, detect horizontal or vertical lines (lower layers)
- Or detect eyes and ears (higher layers)

## Convolutional neural networks (CNN) - architecture
### Aggregate layers (eng. pooling):
- Aggregate information from previous layers (mostly convolutional layers)
- As aggregating function mostly $max$ or $average$ is used

\begin{figure}[h!]
    \centering
	\includegraphics[scale=0.37]{./images/maxpooling}
    \caption{Aggregation with maximum function.}
\end{figure}

## Convolutional neural networks (CNN) - architecture
### Fully connected layer:
- Mostly used in the last few layers in a CNN to build a reggresor or classifier

## Convolutional neural networks (CNN) - example
\begin{figure}[h!]
    \centering
    \includegraphics[width=\textwidth]{./images/covnetarch}
\end{figure}

## Convolutional neural networks (CNN) - example
\begin{figure}[h!]
    \centering
	\includegraphics[scale=0.27]{./images/covnet01}
    \caption{CNN exmaple.}
\end{figure}

## Convolutional neural networks (CNN)
- Nice interactive example: [ovde](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)

\begin{figure}[h!]
    \centering
    \includegraphics[scale=0.4]{./images/covnet-web}
\end{figure}

## PyTorch
\theimage{./images/pytorch}

## PyTorch
- Library and framework for deep learning
- Continuation of the original *Torch* library for language *Lua*
- Primarily developed by AI team from *Facebook*
- PyTorch is *free* and *open source*
- Primary used from language `Python`, but there is also an **experimental** C++ support
    - If you really need C++, `TensorFlow` is recommended

# PyTorch: Computation graph

## Computation graph
- Neural networks represents an approximation of some function
    - Neural network is also a *complex* function
    - For example, function gets an image as input, and gives a probability of a person smiling or not
- To represent this function easier, we used a computation graph instead of a formula
- Computation graph shows **how** is *input* is transformed onto *output*
- Shows how is the data transformed **during** computations

## Computation graph

$$
f(x, y, z) = (x + y) \cdot z
$$

\begin{figure}[h!]
    \centering
	\includegraphics[scale=0.4]{./images/comp-graph}
    \caption{Computation graph of function $f$, \href{https://medium.com/tebs-lab/deep-neural-networks-as-computational-graphs-867fcaa56c9}{source}}
\end{figure}

## Computation graph
- Using library as `PyTorch` we can define a computation graph
- Libraries are allowing us to perform these computations on GPUs
- This is very useful as this can reduce our work and reduce calculation time
- Lots of operations in NNs are matrix operations

# PyTorch: computation graph (code!)

# PyTorch: CNN (code!)

# PyTorch: transfer learning

## Predator vs Alien
\theimage{./images/alien-vs-predator}

## Predator vs Alien
\theimage{./images/alien-predator}

## Predator vs Alien
- We wish to perform binary classification
- But, we only have 447 images per class
- CNNs often require big amount of data to train properly

## Transfer learning
- Includes using and training an already *trained* model on some other data 
- Idea is that learned filters in one problem are useful in some other *similar* problem
- There is not a *formal* algorithm for transfer learning, but some variants are:
    - Freeze convolutional layers, remove all fully connected, and put new fully connected layers
    - TODO
    - Koristiti konvolutivne slojeve neke mreže za dobijanje nove reprezentacije podataka, dalje raditi sa raznim drugim modelima
    - Zamrznuti deo konvolutivnih slojeva, obučiti ostatak
    - I slično...

## Transfer učenja
\begin{figure}[h!]
    \centering
	\includegraphics[scale=0.3]{./images/transfer_learning}
    \caption{Transfer učenja}
\end{figure}

# Transfer učenja (code!)

# Questions?

## Literatura
