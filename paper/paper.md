---
title: 'MemCNN: A Python/PyTorch package for creating memory-efficient invertible neural networks'
tags:
  - MemCNN
  - Python
  - PyTorch
  - machine learning
  - invertible networks
  - deep learning  
authors:
  - name: Sil C. van de Leemput
    orcid: 0000-0001-6047-3051
    affiliation: 1
  - name: Jonas Teuwen
    affiliation: 1    
  - name: Rashindra Manniesing
    affiliation: 1    
affiliations:
 - name: Radboud University Medical Center, Department of Radiology and Nuclear Medicine, Nijmegen, The Netherlands
   index: 1
date: 28 June 2019
bibliography: paper.bib
---

# Summary

Reversible operations have recently been successfully applied to classification problems to reduce memory requirements during neural network training. This feature is accomplished by removing the need to store the input activation for computing the gradients at the backward pass and instead reconstruct them on demand. However, current approaches rely on custom implementations of backpropagation, which limits applicability and extendibility. We present MemCNN, a novel PyTorch framework which simplifies the application of reversible functions by removing the need for a customized backpropagation. The framework contains a set of practical generalized tools, which can wrap common operations like convolutions and batch normalization and which take care of the memory management. We validate the presented framework by reproducing state-of-the-art experiments comparing classification accuracy and training time on Cifar-10 and Cifar-100 with the existing state-of-the-art, achieving similar classification accuracy and faster training times.

# Background

Reversible functions, which allow exact retrieval of its input from its output, can reduce memory overhead when used within the context of training neural networks with backpropagation. That is, since only the output requires to be stored, intermediate feature maps can be freed on the forward pass and recomputed from the output on the backward pass when required. Recently, reversible functions have been used with some success to extend the well established residual network (ResNet) for image classification from [@He2015] to more memory efficient invertible convolutional neural networks [@Gomez17; @Chang17; @jaco18] showing competing performance on datasets like Cifar-10, Cifar-100 [@krizhevsky2009learning] and ImageNet [@imagenet_cvpr09]. %However, practical applicability and extendibility of reversible functions for reduction of memory overhead has been limited, since current implementations require customized backpropagation, which does not work conveniently with modern deep learning frameworks and requires substantial manual design. 

The reversible residual network (RevNet) of [@Gomez17] is a variant on ResNet, which hooks into its sequential structure of residual blocks and replaces them with reversible blocks, that creates an explicit inverse for the residual blocks based on the equations from [@Dinh14] on nonlinear independent components estimation. The reversible block takes arbitrary non-linear functions $\mathcal{F}$ and $\mathcal{G}$ and renders them invertible. Their experiments show that RevNet scores similar classification performance on Cifar-10, Cifar-100, and ImageNet, with less memory overhead. 

Reversible architectures like RevNet have subsequently been studied in the framework of ordinary differential equations (ODE) [@Chang17]. Three reversible neural network based on Hamiltonian systems are proposed, which are similar to the RevNet, but have a specific choice for the non-linear functions $\mathcal{F}$ and $\mathcal{G}$ which are shown stable during training within the ODE framework on Cifar-10 and Cifar-100.

The i-RevNet architecture extends the RevNet architecture by also making the downscale operations invertible [@jaco18], effectively creating a fully invertible architecture up until the last layer, while still showing good classification accuracy compared to ResNet on ImageNet. One particularly interesting finding shows that bottlenecks are not a necessary condition for training neural networks, which shows that the study of invertible networks can lead to a better understanding of neural networks training in general.

The different reversible architectures proposed in the literature [@Gomez17; @Chang17; @jaco18] have all been modifications of the ResNet architecture and all have been implemented in TensorFlow [@TF2015]. However, these implementations rely on custom backpropagation, which limits creating novel invertible networks and application of the concepts beyond the application architecture. Our proposed framework MemCNN overcomes this issue by being compatible with the default backpropagation facilities of PyTorch. Furthermore, PyTorch offers convenient features over other deep learning frameworks like a dynamic computation graph and simple inspection of gradients during backpropagation, which facilitates inspection of invertible operations in neural networks.

# Methods

## The reversible block

The core reversible operation of MemCNN is a function $R : X \rightarrow{Y}$, which has an inverse function $R^{-1} : Y \rightarrow{X}$. Here, $x\in X$ and $y\in Y$ can be arbitrary tensors with the same size and number of dimension, i.e.: $shape(x)=shape(y)$. Additionally, it must be possible to partition the input $x=(x_1, x_2)$ and output tensors $y=(y_1, y_2)$ in half, where each partition has the same shape, i.e.: $shape(x_1)=shape(x_2)=shape(y_1)=shape(y_2)$. This formalization of the reversible block operation (equation 1), its inverse (equation 2), and its partition constraints (equation 3) provide a sufficiently general framework for implementing reversible operations. 

\label{eq:revblockforward}

\begin{equation}  \quad R(x) = y = (y_1, y_2) \end{equation}
\begin{equation} R^{-1}(y) = x = (x_1, x_2) \end{equation}
\begin{equation} shape(x_1) = shape(x_2) = shape(y_1) = shape(y_2) \end{equation}


## Couplings

Using above definitions we provide two different implementations for the reversible block in MemCNN, which we will call `couplings'. A coupling provides a reversible mapping from $(x_1, x_2)$ to $(y_1, y_2)$. MemCNN supports two couplings: the additive coupling and the affine coupling.

### Additive coupling

Equation 4 represents the additive coupling, which follows the equations of [@Dinh14] and [@Gomez17]. These support a reversible implementations through arbitrary (nonlinear) functions $\mathcal{F}$ and $\mathcal{G}$. These functions can be convolutions, ReLus, etc., as long as they have matching input and output shapes. The additive coupling is obtained by first computing $y_1$ from input partitions $x_1, x_2$ and function $\mathcal{F}$ and subsequently $y_2$ is computed from partitions $y_1, x_2$ and function $\mathcal{G}$. Next, equation 4 can be rewritten to obtain an exact inverse function as shown in equation 5. Figure 1 shows a graphical representation of the additive coupling and its inverse.

![Graphical representation of additive coupling. The left graph shows the forward computations and the right graph shows its inverse. First, input $x_1$ and $\mathcal{F}(x_2)$ are added to form $y_1$, next $x_2$ and $\mathcal{G}(y_1)$ are added to form $y_2$. Going backwards, first, $\mathcal{G}(y_1)$ is subtracted from $y_2$ to obtain $x_2$; subsequently, $\mathcal{F}(x_2)$ is subtracted from $y_1$ to obtain $x_1$. Here, $+$ and $-$ stand for respectively element-wise summation and element-wise subtraction.](additive_005.pdf)

\begin{equation}\label{eq:additiveforward}
\begin{split}
y_1 &= x_1 + \mathcal{F}(x_2) \\
y_2 &= x_2 + \mathcal{G}(y_1) \\
\end{split}
\end{equation}
\begin{equation}\label{eq:additivebackward}
\begin{split}
x_2 &= y_2 - \mathcal{G}(y_1)\\
x_1 &= y_1 - \mathcal{F}(x_2)
\end{split}
\end{equation}



### Affine coupling

Equation 6 gives the affine coupling, introduced by [@dinh2016density] and later used by [@kingma2018glow], which is more expressive than the additive coupling. The affine coupling, similar to the additive coupling, supports a reversible implementations through arbitrary (nonlinear) functions $\mathcal{F}$ and $\mathcal{G}$. It also first computes $y_1$ from input partitions $x_1, x_2$ and function $\mathcal{F}$ and subsequently it computes $y_2$ from partitions $y_1, x_2$ and function $\mathcal{G}$. The difference with the additive coupling is that now the functions $\mathcal{F}=(s,t)$ and $\mathcal{G}=(s',t')$ each produce two equally sized partitions for scaling and translation, so $shape(x_1) = shape(s) = shape(t) = shape(s') = shape(t')$ holds. These components are then used to compute the output using element-wise product ($\odot$) and element-wise exponentiation with base $e$ and element-wise addition ($+$). Equation 6 can be rewritten to obtain an exact inverse function as shown in equation 7, which uses element-wise division ($/$) and element-wise subtraction ($-$). Figure 2 shows a graphical representation of the affine coupling and its inverse. 

\break

![Graphical representation of the affine coupling. The left graph shows the forward computations and the right graph shows its inverse. Here, $\odot, /, +, -,$ and $e$ stand for element-wise multiplication, element-wise division, element-wise addition, element-wise subtraction, and element-wise exponentiation with base $e$ respectively. First, $s, t$ are computed for $\mathcal{F}(x_2)$, next input $x_1$ is element-wise multiplied with $e^{s}$ and added to $t$ to form $y_1$, subsequently $s', t'$ are computed for $\mathcal{G}(y_1)$ and then $x_2$ is element-wise multiplied with $e^{s'}$ and added to $t'$ to form $y_2$.](affine_005.pdf)


\begin{equation}\label{eq:affineforward}
\begin{split}
y_1 &= x_1 \odot e^{s} + t \;\; \textnormal{with} \;\; \mathcal{F}(x_2) = (s, t)  \\
y_2 &= x_2 \odot e^{s'} + t' \;\; \textnormal{with} \;\; \mathcal{G}(y_1) = ('s, t')
\end{split}
\end{equation}
\begin{equation}\label{eq:affinebackward}
\begin{split}
x_2 &= (y_2 - t') / e^{s'} \;\; \textnormal{with} \;\; \mathcal{G}(y_1) = (s', t') \\
x_1 &= (y_1 - t) / e^{s} \;\; \textnormal{with} \;\; \mathcal{F}(x_2) = (s, t) 
\end{split}
\end{equation}


## Implementation details

The reversible block has been implemented as a "torch.nn.Module" which wraps other PyTorch modules of arbitrary complexity for coupling functions $\mathcal{F}$ and $\mathcal{G}$. Each memory saving coupling is implemented using at least one `"torch.autograd.Function", which provides a custom forward and backward pass that works with the automatic differentiation system of PyTorch. Memory savings are implemented at the level of the reversible block and are achieved by setting the size of the underlying tensor storage to zero for inputs on the forward pass and restoring the storage size to the original size on the backward pass once it is required for computing gradients.

## Building larger networks

![Graphical representation of chaining multiple reversible block layers. ](coupling_001.pdf)

The reversible block $R$ can be chained by subsequent reversible blocks, e.g.: $R_3 \circ R_2 \circ R_1$ for reversible blocks $R_1, R_2, R_3$, which creates a fully reversible chain of operations (see Figure 3). Additionally, reversible blocks can be mixed with regular functions $f$, e.g. $f \circ R$ or $R \circ f$ for reversible block $R$ and regular function $f$. Note that mixing regular functions with reversible blocks often breaks the invertibility of a reversible chains. 

## Memory savings

**Table 1:** Comparison of memory and computational complexity for training a residual network (ResNet) between various memory saving techniques (extended table from [@Gomez17]). $L$ depicts the number of residual layers in the ResNet.

\begin{center}

\vspace{0.3cm}

\begin{tabular}{llp{2.5cm}p{2.5cm}}
\hline \textbf{Technique} & \textbf{Authors} & \textbf{Memory Complexity} & \textbf{Computational Complexity} \\ \hline
Naive & & $O(L)$ & $O(L)$ \\
Checkpointing & (Martens et al. 2012) & $O(\sqrt{L})$ & $O(L)$ \\
Recursive & (Chen et al. 2016) & $O(log\; L)$ & $O(L \;log\; L)$ \\
Additive coupling & (Gomez et al. 2017) & $O(1)$ & $O(L)$ \\
Affine coupling & (Dinh et al. 2016) & $O(1)$ & $O(L)$ \\ \hline
\end{tabular}

\vspace{0.6cm}

\end{center}

The reversible block model has an advantageous memory footprint when chained in a sequence when training neural networks. After computing each $R(x) = y$ (equation 1) on the forward pass, input $x$ can be freed from memory and be recomputed on the backward pass, using the inverse function $R^{-1}(y)=x$ (equation 2). Once the input is restored, the gradients for the weights and the inputs can be recomputed as normal using the PyTorch `autograd' solver. This effectively yields a memory complexity of $O(1)$ in the number of chained reversible blocks. Table 1 shows a comparison of memory versus computational complexity for different memory saving techniques.

# Experiments and results

 **Table 2:** Accuracy (acc.) and training time (time, in hours:minutes) comparison of the PyTorch implementation (MemCNN) versus the Tensorflow implementation from [@Gomez17] on Cifar-10 and Cifar-100 [@krizhevsky2009learning].
 
\begin{center}
 
\vspace{0.3cm}
 
\begin{tabular}{lcccccccc}
\hline         & \multicolumn{4}{c}{\textbf{Tensorflow}} & \multicolumn{4}{c}{\textbf{PyTorch}} \\
           & \multicolumn{2}{c}{\textbf{Cifar-10}} & \multicolumn{2}{c}{\textbf{Cifar-100}} & \multicolumn{2}{c}{\textbf{Cifar-10}} & \multicolumn{2}{c}{\textbf{Cifar-100}} \\
\textbf{Model} & \textbf{acc.} & \textbf{time} & \textbf{acc.} & \textbf{time} & \textbf{acc.} & \textbf{time} & \textbf{acc.} & \textbf{time} \\\hline
resnet-32  & 92.74  & \,\,2:04  & 69.10   & \,\,1:58   & 92.86  & 1:51    & 69.81   & 1:51    \\
resnet-110 & 93.99  & \,\,4:11 & 73.30   & \,\,6:44   & 93.55  & 2:51    & 72.40   & 2:39    \\
resnet-164 & 94.57  & 11:05 & 76.79   & 10:59  & 94.80  & 4:59    & 76.47   & 3:45    \\
revnet-38  & 93.14  & \,\,2:17 & 71.17   & \,\,2:20   & 92.80  & 2:09    & 69.90   & 2:16    \\
revnet-110 & 94.02  & \,\,6:59 & 74.00   & \,\,7:03    & 94.10  & 3:42    & 73.30   & 3:50    \\
revnet-164 & 94.56  & 13:09 & 76.39   & 13:12  & 94.90  & 7:21    & 76.90   & 7:17    \\ \hline
\end{tabular}

\vspace{0.6cm}

\end{center}

To validate MemCNN, we reproduced the experiments from [@Gomez17] on Cifar-10 and Cifar-100 [@krizhevsky2009learning] using their Tensorflow [@TF2015] implementation on GitHub\footnote{\url{https://github.com/renmengye/revnet-public}}, and made a direct comparison with our PyTorch implementation on accuracy and train time. We have tried to keep all the experimental settings, like data loading, loss function, train procedure, and training parameters, as similar as possible. All experiments were performed on a single NVIDIA GeForce GTX 1080 with 8GB of RAM. The results are listed in Table 2. Model performance of our PyTorch implementation obtained similar accuracy to the TensorFlow implementation with less training time on Cifar-10 and Cifar-100. All models and experiments are included in MemCNN and can be rerun for reproducibility.


 **Table 3:** GPU VRAM memory usage using the MemCNN implementation during training for all models. All models were trained on a NVIDIA GeForce GTX 1080 with 8GB of RAM. Significant memory savings were observed when using reversible operations as the number of layers increased. 

\begin{center}

\vspace{0.3cm}

\begin{tabular}{cllr}
\hline \textbf{Model} & \textbf{Layers} & \multicolumn{2}{c}{\textbf{GPU VRAM}} \\ \hline
resnet & 32  & 766  & MB \\
resnet & 110 & 1357 & MB \\
resnet & 164 & 3083 & MB \\
revnet & 38  & 677  & MB \\
revnet & 110 & 706  & MB \\
revnet & 164 & 1226 & MB \\ \hline
    \end{tabular}

\vspace{0.6cm}

\end{center}

Table 3 shows the average memory usage during model training using MemCNN. The results show significant memory savings using the invertible operations as the number of layers increase with respect to ResNet without the use of invertible operations.

# Works using MemCNN

MemCNN has recently been used to create reversible GANs for memory-efficient image-to-image translation by [@Ouderaa_2019_CVPR]. Image-to-image translation considers the problem of mapping both $X \rightarrow Y$ and $Y \rightarrow X$ given two image domains $X$ and $Y$ using either paired or unpaired examples. In this work the CycleGAN [@zhu2017unpaired] model has been enlarged and extended with an invertible core using the reversible block, which they call RevGAN. Since the invertible core is weight tied, training the model for the mapping $X \rightarrow Y$ automatically trains the model for mapping $Y \rightarrow X$. They show similar or increased performance of RevGAN with respect to similar non-invertible models like the CycleGAN with less memory overhead during training.

# Acknowledgements

This work was supported by research grants from the Netherlands Organization for Scientific Research (NWO), the Netherlands and Canon Medical Systems Corporation, Japan.

# References
