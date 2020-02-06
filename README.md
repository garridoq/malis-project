# Maximin Affinity Learning of Image Segmentation (MALIS)

The goal of this project is to implement MALIS as desribed first in:
> [*Maximin affinity learning of image segmentation*, Srinivas C. Turaga et al. Advances in Neural Information Processing Systems, 2009](http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation)

Them method was improved afterwards in :
> [ *Large Scale Image Segmentation with Structured Loss Based Deep Learning for Connectome Reconstruction*, J. Funke et al., IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, pp. 1669–1680, July 2019.](https://ieeexplore.ieee.org/document/8364622)

We will implement both methods and will provide commented Jupyter Notebooks to both explain in detail every step of the process and also to ease the reproduction.


Here is a list of present notebooks and what you will find in them:
|Notebook name| Content |
|---|---|
| *MALIS_training* | MALIS implementation and training |
| *Inference* | Using a trained model to obtain a segmentation 
| *Evaluation* | Evaluation of a model on the CREMI dataset |

On this repository you will also find a document called *Report.pdf* which contains an explanation of the methods, of our implementation, and most importantly of of results.

## Technology used

For this project, we used the following libraries :
 - PyTorch for the deep learning aspect
 - Higra for the parts requiring the use of graphs

