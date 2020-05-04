# Maximin Affinity Learning of Image Segmentation (MALIS)

The goal of this project is to implement MALIS as desribed first in:
> [*Maximin affinity learning of image segmentation*, Srinivas C. Turaga et al. Advances in Neural Information Processing Systems, 2709](http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation)

The method was improved afterwards in :
> [ *Large Scale Image Segmentation with Structured Loss Based Deep Learning for Connectome Reconstruction*, J. Funke et al., IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, pp. 1669–1680, July 2019.](https://ieeexplore.ieee.org/document/8364622)

We will implement both methods and will provide commented Jupyter Notebooks to both explain in detail every step of the process and also to ease the reproduction.


Here is a list of present notebooks and what you will find in them:

|Notebook name| Content |
|---|---|
| *MALIS_training* | MALIS implementation and training |
| *Inference* | Using a trained model to obtain a segmentation |
| *Evaluation* | Evaluation of a model on the CREMI dataset |

On this repository you will also find a document called *Report.pdf* which contains an explanation of the methods, of our implementation, and most importantly of of results.

The slides used for our oral presentation are also available as *presentation.pdf*.

## Technology used

For this project, we used the following libraries :
 - PyTorch for the deep learning aspect
 - Higra for the parts requiring the use of graphs
 - [github/jvanvugt/pytorch-unet](https://github.com/jvanvugt/pytorch-unet) for our unet architecture
 - [github/cremi/cremi_python](https://github.com/cremi/cremi_python) for the evaluation, we adapted the source code to python 3

## Image results

We applied the method on the [CREMI dataset](https://cremi.org), which is composed of drosophilia brain images.
We display cell borders and not the labels to ease the visualization


We got the following results when using a Unet with the constrained MALIS loss.

|Image|Raw image|Groundtruth|Our results| 
|---|---|---|---|
|Image 1|<img src="https://imgur.com/h4JB8dq.png" width="270" >|<img src="https://i.imgur.com/XdL5fWh.png" width="270" >|<img src="https://i.imgur.com/tswCPUG.png" width="270" >|
|Image 2|<img src="https://i.imgur.com/LH86jJu.png" width="270" >|<img src="https://i.imgur.com/tQWoAO5.png" width="270" >|<img src="https://i.imgur.com/afO8UEH.png" width="270" >|
|Image 3|<img src="https://i.imgur.com/wOq1hRK.png" width="270" >|<img src="https://i.imgur.com/UCm4lui.png" width="270" >|<img src="https://i.imgur.com/4HQ3dwP.png" width="270" >|
|Image 4|<img src="https://i.imgur.com/rPymVdS.png" width="270" >|<img src="https://i.imgur.com/ZRzg0rx.png" width="270" >|<img src="https://i.imgur.com/zuWdFyJ.png" width="270" >|

## Team

This project was done throughout the year with the following team :
- Quentin GARRIDO (Team Leader)
- Tiphanie LAMY VERDIN 
- Josselin LEFÈVRE 
- Annie LIM
- Raphaël LAPERTOT (only for the second semester)

We were supervised by Laurent NAJMAN, who was of great help to us.
