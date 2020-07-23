# Incremental learning in image classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master)

Manuele Macchia, Francesco Montagna, Giacomo Zema

*Machine Learning and Deep Learning<br>
Politecnico di Torino<br>
A.Y. 2019/2020*

**Abstract.** Extending the knowledge of a model is an open problem in deep learning. A central issue in incremental learning is catastrophic forgetting, resulting in degradation of previous knowledge when gradually learning new information.
The scope of the described implementation is to reproduce some existing baselines that address the difficulties posed by incremental learning, to propose variations to the existing frameworks in order to gain a deeper knowledge of their components and in-depth insights, and finally to define new approaches to overcome existing limitations.

→ [Open paper](report/report.pdf)

## Usage
To run an experiment, open one of the notebooks in the root directory of the repository and execute all cells related to the desired experiment. For example, to reproduce the iCaRL baseline experiment, run the iCaRL section of [`baselines.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/baselines.ipynb). The following section illustrates the directory structure and presents the contents of the notebooks.

## Structure
### Notebooks
- [`baselines.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/baselines.ipynb) includes baseline experiments such as fine-tuning, Learning without Forgetting and iCaRL.

- [`studies_loss.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/studies_loss.ipynb) contains experiments aimed at observing the behaviour of the network when replacing classification and distillation losses with different combinations.

- [`studies_classifier.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/studies_classifier.ipynb) implements and evaluates different classifiers in place of iCaRL's standard nearest-mean-of-exemplars.

- [`distillation_targets.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/distillation_targets.ipynb) implements an analysis of distillation targets and a variation to the iCaRL framework which produces a slight performance enhancement.

- [`representation_drift.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/representation_drift.ipynb) and [`representation_drift_tsne.ipynb`](https://colab.research.google.com/github/manuelemacchia/incremental-learning-image-classification/blob/master/representation_drift_tsne.ipynb) consist in the second variation applied to the iCaRL framework, along with related visualizations.

More information regarding the baseline experiments, ablation studies and variations are available in the report.

### Directories
- `data` contains a class for handling CIFAR-100 in an incremental setting. Its main purpose is dividing the dataset in ten batches of ten classes each.

- `model` contains classes that implement the baselines, _i.e._, fine-tuning, Learning without Forgetting and iCaRL, and the ResNet-32 backbone.

- `report` contains the source code of the report and a [pre-compiled PDF version](report/report.pdf).

- `results` contains logs of (some of) the experiments that we carried out. We include a notebook that provides a framework for exploring the results. Logs are serialized and saved as pickle objects.

- `utils` contains utility functions, mainly for producing plots and heatmaps.

## References
[1] K. He, X. Zhang, S. Ren, and J. Sun. _Deep Residual Learning for Image Recognition._ IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778, 2016.  
[2] G. Hinton, O. Vinyals, and J. Dean. _Distilling the Knowledge in a Neural Network_, 2015.  
[3] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin. _Learning a Unified Classifier Incrementally via Rebalancing._ IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 831–839, 2019.  
[4] A. Krizhevsky. _Learning Multiple Layers of Features from Tiny Images._ 2009.  
[5] Z. Li and D. Hoiem. _Learning without Forgetting._ European Conference on Computer Vision (ECCV), 2016.  
[6] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert. _iCaRL: Incremental Classifier and Representation Learning._ IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 5533–5542, 2017.
