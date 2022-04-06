# BD-BNN

# 
 - Bimodal Distributed Binarized Neural Networks



# BD-BNN: Bimodal Distributed Binarized Neural Networks
This repository is a pytorch implementation of our paper [BD-BNN](https://arxiv.org/abs/2204.02004)

<p align="center">
  <img src="https://github.com/BlueAnon/BD-BNN/blob/master/image-bdbnn.png" width="700" >
</p>



## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* *To install* and develop locally:
bash
git clone https://github.com/BlueAnon/BD-BNN.git
cd BD-BNN
pip install -r requirements.txt


## Getting Started

#for cifar10:
python RUN_COMMAND cifar10
#for imagenet
python RUN_COMMAND imagenet


## TODO:
1. Upload trained models
2. Create requirements.txt file
3. Add Runnign commands (and loading trained BNN command)
4. Get accepted to ECCV :)


## License
released under the [MIT license](LICENSE).
