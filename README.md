# gf2-gf3
this is a projector for gf2/gf3 contest

Models
(Deeplab V3+) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [Paper]
(GCN) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [Paper]
(UperNet) Unified Perceptual Parsing for Scene Understanding [Paper]
(DUC, HDC) Understanding Convolution for Semantic Segmentation [Paper]
(PSPNet) Pyramid Scene Parsing Network [Paper]
(ENet) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [Paper]
(U-Net) Convolutional Networks for Biomedical Image Segmentation (2015): [Paper]
(SegNet) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [Paper]
(FCN) Fully Convolutional Networks for Semantic Segmentation (2015): [Paper]

Losses
In addition to the Cross-Entorpy loss, there is also 
Dice-Loss, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize.
CE Dice loss, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.
Focal Loss, an alternative version of the CE, used to avoid class imbalance where the confident predictions are scaled down.
Lovasz Softmax lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses (for more details, check the paper: The Lovász-Softmax loss).
Learning rate schedulers
Poly learning rate, where the learning rate is scaled down linearly from the starting value down to zero during training. Considered as the go to scheduler for semantic segmentaion (see Figure below).
One Cycle learning rate, for a learning rate LR, we start from LR / 10 up to LR for 30% of the training time, and we scale down to LR / 25 for remaining time, the scaling is done in a cos annealing fashion (see Figure bellow), the momentum is also modified but in the opposite manner starting from 0.95 down to 0.85 and up to 0.95, for more detail see the paper: Super-Convergence.


Data augmentation
All of the data augmentations are implemented using OpenCV in \base\base_dataset.py, which are: rotation (between -10 and 10 degrees), random croping between 0.5 and 2 of the selected crop_size, random h-flip and blurring

Training
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:
#训练
python train.py --config config.json
The training will automatically be run on the GPUs (if more that one is detected and multipple GPUs were selected in the config file, torch.nn.DataParalled is used for multi-gpu training), if not the CPU is used. The log files will be saved in saved\runs and the .pth chekpoints in saved\, to monitor the training using tensorboard, please run:
#查看训练指标
tensorboard --logdir saved
#inference
python inference.py input_path output_path
