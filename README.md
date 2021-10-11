## README ##

## Requirements
**This implementation is only tested under Ubuntu environment with Nvidia GPUs and CUDA installed.**

## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/brendonrandall/3d_face_reconstruction.git --recursive
cd 3d_face_reconstruction
conda env create -f environment.yml
source activate 3d_face_reconstruction
```

2. Install Nvdiffrast library:
```
cd nvdiffrast    # ./3d_face_reconstruction/nvdiffrast
pip install .
```

3. Install Arcface Pytorch:
```
cd ..    # ./3d_face_reconstruction
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch/ ./models/
```

4. Install CMAKE and DLIB
```
conda install -c conda-forge cmake
conda install -c conda-forge dlib
conda install -c conda-forge imutils
```

## Inference with a pre-trained model

### Prepare prerequisite models
1. Our method uses [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) to represent 3d faces. Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download "01_MorphableModel.mat". In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:
```
3d_face_reconstruction
│
└─── BFM
    │
    └─── 01_MorphableModel.mat
    │
    └─── Exp_Pca.bin
    |
    └─── ...
```
2. We provide a model trained on a combination of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 
[LFW](http://vis-www.cs.umass.edu/lfw/), [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm),
[IJB-A](https://www.nist.gov/programs-projects/face-challenges), [LS3D-W](https://www.adrianbulat.com/face-alignment), and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. Download the pre-trained model using this [link (google drive)](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP?usp=sharing) and organize the directory into the following structure:
```
3d_face_reconstruction
│
└─── checkpoints
    │
    └─── <model_name>
        │
        └─── epoch_20.pth

```

### Test with custom images
To reconstruct 3d faces from test images, organize the test image folder as follows:
```
3d_face_reconstruction
│
└─── <folder_to_test_images>
    │
    └─── *.jpg/*.png
    |
    └─── detections
        |
	└─── *.txt
```
The \*.jpg/\*.png files are test images. The \*.txt files are detected 5 facial landmarks with a shape of 5x2, and have the same name as the corresponding images. Check [./datasets/examples](datasets/examples) for a reference.

Then, run the test script:
```
# get reconstruction results of your custom images
python test.py --name=<model_name> --epoch=20 --img_folder=<folder_to_test_images>

# get reconstruction results of example images
python test.py --name=<model_name> --epoch=20 --img_folder=./datasets/examples
```

Results will be saved into ./checkpoints/<model_name>/results/<folder_to_test_images>, which contain the following files:
| \*.png | A combination of cropped input image, reconstructed image, and visualization of projected landmarks.
|:----|:-----------|
| \*.obj | Reconstructed 3d face mesh with predicted color (texture+illumination) in the world coordinate space. Best viewed in Meshlab. |
| \*.mat | Predicted 257-dimensional coefficients and 68 projected 2d facial landmarks. Best viewd in Matlab.

## Training a model from scratch
### Prepare prerequisite models
1. We rely on [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) to extract identity features for loss computation. Download the pre-trained model from Arcface using this [link](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#ms1mv3). By default, we use the resnet50 backbone ([ms1mv3_arcface_r50_fp16](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC)), organize the download files into the following structure:
```
3d_face_reconstruction
│
└─── checkpoints
    │
    └─── recog_model
        │
        └─── ms1mv3_arcface_r50_fp16
	    |
	    └─── backbone.pth
```
2. We initialize R-Net using the weights trained on [ImageNet](https://image-net.org/). Download the weights provided by PyTorch using this [link](https://download.pytorch.org/models/resnet50-0676ba61.pth), and organize the file as the following structure:
```
3d_face_reconstruction
│
└─── checkpoints
    │
    └─── init_model
        │
        └─── resnet50-0676ba61.pth
```
3. We provide a landmark detector (tensorflow model) to extract 68 facial landmarks for loss computation. The detector is trained on [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), [LFW](http://vis-www.cs.umass.edu/lfw/), and [LS3D-W](https://www.adrianbulat.com/face-alignment) datasets. Download the trained model using this [link (google drive)](https://drive.google.com/file/d/1Jl1yy2v7lIJLTRVIpgg2wvxYITI8Dkmw/view?usp=sharing) and organize the file as follows:
```
3d_face_reconstruction
│
└─── checkpoints
    │
    └─── lm_model
        │
        └─── 68lm_detector.pb
```
### Data preparation
1. To train a model with custom images，5 facial landmarks of each image are needed in advance for an image pre-alignment process. We recommend using [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn) to detect these landmarks. Then, organize all files into the following structure:
```
3d_face_reconstruction
│
└─── datasets
    │
    └─── <folder_to_training_images>
        │
        └─── *.png/*.jpg
	|
	└─── detections
            |
	    └─── *.txt
```
The \*.txt files contain 5 facial landmarks with a shape of 5x2, and should have the same name with their corresponding images.

2. Generate 68 landmarks and skin attention mask for images using the following script:
```
# preprocess training images
python data_preparation.py --img_folder <folder_to_training_images>

# alternatively, you can preprocess multiple image folders simultaneously
python data_preparation.py --img_folder <folder_to_training_images1> <folder_to_training_images2> <folder_to_training_images3>

# preprocess validation images
python data_preparation.py --img_folder <folder_to_validation_images> --mode=val
```
The script will generate files of landmarks and skin masks, and save them into ./datasets/<folder_to_training_images>. In addition, it also generates a file containing the path of all training data into ./datalist which will then be used in the training script.

### Train the face reconstruction network
Run the following script to train a face reconstruction model using the pre-processed data:
```
# train with single GPU
python train.py --name=<custom_experiment_name> --gpu_ids=0

# train with multiple GPUs
python train.py --name=<custom_experiment_name> --gpu_ids=0,1

# train with other custom settings
python train.py --name=<custom_experiment_name> --gpu_ids=0 --batch_size=32 --n_epochs=20
```
Training logs and model parameters will be saved into ./checkpoints/<custom_experiment_name>. 

By default, the script uses a batchsize of 32 and will train the model with 20 epochs. For reference, the pre-trained model in this repo is trained with the default setting on a image collection of 300k images. A single iteration takes 0.8~0.9s on a single Tesla M40 GPU. The total training process takes around two days.

To use a trained model, see [Inference](https://github.com/sicxu/3d_face_reconstruction#inference-with-a-pre-trained-model) section.
## Contact
If you have any questions, please contact the paper authors.

## Citation

Please cite the following paper if this model helps your research:

	@inproceedings{deng2019accurate,
	    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
	    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
	    year={2019}
	}
##
The face images on this page are from the public [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset released by MMLab, CUHK.

Part of the code in this implementation takes [CUT](https://github.com/taesungp/contrastive-unpaired-translation) as a reference.

