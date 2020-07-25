# cv-keypoints-detector
Computer vision project to detect faces and their 64 unique keypoints (for eyebrows, eyes. nose, mouth, face contours).
This is based on Udacity Nanodegree Computer Vision project found at: https://github.com/udacity/P1_Facial_Keypoints

# Pre-requisities
1. You have already installed `conda` and `pip` on your computer.
-- Else do refer to this [link](https://github.com/udacity/CVND_Exercises#configure-and-manage-your-environment-with-anaconda), under section **Configure and Manage Your Environment with Anaconda** to complete step 1 on **Installation**.  

2. You have some knowledge of neural nets and convolutional neural nets in particular.  

3. Some basic knowledge of `pytorch`.

# Instructions
1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/pangteckchun/cv-keypoints-detector.git  

cd cv-keypoints-detector
```

2. Create (and activate) a new environment, named `cv-kp` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-kp python=3.6
	source activate cv-kp
	```
	- __Windows__: 
	```
	conda create --name cv-kp python=3.6
	activate cv-kp
	```
	
	At this point your command line should look something like: `(cv-kp) <User>:CVND_Exercises <user>$`. The `(cv-kp)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install the required pip packages for this project to run properly. These are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

7. Ready,set, go!

 Assuming you're environment is still activated (i.e. you still see the `cp-kv` in the command prompt, you can start the notebook as follows:

```
cd cv-keypoints-detector
jupyter notebook
```

You will see a list of notebooks.
These are ones you should run in sequence to get a feel of the whole project:  
a) 1. Load and Visualize Data.ipynb  
b) 2. Define the Network Architecture.ipynb  
c) 3. Facial Keypoint Detection, Complete Pipeline.ipynb  
d) 4. Fun with Keypoints.ipynb

To exit the environment when you have completed your work session, simply close the terminal window.

# Changing the work to suit your needs  
Notebook **2. Define the Network Architecture.ipynb** & **models.py** are where the main changes should be made to tailor them to your needs:  
a. **models.py** - defining the CNN architecture or layers  

b. **2. Define the Network Architecture.ipynb** - try out different loss functions and optimisers, hyper-parameters for batch_size, learning rate, epoch_size to train and test your CNN architecture.  

There are various **2. Define the Network Architecture-XXX.ipynb** notebooks which have experimented with different loss functions, optimisers and hyper-parameters. You may look at the results of those and gain an inituiton of what else to change to suit your needs or improve the accuracy further.
