# What is this?
Image captioning systems take a single image as input and automatically generates text that describes the scene of the image. This problem has amassed over a decade of active research and is starting to branch out to more interesting directions. For this specific project, the guiding principle is to design a system that allows users to shift the focus of the generated caption, without losing information about the surroundings. A user provides an image, and specifies a bounding box, and the system will generate a caption that describes the entities inside the bounding box first (the subject) then the surroundings (the context).


# Prerequisites
- Python version 3.8 or above
- Jupyter Notebook
  
It is suggested to first create a new virtual environment before install required dependencies. This can be done using conda (installation required if not already)/

To create an environment:
```
conda create --name myenv python=3.8
```
Then, activate the environment
```
conda activate myenv
```
And install the required dependencies
```
pip install -r path/to/requirements.txt
```

# Using the notebook
There are a few steps to be taken before the notebook can be used.
## 1. downloading the trained models
Go to the following link and download all the models contained:
https://drive.google.com/drive/folders/1ALqmNK-Sv06uH7Lz8A4YQ7XEW38-J9Qj?usp=sharing

Here is a list of models, for reference purpose only
- coco_newer_05.pt
- fine_tune_latest.pt
- resnet50-0676ba61.pth
- TranSalNet_Res.pth

After downloading, all the models must be moved to the directory *pretrained_models*

# Using the application
1. Open the notebook titled inference.ipynb
2. Make sure that the environment created earlier is activated, or all dependecies are properly installed in your local environment
3. Run the cells one by one, they are commented with clear instructions. Cells without instructions should be run regardless, but only certain commented cells require user input.
