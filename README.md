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
1. Open the file named inference.ipynb
2. Make sure that the environment created earlier is activated, or all dependecies are properly installed in your local environment
3. Run the cells one by one, they are commented with clear instructions. Cells without instructions should be run regardless, but only commented cells require user input.