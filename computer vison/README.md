#### Physical Reasoning Challenge

> The aim of this project is to predict the stable height of vertically stacked blocks based on a single image. The project dataset uses the ShapeStacks dataset, which contains stacked images of different shapes, structures, colors, and shooting angles, with the goal of predicting the stable height of each stack. To achieve this, we explored a variety of pre-trained models, such as ResNet50, DenseNet121, and GoogleNet, and tried to use different data augmentation techniques to simulate multiple scenarios and improve the generalization ability of the models. In addition, the Detection Transformer (DETR) model and attention mechanism were combined for further optimization.



## Getting Started

#### Prerequisites

> Before you begin, ensure you have met the following requirements:

* **Python 3.6** or higher installed 

* **Jupyter Notebook** installed

* **PyTorch** installed
```bash
pip install numpy
pip install matplotlib
pip install torch (PyTorch, for deep learning models)
pip install vscikit-learn(for machine learning tools)
pip install pandas (for data processing)
pip install opencv-python (for image processing)
pip install torch torchvision transformers matplotlib opencv-python pandas tqdm

```


## Directory Structure



- **Project_results/**: Contains best result files like `googlenet_result.json`, `resnet50_result.json`, and `predictions.csv`(submitted to kaggle).
- **saved_models/**: Used for saving the best trained models.
- **test/**: test dataset.
- **train/**: training dataset.
- **utils/**: Utility scripts that assist with model training,data preprocessing, visualization etc.
- **Centroid_Stability_Prediction.ipynb**: model 1.
- **DETR_combined_transformer.ipynb**:model 2.
- **google_net.ipynb**: model 3.
- **resnet_50_model.ipynb**: model 4(Best model).
- **train.csv**: train data (csv format).
- **test.csv**: test data (csv format).

```css
A4_project_code_group_103/
│
├── Project_results/
│   ├── googlenet_result.json
│   ├── resnet50_result.json
│   └── predictions.csv
│
├── saved_models/
│   └── ... (saved model files)
│
├── test/
│   └── ... (test dataset files)
│
├── train/
│   └── ... (training dataset files)
│
├── utils/
│   └── ...
|
├── train.csv
├── test.csv
|
├── Centroid_Stability_Prediction.ipynb
├── DETR_combined_transformer.ipynb
├── google_net.ipynb
└── README.md

```

> hint : if you want to directly use the best trained models, you can load them from the saved_models/ directory

## Running each Model



#### Ensure the Dataset is in Place:
- Place the training images outside the `train/` directory as mentioned above.
- Make sure `train.csv` is outside the `train/` directory, as it contains the necessary labels and image paths.

#### Run the Notebook:
- Open each model file in Jupyter Notebook or Jupyter Lab.
- Run all the cells step-by-step.
- The code will save the training and evaluation results to a JSON file, which can then be visualized or used for analysis later.

#### Output:
- After running the notebook, the results will be saved in `predictions.csv` and save your model to `saved_models/` directory.
- 
