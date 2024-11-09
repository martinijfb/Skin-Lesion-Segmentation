# Skin Lesion Segmentation with UNet and Streamlit

This project implements a deep learning model for skin lesion segmentation using a UNet architecture. The model is developed in PyTorch and deployed with an interactive Streamlit application, allowing users to upload skin lesion images and view segmentation results in real-time. Evaluation metrics, including Dice Coefficient and Intersection over Union (IoU), are also displayed to assess model performance.

## Features

- **UNet Architecture**: A neural network designed for image segmentation tasks, specifically optimized for biomedical applications like skin lesion segmentation.
- **Streamlit Interface**: User-friendly web application to upload images, run segmentation, and display results interactively.
- **Evaluation Metrics**: Dice Coefficient and IoU scores are computed to evaluate segmentation quality.
- **GPU-Accelerated Training**: Model training was conducted on Google Colab for faster processing using GPU support.

## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- [Python 3.7+](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/) (see installation instructions for your specific environment)
- [Streamlit](https://streamlit.io/)
- [Git LFS](https://git-lfs.github.com/)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/martinijfb/Skin-Lesion-Segmentation.git
    cd skin-lesion-segmentation
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the model weights using Git LFS:
    ```sh
    git lfs install
    git lfs pull
    ```

### Usage

1. **Run the Streamlit Application**:
    ```sh
    streamlit run Streamlit/main.py
    ```

2. **Upload an Image**:
    - Open the Streamlit application in your browser.
    - Upload an image of a skin lesion.
    - View the segmentation results and evaluation metrics.

### Project Structure

- **DATA/**: Contains the dataset for training and testing. Note that the data is not included in the repository due to its size. You can download the dataset from the [ISIC 2016 Challenge](https://challenge.isic-archive.com/data/).
  - `TestImages/`: Images for testing the model.
  - `TestMasks/`: Ground truth masks for testing.
  - `TrainingImages/`: Images for training the model.
  - `TrainingMasks/`: Ground truth masks for training.

- **Experimenting/**: Contains Jupyter notebooks for experimenting with the model.
  - `skin_lesion.ipynb`: Notebook for training and evaluating the UNet model.

- **Models/**: Contains the pre-trained model weights.
  - `unet_skin_lesion_segmentation.pth`: Pre-trained UNet model weights.

- **Streamlit/**: Contains the Streamlit application and model definition.
  - `main.py`: Main file to run the Streamlit application.
  - `unet_module.py`: Contains the UNet model definition and helper functions.

- **README.md**: Project documentation.

### Training the Model

To train the model from scratch, follow the steps in the `Experimenting/skin_lesion.ipynb` notebook. Ensure you have the training data in the `DATA/TrainingImages` and `DATA/TrainingMasks` directories.

### Evaluation Metrics

The model's performance is evaluated using the following metrics:
- **Dice Coefficient**: Measures the overlap between the predicted and ground truth masks.
- **Intersection over Union (IoU)**: Measures the intersection over the union of the predicted and ground truth masks.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

- The UNet architecture is based on the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
- The dataset used for training and testing is from the ISIC 2016 Challenge: Gutman, David; Codella, Noel C. F.; Celebi, Emre; Helba, Brian; Marchetti, Michael; Mishra, Nabin; Halpern, Allan. "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin Imaging Collaboration (ISIC)". eprint arXiv:1605.01397. 2016.