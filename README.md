
# Proxy-Decidability Loss (PD-Loss) for Deep Metric Learning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Choose your license -->

This repository contains the PyTorch implementation for the **Proxy-Decidability Loss (PD-Loss)**, a novel approach for deep metric learning inspired by the Decidability Index (d'). This work aims to improve upon the original [Decidability-Based Loss (D-Loss)](https://ieeexplore.ieee.org/document/9891934) by leveraging class proxies, mitigating the dependency on large mini-batches and enhancing scalability.

The core idea is to optimize the statistical separability between genuine similarities (embeddings vs. their class proxy) and impostor similarities (embeddings vs. other class proxies).

## Features

*   **PDLoss Implementation:** A robust and well-documented PyTorch module (`pd_loss.py` - *assuming you save the loss class there*).
*   **Embedding Network:** Example ResNet-50 based network for embedding extraction (`model.py` - *assuming you save the model class there*).
*   **Training Script:** A basic script (`train.py` - *assuming you save the main script there*) demonstrating how to train a model using PD-Loss on the CUB-200-2011 dataset.
*   **Clear Setup Instructions:** Comprehensive guide for environment setup and dataset preparation.

## Installation

### Prerequisites

*   Python 3.8+
*   Anaconda or Miniconda (recommended for managing environments) or `python3 -m venv`
*   CUDA-enabled GPU (recommended for reasonable training times)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/pd-loss.git # <<< Replace with your repo URL
    cd pd-loss
    ```

2.  **Create a virtual environment:**

    *   **Using Conda:**
        ```bash
        conda create -n pdloss python=3.9 # Or choose your preferred Python version
        conda activate pdloss
        ```
    *   **Using venv:**
        ```bash
        python3 -m venv venv_pdloss
        source venv_pdloss/bin/activate # On Windows use `venv_pdloss\Scripts\activate`
        ```

3.  **Install PyTorch:**
    PyTorch installation depends heavily on your operating system and CUDA version. **It is strongly recommended to install PyTorch following the official instructions:**

    Go to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

    Select your OS, package manager (`conda` or `pip`), compute platform (CUDA version or CPU), and run the generated command. For example, a common command for Linux with CUDA 11.8 using pip would be:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    *(Do not just run the command above, use the one generated for YOUR system!)*

4.  **Install other dependencies:**
    Once PyTorch is installed correctly, install the remaining packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup (CUB-200-2011)

This project is configured to use the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset.

1.  **Download the dataset:**
    You can download it from the official website:
    [http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    You will need `CUB_200_2011.tgz`.

2.  **Extract the dataset:**
    Extract the archive to a location of your choice. The structure should look something like this:
    ```
    <your_dataset_path>/
        CUB_200_2011/
            images/
                001.Black_footed_Albatross/
                002.Laysan_Albatross/
                ...
            attributes/
            parts/
            README
            bounding_boxes.txt
            classes.txt
            image_class_labels.txt
            images.txt
            ...
    ```

3.  **Configure the path:**
    You need to tell the training script where to find the dataset. Open the main training script (e.g., `train.py`) and modify the `CUB_ROOT` variable in the `if __name__ == "__main__":` block:
    ```python
    # --- Configuration ---
    # ... other configs ...
    CUB_ROOT = "/path/to/your/CUB_200_2011_folder" # <<< --- SET YOUR CUB PATH HERE
    # ...
    ```
    *(Replace `/path/to/your/CUB_200_2011_folder` with the actual path to the **parent directory** containing the `CUB_200_2011` folder).*

    *(Future Improvement: Implement command-line arguments using `argparse` to pass the dataset path instead of hardcoding it).*

## Usage

### Training

To start training the embedding model using PD-Loss on the CUB-200 dataset:

```bash
python train.py # Assuming train.py is your main script
```

You can modify hyperparameters like `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `EMBEDDING_SIZE`, etc., directly within the `if __name__ == "__main__":` block of the `train.py` script.

*(Future Improvement: Add command-line arguments for these hyperparameters).*

### Evaluation

The current `train.py` script focuses on the training loop. An evaluation script (`evaluate.py` or similar) needs to be created separately. This script should:

1.  Load a trained model checkpoint.
2.  Extract embeddings for the test set.
3.  Use a library like `pytorch-metric-learning` to compute standard metric learning benchmarks (e.g., Recall@K, NMI, mAP).

Example evaluation steps (conceptual):

```python
# In evaluate.py (conceptual)
# 1. Load model state dict
# 2. Load test dataset/loader
# 3. Extract embeddings for gallery and query sets
# 4. Use pytorch_metric_learning.utils.accuracy_calculator.AccuracyCalculator
#    or similar tools to compute metrics.
```

## Implementation Details

*   **Loss Function:** Implements the Log-Negative Decidability formulation (`-log(mu_gen - mu_imp) + 0.5 * log(var_gen + var_imp)`).
*   **Proxies:** One learnable proxy vector per class - initialized with the mean vector for each class.
*   **Normalization:** Embeddings and proxies are L2-normalized before calculating cosine similarity.
*   **Backbone:** ResNet-50, Vit, optionally pretrained on ImageNet.
*   **Stability:** Epsilon values and clamping are used in the loss calculation to prevent NaN/Inf values.

## Results

*(Placeholder: This section will be updated with tables and figures summarizing experimental results, comparing PD-Loss against baselines like D-Loss, ProxyNCA, ProxyAnchor, MS Loss, Circle Loss, etc., on datasets like CUB-200, CARS196, SOP, LFW).*

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. *(Create a LICENSE file with the MIT license text)*

## Acknowledgements & References

*   This work is inspired by the original D-Loss paper:
    > Silva, P. H., Moreira, G., Freitas, V., Silva, R., Menotti, D., & Luz, E. (2022, July). A Decidability-Based Loss Function. In *2022 International Joint Conference on Neural Networks (IJCNN)* (pp. 1-8). IEEE.
*   Leverages concepts from proxy-based metric learning methods like ProxyNCA and ProxyAnchor.


---
