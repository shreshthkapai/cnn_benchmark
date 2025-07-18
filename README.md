# PyTorch CNN Benchmark: Custom CNN vs. ResNet18 (from scratch)

A comprehensive project to benchmark a custom-built Convolutional Neural Network (CNN) against a ResNet18 model (trained from scratch) on the CIFAR-10 dataset.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shreshth2002/CNN_Benchmark)

## 🚀 Live Demo

A live demo of this project is deployed on Hugging Face Spaces. You can upload an image and see the model's prediction in real-time.

**[Try the live demo here!](https://huggingface.co/spaces/Shreshth2002/CNN_Benchmark)**

## ✨ Features

*   **Custom CNN Architecture:** A CNN model built from scratch using PyTorch's `nn.Module`.
*   **ResNet18 Model:** Leverages a ResNet18 architecture from `torchvision.models`. **Note:** The model is trained from scratch, not used with pre-trained weights.
*   **Data Handling:** Custom `DataLoader` for the CIFAR-10 dataset with data augmentation.
*   **Comprehensive Evaluation:** Benchmarking of model size, inference speed, and accuracy.
*   **Rich Visualizations:** Learning curves (loss and accuracy), and confusion matrices to assess model performance.
*   **Experiment Tracking:** Integration with TensorBoard or Weights & Biases for logging and monitoring training metrics.
*   **Interactive UI (Optional):** A Gradio interface for easy interaction with the trained models.

## 📂 Project Structure

```
cnn-benchmark/
├── main.py                     # Main experiment pipeline (train, eval, compare)
├── data/                        # Downloaded datasets or processed subsets
├── models/
│   ├── custom_cnn.py           # Your custom CNN model
│   └── resnet18.py             # ResNet18 model architecture
├── train.py                    # Core training loop logic
├── eval.py                     # Evaluation and benchmarking functions
├── utils/
│   ├── data_loader.py          # Custom DataLoader, augmentations
│   └── visualizations.py       # Confusion matrix, learning curves
├── gradio_ui.py                # Gradio interface
├── wandb_utils.py              # WandB integration
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

*   **Framework:** [PyTorch](https://pytorch.org/)
*   **Data Handling:** [torchvision](https://pytorch.org/vision/stable/index.html)
*   **Visualization:** [matplotlib](https://matplotlib.org/), [Weights & Biases](https://wandb.ai/)
*   **UI (Optional):** [Gradio](https://www.gradio.app/)

## ⚙️ Setup and Usage

1.  **Download the CIFAR-10 Dataset:**
    Download the CIFAR-10 python version from the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) and place the extracted `cifar-10-batches-py` folder into the `data/` directory.

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd cnn-benchmark
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Experiments:**
    The `main.py` script is the primary entry point for training, evaluating, and comparing models.

    *   **To run the full comparison between the Custom CNN and ResNet18 (default):**
        ```bash
        python main.py
        ```
        This will train both models, evaluate them, save the best checkpoints, and print a comparison summary.

    *   **To run an experiment for a single model (e.g., Custom CNN):**
        ```bash
        python main.py --mode single --model custom
        ```

    *   **To run a hyperparameter sweep (requires W&B setup):**
        ```bash
        python main.py --mode sweep --model custom
        ```

5.  **Launch the Gradio UI (Optional):**
    ```bash
    python gradio_ui.py
    ```

## 📊 Results and Benchmarks

Here's a summary of the model comparison and per-class performance.

### 🏆 Model Comparison Summary

| Model        | Test Accuracy | Parameters   | Model Size | Inference Time | Throughput         |
|--------------|---------------|--------------|------------|----------------|--------------------|
| **Custom CNN** | **90.64%**    | **2,332,106**  | **8.90 MB**  | **0.04 ms**    | **27253 samples/sec** |
| ResNet18     | 84.86%        | 11,181,642   | 42.65 MB   | 0.08 ms        | 13067 samples/sec  |

**🏆 Winner: CustomCNN (90.64% vs 84.86%)**

### 📈 Per-Class Performance (ResNet18)

```
              precision    recall  f1-score   support

    airplane      0.858     0.884     0.871      1000
  automobile      0.929     0.908     0.919      1000
        bird      0.825     0.802     0.813      1000
         cat      0.685     0.716     0.700      1000
        deer      0.840     0.816     0.828      1000
         dog      0.794     0.769     0.782      1000
        frog      0.866     0.891     0.878      1000
       horse      0.895     0.874     0.884      1000
        ship      0.920     0.915     0.917      1000
       truck      0.880     0.911     0.895      1000

    accuracy                          0.849     10000
   macro avg      0.849     0.849     0.849     10000
weighted avg      0.849     0.849     0.849     10000
```

## 🎯 Future Work

*   [ ] Experiment with other datasets like FashionMNIST.
*   [ ] Implement more complex custom architectures.
*   [ ] Explore different hyperparameter tuning strategies.
*   [ ] Optimize the models for deployment (e.g., quantization, pruning).
