# LatentForge: Probabilistic ML and Deep Generative Modeling

This repository serves as a technical deep-dive into the foundations of generative modeling and probabilistic machine learning. It covers the end-to-end implementation of Variational Autoencoders (VAEs) and an exploration of the mathematical theory driving modern machine learning loss functions and metric optimization.

---

## Project Structure

### Notebooks
*   **`VAE.ipynb`**: Implementation of a Variational Autoencoder (VAE) from scratch using PyTorch.
    *   Features: Custom Encoder/Decoder architecture, Reparameterization Trick, KL-Divergence regularization, and latent space 2D manifold visualization using the MNIST dataset.
*   **`BCE.ipynb`**: Theoretical exploration of probabilistic ML.
    *   Features: Visualizing Mahalanobis vs. Euclidean distance, analysis of the Binary Cross-Entropy (BCE) loss landscape, and probability distribution sampling (transformation of variables).
*   **`Diffusion.ipynb`**: Data pipeline and environment setup.
    *   Features: Demonstrates automated dataset acquisition (Stanford Cars dataset) via the Kaggle API and necessary environment configuration for deep learning workflows.

---

## Concepts & Techniques Implemented

*   **Generative Modeling**: Architecting models capable of learning latent probability distributions to generate new data samples.
*   **Reparameterization Trick**: Enabling backpropagation through stochastic nodes by decoupling the sampling process.
*   **Metric Learning**: Utilizing covariance-aware metrics (Mahalanobis) to improve clustering in non-isotropic data spaces.
*   **Information Theory**: Deep-dive analysis into Entropy, Cross-Entropy, and KL Divergence as loss optimization targets.
*   **Deep Learning Pipelines**: Practical experience with PyTorch, Tensor operations, and automated data processing.

---

## Setup & Execution

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/YourUsername/LatentForge-Probabilistic-ML-and-Deep-Generative-Modeling.git](https://github.com/YourUsername/LatentForge-Probabilistic-ML-and-Deep-Generative-Modeling.git)
    cd LatentForge...
    ```
2.  **Environment Setup**:
    Ensure you have `jupyter` and `torch` installed:
    ```bash
    pip install torch torchvision matplotlib scipy scikit-learn jupyter
    ```
3.  **Run the Notebooks**:
    Launch the notebooks to view the training process and visualizations:
    ```bash
    jupyter notebook
    ```

> **Note:** The `Diffusion.ipynb` requires a Kaggle API key configured to download the dataset. Ensure the `kaggle.json` is placed in `~/.kaggle/` as per the Kaggle documentation.

---

## 👤 Author
**Arman Kumar Singh**  
IIT Kanpur | 240183
