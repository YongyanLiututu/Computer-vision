# **Physical Reasoning Challenge**

This project tackles the challenging task of predicting the **stable height** of vertically stacked blocks from a single image using the **ShapeStacks dataset**. The dataset comprises images of stacked objects with diverse shapes, colors, and camera perspectives, making this a benchmark task for **visual physical reasoning**. The results have practical applications in areas like **augmented reality (AR)**, where virtual structures must align with real-world constraints, and **robotics**, where precise stability prediction is crucial for object manipulation.

---

## **Advanced Methodologies**

### **1. Multimodal Feature Extraction**
We employed a combination of cutting-edge pretrained models and task-specific feature engineering to extract comprehensive visual and physical reasoning features:
- **ResNet50**: The deep residual network structure enables hierarchical feature extraction, capturing both low-level (edges, textures) and high-level (spatial relationships, centroids) features. This ensures precise modeling of dependencies critical for stability analysis, such as support surfaces and weight distribution.
- **DenseNet121**: Dense connectivity facilitates efficient gradient propagation and reuses features from earlier layers, making it especially effective for detecting small perturbations in block alignments and analyzing fine-grained inter-block interactions.
- **GoogleNet with Inception Modules**: By leveraging multi-scale convolutional filters, GoogleNet excels at capturing local patterns like block edges and centroids while also understanding global spatial structures, enabling the modeling of complex, irregular stacking geometries.

In addition, the pretrained models were fine-tuned using **transfer learning** to incorporate domain-specific priors, ensuring the network adapts to the nuances of the ShapeStacks dataset. To mitigate overfitting, we employed **layer freezing** during the early stages and gradually unlocked layers as the task complexity increased.

---

### **2. Context-Aware Data Augmentation**
To generalize the model across diverse stacking configurations, a **physics-informed data augmentation strategy** was developed, emphasizing both visual variability and physical realism:
- **Geometric Transformations**:
  - Random rotations (±45°), flips, and scalings simulate varied camera perspectives and object orientations.
  - Multi-scale cropping forces the model to infer stability from partial or off-centered stacks, mimicking real-world scenarios.
- **Occlusion and Partial Masking**:
  - Random erasure and Gaussian noise simulate object occlusions and background interference, improving robustness in cluttered environments.
- **Lighting Adjustments**:
  - Brightness, contrast, and hue jittering emulate changes in ambient lighting, enabling adaptability across diverse visual conditions.
- **Stack Manipulations**:
  - Synthetic modifications such as simulated toppling, re-stacking, or partial removal of blocks were introduced to force the model to predict stability under non-ideal configurations.

This augmentation pipeline expanded the dataset’s effective diversity, ensuring robustness against unseen conditions and enabling reliable performance across domains like AR and robotics.

---

### **3. Advanced Loss Function Engineering**
#### **Piecewise Penalty Loss**
Designed specifically for stability prediction, this loss function penalizes deviations relative to stability-critical thresholds:
1. Mild penalties for near-correct predictions, encouraging fine-tuned adjustments.
2. Steeper penalties for significant errors to guide the model away from high-magnitude deviations.

$$
L(ŷ, y, h) =
\begin{cases} 
   L_{\text{base}}(ŷ, y), & ŷ \leq h \\
   L_{\text{base}}(ŷ, y) + λ_1(ŷ - h), & h < ŷ \leq h + δ_1 \\
   L_{\text{base}}(ŷ, y) + λ_2(ŷ - h), & h + δ_1 < ŷ \leq h + δ_2 \\
   L_{\text{base}}(ŷ, y) + λ_3(ŷ - h), & ŷ > h + δ_2
\end{cases}
$$


#### **Adaptive Loss Function**
To address the imbalance between simple and complex stacking scenarios, an adaptive weighting mechanism was introduced:
- Samples are dynamically weighted based on **stack attributes** (e.g., shape complexity, centroid deviations).
- During early training, simpler samples are prioritized to establish foundational stability reasoning.
- As training progresses, the model shifts focus to more challenging configurations, ensuring balanced learning.

$$
L_{\text{adaptive}}(ŷ, y) = w \cdot L_{\text{base}}(ŷ, y)
$$
where \( w \) is a function of stack complexity, defined as:
$$
w =
\begin{cases} 
   1.0, & \text{simple stacks (low variance)} \\
   0.5, & \text{moderate complexity stacks} \\
   0.25, & \text{highly unstable or irregular stacks}
\end{cases}
$$

---

### **4. Transformer-Based Attention Mechanisms**
To enhance the model’s ability to reason about spatial relationships and dependencies:
- **Detection Transformer (DETR)**:
  - Utilizes multi-head self-attention to model global relationships between blocks.
  - Outputs bounding boxes and class probabilities, providing high-level geometric context for stability analysis.
- **Spatial Attention Layers**:
  - Added to CNN backbones to prioritize critical regions, such as tipping points or unstable centroids.
  - Dynamically refines feature maps to ensure focus on regions with the greatest impact on stability.

---

### **5. Multistage Training Pipeline**
A **three-stage curriculum learning strategy** was implemented to gradually enhance the model’s reasoning capabilities:
1. **Stage 1**: Training on simplified stacks with minimal variation to establish a baseline understanding of stability principles, such as centroids and support surfaces.
2. **Stage 2**: Introducing augmented and synthetic scenarios, including toppling and occlusion effects, to enhance robustness under diverse configurations.
3. **Stage 3**: Fine-tuning on high-complexity stacks, leveraging adaptive loss weighting to address edge cases and challenging scenarios.

Each stage incorporated dynamic hyperparameter tuning, including learning rate decay, dropout regularization, and data sampling adjustments to ensure optimal convergence.

---

### **6. Cross-Domain Generalization**
To validate the model’s practical applicability:
- Conducted domain adaptation experiments, transferring the model to new datasets with unseen stacking configurations.
- Evaluated generalization to real-world tasks, such as robotic manipulation and AR-based virtual object alignment.

This comprehensive pipeline ensures that the model is not only accurate but also adaptable across diverse use cases in **visual physical reasoning**.

---

## **Technical Specifications**

### **Prerequisites**
To replicate and execute the project workflows, ensure the following dependencies are installed in your Python environment:

```bash
pip install numpy matplotlib torch torchvision scikit-learn pandas opencv-python tqdm
```
## **Directory Structure**
The project is organized as follows:
```bash
Physical_Reasoning_Challenge/
├── Project_results/        # Model results and predictions
├── saved_models/           # Trained models
├── test/                   # Test dataset
├── train/                  # Training dataset
├── utils/                  # Utility scripts
├── Centroid_Stability_Prediction.ipynb  # Methodology 1
├── DETR_combined_transformer.ipynb     # Methodology 2
├── google_net.ipynb                    # Methodology 3
├── resnet_50_model.ipynb               # Best model
├── train.csv              # Training data metadata
├── test.csv               # Test data metadata
└── README.md              # Documentation
```
## **Experimental Results**

### **Dataset**
- **Training Set**: 7,680 images (generated via 3D rendering for diversity).
- **Test Set**: 1,920 images (evaluated on Kaggle).

### **Performance Metrics**
- **Exact Match Accuracy**: Percentage of predictions matching ground truth.
- **Top-2 Accuracy**: Predictions within ±1 block considered correct.

### **Key Results**
The **ResNet50 model** emerged as the most effective architecture:
- **Exact Match Accuracy**: 80% on the validation set.
- **Generalization**: Demonstrated robust handling of unseen configurations.

---

## **Insights and Contributions**

### **Theoretical Advancements**
1. Integrated spatial reasoning via transformers and attention mechanisms for enhanced physical stability understanding.
2. Highlighted limitations of centroid-based approaches under occlusions and irregular configurations.

### **Practical Implications**
1. Ensured generalization across AR and robotics applications through tailored augmentation and adaptive learning.

### **Future Directions**
1. **3D Geometry Integration**: To enhance depth reasoning for stability prediction.
2. **Physics Simulations**: Embedding real-world properties such as friction and weight distribution into the training process.

---

This project advances the field of **visual physical reasoning**, bridging the gap between vision-based analysis and practical stability predictions for real-world applications.
