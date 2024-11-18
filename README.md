# **Physical Reasoning Challenge**

This project tackles the challenging task of predicting the **stable height** of vertically stacked blocks from a single image using the **ShapeStacks dataset**. The dataset comprises images of stacked objects with diverse shapes, colors, and camera perspectives, making this a benchmark task for **visual physical reasoning**. The results have practical applications in areas like **augmented reality (AR)**, where virtual structures must align with real-world constraints, and **robotics**, where precise stability prediction is crucial for object manipulation.

---

## **Advanced Methodologies**

### **1. Multimodal Feature Extraction**
We employed multiple pretrained models, leveraging their unique capabilities to extract diverse visual features:
- **ResNet50**: With residual connections, this model extracts hierarchical features that enhance the understanding of inter-object dependencies, such as centroids and support surfaces.
- **DenseNet121**: Dense feature propagation captures fine-grained details, making it effective for detecting subtle instabilities in stack configurations.
- **GoogleNet**: Its multi-scale feature extraction via Inception modules models both local patterns (e.g., edges) and global spatial relationships critical for irregular stacks.

These architectures were fine-tuned with transfer learning, embedding task-specific priors to optimize feature extraction for the ShapeStacks dataset.

---

### **2. Context-Aware Data Augmentation**
To simulate diverse real-world scenarios, we designed a context-aware augmentation pipeline:
- **Geometric Transformations**: Random rotations and flipping emulate varying camera perspectives, ensuring invariance to orientation.
- **Occlusion and Partial Masking**: Random erasure simulates real-world scenarios where objects may be obscured, enhancing robustness under incomplete data.
- **Lighting Adjustments**: Dynamic variations in brightness and contrast model diverse environmental conditions, ensuring adaptability to AR and robotics applications.

---

### **3. Advanced Loss Function Engineering**
We designed hybrid loss functions incorporating domain knowledge:
1. **Piecewise Penalty Loss**:
    - Progressively penalizes predictions based on their deviation from stable height, ensuring focus on critical transitions.
   ```math
   L(ŷ, y, h) =
   \begin{cases} 
      L_{\text{base}}(ŷ, y), & ŷ \leq h \\
      L_{\text{base}}(ŷ, y) + λ_1(ŷ - h), & h < ŷ \leq h + δ_1 \\
      L_{\text{base}}(ŷ, y) + λ_2(ŷ - h), & h + δ_1 < ŷ \leq h + δ_2 \\
      L_{\text{base}}(ŷ, y) + λ_3(ŷ - h), & ŷ > h + δ_2
   \end{cases}
### **Adaptive Loss Function**
Dynamically weighs samples based on stack complexity, prioritizing simpler cases during early training to establish a robust baseline before tackling complex configurations.

---

### **Transformer-Based Attention Mechanisms**
To address the complexity of spatial dependencies in stacking:

- **DETR (Detection Transformer)**: Detects object bounding boxes and incorporates multi-head attention to model spatial relationships.
- **Spatial Attention Layers**: Embedded into CNN backbones to enhance focus on critical regions, such as potential tipping points or centroids.

### **5. Multistage Training Pipeline**
We implemented a progressive training strategy to ensure the model learns effectively while maintaining robust generalization:

- **Stage 1**: Fine-tuning on simpler stacking configurations to establish a foundational understanding of stability patterns and geometric reasoning.
- **Stage 2**: Introducing complex synthetic configurations generated via advanced data augmentation techniques, improving the model's ability to handle real-world scenarios.
- **Stage 3**: Applying **curriculum learning** to progressively increase the complexity of training samples, mimicking human-like learning progression and enhancing stability prediction under challenging conditions.

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
