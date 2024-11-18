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
