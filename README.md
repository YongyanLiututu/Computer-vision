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

# 中文版
# **物理推理挑战**

本项目旨在通过单张图像预测垂直堆叠方块的**稳定高度**，使用的是**ShapeStacks数据集**。该数据集包含形状、颜色、拍摄角度各异的堆叠对象图像，是**视觉物理推理**领域的重要基准任务。本研究的结果在**增强现实（AR）**和**机器人技术**领域具有实际应用价值。例如，在增强现实中，虚拟结构需要与现实环境约束保持一致；在机器人技术中，物体操控需要精准的稳定性预测。

---

## **高级方法**

### **1. 多模态特征提取**
我们结合了最先进的预训练模型和特定任务的特征工程，提取全面的视觉和物理推理特征：
- **ResNet50**：深度残差网络结构可以实现分层特征提取，捕获低级（边缘、纹理）和高级（空间关系、质心）特征。这确保了对稳定性分析中关键依赖关系（如支撑面和重量分布）的精确建模。
- **DenseNet121**：通过层间密集连接实现高效的梯度传播，同时重用早期层的特征，尤其适合检测方块排列中的小扰动以及分析细粒度的块间相互作用。
- **GoogleNet with Inception Modules**：利用多尺度卷积滤波器，GoogleNet在捕获局部模式（如方块边缘和质心）的同时，也能理解全局空间结构，从而有效建模复杂、不规则的堆叠几何结构。

此外，通过**迁移学习**对预训练模型进行微调，嵌入领域特定的先验知识，以确保网络适应ShapeStacks数据集的细微特性。为了减轻过拟合，在早期阶段使用**层冻结**，并随着任务复杂性的增加逐步解锁层。

---

### **2. 面向场景的上下文数据增强**
为了使模型能够泛化到各种堆叠配置，我们开发了一个**基于物理启发的数据增强策略**，同时强调视觉多样性和物理现实性：
- **几何变换**：
  - 随机旋转（±45°）、翻转和缩放模拟不同的相机视角和物体方向。
  - 多尺度裁剪使模型能够从部分或非中心堆叠中推断稳定性，模拟现实场景。
- **遮挡和部分掩蔽**：
  - 随机擦除和高斯噪声模拟物体遮挡和背景干扰，增强在复杂环境中的鲁棒性。
- **光照调整**：
  - 亮度、对比度和色调的抖动模拟环境光变化，确保适应性强的视觉条件。
- **堆叠操控**：
  - 引入合成的堆叠修改，例如模拟倒塌、重新堆叠或部分移除块，迫使模型在非理想配置下预测稳定性。

这一增强策略扩展了数据集的有效多样性，确保了在未知条件下的鲁棒性，从而为增强现实和机器人领域提供可靠的性能。

---

### **3. 高级损失函数设计**
#### **分段惩罚损失**
专为稳定性预测设计，此损失函数根据稳定性关键阈值对偏差进行惩罚：
1. 对接近正确的预测施加轻微惩罚，鼓励精细调整。
2. 对显著错误施加较大惩罚，引导模型远离高幅度偏差。

$$
L(ŷ, y, h) =
\begin{cases} 
   L_{\text{base}}(ŷ, y), & ŷ \leq h \\
   L_{\text{base}}(ŷ, y) + λ_1(ŷ - h), & h < ŷ \leq h + δ_1 \\
   L_{\text{base}}(ŷ, y) + λ_2(ŷ - h), & h + δ_1 < ŷ \leq h + δ_2 \\
   L_{\text{base}}(ŷ, y) + λ_3(ŷ - h), & ŷ > h + δ_2
\end{cases}
$$

#### **自适应损失函数**
为了解决简单和复杂堆叠场景之间的不平衡问题，我们引入了一种自适应加权机制：
- 根据**堆叠属性**（例如形状复杂性、质心偏差）动态调整样本权重。
- 在早期训练中优先处理简单样本，以建立基础的稳定性推理。
- 随着训练的进行，模型逐渐将注意力转移到更具挑战性的配置，确保平衡学习。

$$
L_{\text{adaptive}}(ŷ, y) = w \cdot L_{\text{base}}(ŷ, y)
$$
其中 \( w \) 是堆叠复杂度的函数，定义如下：
$$
w =
\begin{cases} 
   1.0, & \text{简单堆叠（低方差）} \\
   0.5, & \text{中等复杂度堆叠} \\
   0.25, & \text{高度不稳定或不规则堆叠}
\end{cases}
$$

---

### **4. 基于Transformer的注意力机制**
为增强模型对空间关系和依赖关系的推理能力：
- **DETR（Detection Transformer）**：
  - 使用多头自注意力建模块间的全局关系。
  - 输出边界框和类别概率，为稳定性分析提供高级几何上下文。
- **空间注意力层**：
  - 嵌入CNN骨干网络中，优先处理关键区域，如倾覆点或不稳定质心。
  - 动态优化特征图，确保关注对稳定性影响最大的区域。

---

### **5. 多阶段训练管道**
我们实施了一个**三阶段课程学习策略**，以逐步增强模型的推理能力：
1. **阶段 1**：在变化较小的简化堆叠上进行训练，建立稳定性原则（如质心和支撑面）的基本理解。
2. **阶段 2**：引入增强和合成场景，包括倒塌和遮挡效果，增强对多样化配置的鲁棒性。
3. **阶段 3**：在高复杂度堆叠上进行微调，利用自适应损失加权处理边缘案例和具有挑战性的场景。

每个阶段都结合动态超参数调整，包括学习率衰减、dropout正则化和数据采样调整，以确保最佳收敛。

---

### **6. 跨领域泛化**
为验证模型的实际适用性：
- 进行了领域适应实验，将模型转移到具有未知堆叠配置的新数据集。
- 评估模型在实际任务中的泛化能力，例如机器人操作和基于增强现实的虚拟对象对齐。

这一综合管道确保了模型不仅具有精度，还能适应**视觉物理推理**的多样化实际应用场景。

---

## **技术规范**

### **先决条件**
为复现并执行项目工作流，请确保在您的Python环境中安装以下依赖项：

```bash
pip install numpy matplotlib torch torchvision scikit-learn pandas opencv-python tqdm
```

## **目录结构**
项目结构如下：
```bash
Physical_Reasoning_Challenge/
├── Project_results/        # 模型结果和预测
├── saved_models/           # 训练好的模型
├── test/                   # 测试数据集
├── train/                  # 训练数据集
├── utils/                  # 工具脚本
├── Centroid_Stability_Prediction.ipynb  # 方法1
├── DETR_combined_transformer.ipynb     # 方法2
├── google_net.ipynb                    # 方法3
├── resnet_50_model.ipynb               # 最佳模型
├── train.csv              # 训练数据元数据
├── test.csv               # 测试数据元数据
└── README.md              # 文档
```

## **实验结果**

### **数据集**
- **训练集**：7,680张图像（通过3D渲染生成以增加多样性）。
- **测试集**：1,920张图像（在Kaggle上评估）。

### **性能指标**
- **精确匹配准确率**：预测与真实值完全匹配的比例。
- **Top-2准确率**：预测值在±1块范围内被视为正确的比例。

### **主要结果**
- **ResNet50模型表现最佳**：
  - **精确匹配准确率**：验证集上达到80%。
  - **泛化能力**：在未知配置上表现出强鲁棒性。

---

## **见解与贡献**

### **理论进展**
1. 通过 **Transformer** 和 **注意力机制** 集成空间推理，增强对物理稳定性的理解。
2. 突出了质心方法在遮挡和不规则配置下的局限性。

### **实际意义**
1. 通过定制化增强和自适应学习，确保了 **AR** 和 **机器人** 应用中的泛化能力。

### **未来方向**
1. **3D几何集成**：通过增强深度信息提高稳定性预测能力。
2. **物理仿真**：嵌入摩擦和重量分布等现实世界属性，进一步提升预测精度。

---

本项目推动了**视觉物理推理**领域的发展，在基于视觉的分析与实际稳定性预测之间架起了桥梁。



