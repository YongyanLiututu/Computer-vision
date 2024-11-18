import torch
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class StabilityModel:
    def __init__(self, threshold=0.9):
        # threshold value to determine object stability based on the centroid's height.
        self.threshold = threshold

    def preprocess_image(self, image):


        binary = cv2.inRange(image, (50, 50, 50), (255, 255, 255))
        return binary

    def detect_centroids(self, binary_image):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

        return centroids[1:].astype(int).tolist()

    def predict_stability(self, centroids, image_height):

        stability_results = []
        # determine whether the height of the center of mass exceeds the bottom safety line.
        threshold = image_height * self.threshold

        for centroid in centroids:

            if centroid[1] < threshold:
                stability_results.append(True)
            else:
                stability_results.append(False)

        return stability_results

    def analyze_image(self, image):

        binary_image = self.preprocess_image(image)

        centroids = self.detect_centroids(binary_image)

        image_height = image.shape[0]
        stability_predictions = self.predict_stability(centroids, image_height)

        return centroids, stability_predictions


class CustomTransformerModule(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=6, num_classes=6):
        super(CustomTransformerModule, self).__init__()
        # Define a Transformer model
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, num_classes)  # output six value

    def forward(self, x):
        # Rearrange the dimensions of input tensor
        x = x.permute(1, 0, 2)

        x = self.transformer(x, x)

        x = x.mean(dim=0)

        out = self.fc(x)

        return out

# Test whether the module is operating normally.
def check_tranformer_worked():
    global model

    if __name__ == "__main__":

        input_tensor = torch.randn(32, 10, 256)

        model = CustomTransformerModule()

        output = model(input_tensor)
        print(output.shape)  # 输出张量形状应为 (32, 6)


def compute_centroids_component(image):
    binary = cv2.inRange(image, (50, 50, 50), (255, 255, 255))


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)


    return centroids[1:].astype(int).tolist()


# Visualize the position of the center of mass of an object
def visualize_centroids(image, centroids):
    image_with_centroids = image.copy()

    def random_color():
        return [random.randint(0, 255) for _ in range(3)]

    num_centroids = min(6, len(centroids))

    colors = [random_color() for _ in range(num_centroids)]

    for i, (cX, cY) in enumerate(centroids[:num_centroids]):
        color = colors[i]
        cv2.circle(image_with_centroids, (cX, cY), 5, color, -1)

    plt.imshow(cv2.cvtColor(image_with_centroids, cv2.COLOR_BGR2RGB))

    for i, color in enumerate(colors):
        plt.scatter([], [], color=[c / 255.0 for c in color], label=f'Object {i + 1}')

    plt.legend(loc='upper right')
    plt.axis('off')
    plt.show()


# Visualize the attention weights
def visualize_attention_map(image, attention_map, num_heads=8, query_idx=0):
    # Extract the attention weights
    attention_for_query = attention_map[0, :, query_idx, :].reshape(num_heads, 10, 10)  # Reshape为 10x10 网格

    resized_image = resize(image, (224, 224))

    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))

    for i in range(num_heads):
        ax = axes[i]
        ax.imshow(resized_image)
        ax.imshow(attention_for_query[i], cmap="viridis", alpha=0.6)

        ax.set_title(f"Head {i + 1}")
        ax.axis("off")

    plt.show()
