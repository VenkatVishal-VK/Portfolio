import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load image (grayscale)
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Convert to float
img = np.float32(img)

# Apply PCA
pca = PCA(50)  # reduce to 50 components
reduced = pca.fit_transform(img)

# Reconstruct image
reconstructed = pca.inverse_transform(reduced)

# Show images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed, cmap='gray')

plt.show()