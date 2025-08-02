import numpy as np
import struct
import os

def read_idx_images(filename):
    """
    Read MNIST images from IDX3-UBYTE format
    """
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in {filename}')
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    
    return images

def read_idx_labels(filename):
    """
    Read MNIST labels from IDX1-UBYTE format
    """
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in {filename}')
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels

def load_mnist_data(dataset_path):
    """
    Load MNIST training and test data from dataset folder
    """
    # Define file paths
    train_images_path = os.path.join(dataset_path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(dataset_path, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(dataset_path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(dataset_path, 't10k-labels.idx1-ubyte')
    
    # Load training data
    print("Loading training images...")
    train_images = read_idx_images(train_images_path)
    print("Loading training labels...")
    train_labels = read_idx_labels(train_labels_path)
    
    # Load test data
    print("Loading test images...")
    test_images = read_idx_images(test_images_path)
    print("Loading test labels...")
    test_labels = read_idx_labels(test_labels_path)
    
    return (train_images, train_labels), (test_images, test_labels)

def preprocess_data(train_images, train_labels, test_images, test_labels):
    """
    Preprocess the MNIST data for CNN training
    """
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Reshape images to add channel dimension (for CNN)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    
    # Convert labels to categorical (one-hot encoding)
    from tensorflow.keras.utils import to_categorical
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    # Test the data loading
    dataset_path = "./dataset"
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data(dataset_path)
    
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Test preprocessing
    train_images, train_labels, test_images, test_labels = preprocess_data(
        train_images, train_labels, test_images, test_labels
    )
    
    print(f"After preprocessing:")
    print(f"Training images shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Training images range: [{train_images.min():.3f}, {train_images.max():.3f}]")
