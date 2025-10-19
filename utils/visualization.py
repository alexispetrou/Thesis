import matplotlib.pyplot as plt

def show_image_with_mask(image, mask, title="MRI with Mask"):
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=0.4)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.legend()
    plt.show()
