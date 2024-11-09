import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import your UNet model and metrics
from unet_module import load_model,  preprocess_image  # Replace with actual module name

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device=device)

def main():
    st.title("Skin Lesion Segmentation")
    st.write("Upload an image of a skin lesion to segment it using the UNet model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size # Keep the original size for later
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        input_tensor = preprocess_image(image).to(device)

        # Run model inference
        with torch.no_grad():
            model.eval()
            output = model(input_tensor)
            output = torch.sigmoid(output).cpu().squeeze().numpy()  # Sigmoid and convert to numpy

        # Threshold and convert to binary mask
        binary_mask = (output > 0.5).astype(np.uint8) * 255  # Convert to 0 or 255 for visualization

        # Resize the output mask back to the original image size
        output_resized = Image.fromarray(binary_mask)  # Convert to a PIL image
        output_resized = output_resized.resize(original_size, Image.NEAREST)  # Resize to original size

        # Display results
        st.subheader("Segmentation Result")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[1].imshow(output_resized, cmap="gray")
        ax[1].set_title("Predicted Mask")
        st.pyplot(fig)


if __name__ == "__main__":
    main()

