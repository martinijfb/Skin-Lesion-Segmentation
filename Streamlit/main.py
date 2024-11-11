import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from unet_module import load_model, generate_mask

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device=device)

def main():
    st.title("Skin Lesion Segmentation")
    st.write("Upload an image of a skin lesion to segment it using a UNet model.")

    # Sample image option
    use_sample = st.checkbox("Use Sample Image")

    # Load the image
    if use_sample:
        image = Image.open("DataSample/sample_image.jpg")
        original_size = image.size
        st.image(image, caption="Sample Image", use_container_width=True)
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            original_size = image.size
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.warning("Please upload an image or select the sample option.")
            return

    # Preprocess the image
    mask = generate_mask(model, image, image_size=original_size, device=device)

    # Display the segmentation result
    st.subheader("Segmentation Result")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    st.pyplot(fig)


if __name__ == "__main__":
    main()

