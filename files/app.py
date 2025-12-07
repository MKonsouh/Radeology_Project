import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

# Configure the page
st.set_page_config(
    page_title="TensorFlow Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model(model_path):
    """
    Load TensorFlow model from h5 file
    Uses Streamlit's cache to avoid reloading on every interaction
    """
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_model_input_shape(model):
    """
    Extract the expected input shape from the model
    
    Args:
        model: Loaded Keras model
    
    Returns:
        Tuple of (height, width, channels) or None if cannot determine
    """
    try:
        input_shape = model.input_shape
        # input_shape is typically (None, height, width, channels) or (batch, height, width, channels)
        if len(input_shape) >= 4:
            # Return (height, width, channels), skipping batch dimension
            return (input_shape[1], input_shape[2], input_shape[3])
        elif len(input_shape) == 3:
            # Sometimes batch dimension is not included
            return input_shape
    except:
        pass
    return None

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction
    
    Args:
        image: PIL Image object
        target_size: Target size for the image (width, height)
    
    Returns:
        Preprocessed image array ready for prediction
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(image)
    
    # If image is grayscale, convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # If image has alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values (adjust based on your model's training)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def main():
    st.title("🖼️ TensorFlow Image Classifier")
    st.write("Upload an image to get predictions from your TensorFlow model")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Find available .h5 model files
    import os
    import glob
    
    # Add refresh button
    col1, col2 = st.sidebar.columns([3, 1])
    with col2:
        refresh = st.button("🔄", help="Refresh model list")
    
    # Search for .h5 files in current directory and common model directories
    search_paths = [
        "./*.h5",
        "./models/*.h5",
        "./saved_models/*.h5",
        "../*.h5",
    ]
    
    available_models = []
    for pattern in search_paths:
        available_models.extend(glob.glob(pattern, recursive=False))
    
    # Remove duplicates and sort
    available_models = sorted(list(set(available_models)))
    
    # Model selection
    if available_models:
        st.sidebar.success(f"✓ Found {len(available_models)} model(s)")
        
        # Create friendly display names
        model_display = {}
        for path in available_models:
            filename = os.path.basename(path)
            display_name = f"{filename} ({os.path.dirname(path) or '.'}/)"
            model_display[display_name] = path
        
        selected_display = st.sidebar.selectbox(
            "Select TensorFlow Model",
            options=list(model_display.keys()),
            help="Choose a model from the available .h5 files"
        )
        model_path = model_display[selected_display]
        
        # Show the actual path being used
        st.sidebar.caption(f"📂 Path: `{model_path}`")
    else:
        st.sidebar.warning("⚠️ No .h5 model files found")
        st.sidebar.info("💡 Place your model.h5 file in the current directory or use custom path below")
        model_path = None
    
    # Option to manually enter path
    with st.sidebar.expander("📝 Enter custom path"):
        custom_path = st.text_input(
            "Custom model path",
            value="",
            placeholder="e.g., /path/to/your/model.h5",
            help="Override the selection above with a custom path"
        )
        if custom_path:
            model_path = custom_path
            st.info(f"Using custom path: {custom_path}")
    
    # Load model
    model = None
    img_width, img_height = 224, 224  # defaults
    
    if model_path:
        with st.spinner("Loading model..."):
            model = load_model(model_path)
        
        if model:
            st.sidebar.success("✅ Model loaded successfully!")
            
            # Auto-detect input shape
            detected_shape = get_model_input_shape(model)
            if detected_shape and detected_shape[0] is not None and detected_shape[1] is not None:
                img_height, img_width = detected_shape[0], detected_shape[1]
                st.sidebar.info(f"📐 Detected input shape: {img_height}x{img_width}x{detected_shape[2]}")
            else:
                st.sidebar.warning("⚠️ Could not auto-detect input shape. Using manual settings.")
            
            # Image size configuration (with auto-detected defaults)
            st.sidebar.subheader("Image Preprocessing")
            use_custom = st.sidebar.checkbox("Override auto-detected size", value=False)
            
            if use_custom or detected_shape is None or detected_shape[0] is None:
                img_width = st.sidebar.number_input("Image Width", value=img_width, min_value=32, max_value=1024)
                img_height = st.sidebar.number_input("Image Height", value=img_height, min_value=32, max_value=1024)
            else:
                st.sidebar.write(f"Using detected size: **{img_width}x{img_height}**")
            
            # Display model summary
            with st.sidebar.expander("Model Architecture"):
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
    
    # Main content area
    if model is None:
        st.warning("⚠️ Please provide a valid model path in the sidebar")
        st.info("Make sure your .h5 model file is accessible at the specified path")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image file to classify"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                # Preprocess image
                processed_image = preprocess_image(image, target_size=(img_width, img_height))
                
                # Get prediction
                prediction = model.predict(processed_image, verbose=0)
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Handle different output types
                    if prediction.shape[-1] == 1:
                        # Binary classification
                        confidence = float(prediction[0][0])
                        st.metric("Confidence Score", f"{confidence:.4f}")
                        
                        if confidence > 0.5:
                            st.success(f"Predicted Class: Positive (Confidence: {confidence:.2%})")
                        else:
                            st.info(f"Predicted Class: Negative (Confidence: {(1-confidence):.2%})")
                    
                    else:
                        # Multi-class classification
                        predicted_class = np.argmax(prediction[0])
                        confidence = float(prediction[0][predicted_class])
                        
                        st.metric("Predicted Class", predicted_class)
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Show all class probabilities
                        st.write("**All Class Probabilities:**")
                        for i, prob in enumerate(prediction[0]):
                            st.progress(float(prob), text=f"Class {i}: {prob:.4f}")
                
                # Show raw prediction array
                with st.expander("Raw Prediction Output"):
                    st.write(prediction)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Please ensure the image format matches your model's expected input")

if __name__ == "__main__":
    main()
