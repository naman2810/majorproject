import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import time

# Custom CSS with the green-toned color palette
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ECFAE5 0%, #DDF6D2 100%);
    }
    .title {
        font-family: 'Georgia', serif;
        color: #06402B;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        color: #06402B;
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .stFileUploader {
        background-color: #ECFAE5;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .stButton > button {
        background-color: #06402B;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #B0DB9C;
    }
    .image-container {
        border: 5px solid #CAE8BD;
        border-radius: 15px;
        padding: 15px;
        background-color: #ECFAE5;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        margin: 10px 0;
    }
    .diagnosis {
        font-family: 'Arial', sans-serif;
        font-size: 26px;
        font-weight: bold;
        color: #06402B;
        text-align: center;
        margin-top: 20px;
    }
    .healthy {
        color: #06402B;
    }
    .stExpander {
        background-color: #ECFAE5;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        color: #06402B;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("""
    <h1 class='title'>
        CropConnect: A Deep Learning Based<br>
        Framework for Tomato Leaf Disease<br>
        Diagnosis
    </h1>
""", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Upload a tomato leaf image 🍅 to diagnose potential diseases with AI-powered precision.</p>", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model_resnet = load_model('models/resnet.h5')
        model_inception = load_model('models/inception.h5')
        model_efficient = load_model('models/efficient.h5')
        model_vgg19 = load_model('models/vgg19.h5')
        return [model_resnet, model_inception, model_efficient, model_vgg19]
    except Exception as e:
        st.error(f"Error loading disease models: {str(e)}")
        return None

@st.cache_resource
def load_leaf_detector():
    try:
        return load_model('models/leaf_nonleaf_classifier.h5')
    except Exception as e:
        st.error(f"Error loading leaf detector model: {str(e)}")
        return None

models = load_models()
leaf_detector = load_leaf_detector()
if models is None or leaf_detector is None:
    st.stop()

def ensemble_predict(models, X):
    preds = []
    for model in models:
        input_shape = model.input_shape[1:3]
        if X.shape[1:3] != input_shape:
            X_resized = tf.image.resize(X, input_shape).numpy()
        else:
            X_resized = X
        pred = model.predict(X_resized, verbose=0)
        preds.append(pred)
    avg_preds = np.mean(preds, axis=0)
    return avg_preds

# Disease information
disease_info = {
    'Bacterial Spot': {
        'description': 'Bacterial Spot is caused by Xanthomonas species, leading to small, water-soaked spots on leaves that turn dark brown or black.',
        'symptoms': '- Small, water-soaked spots\n- Dark brown to black lesions with yellow halos\n- Leaf yellowing and drop\n- Reduced fruit quality'
    },
    'Early Blight': {
        'description': 'Early Blight, caused by Alternaria solani, is a fungal disease affecting leaves and stems, often in warm, wet conditions.',
        'symptoms': '- Concentric rings or "bullseye" spots\n- Dark brown to black spots\n- Leaf yellowing and premature drop\n- Stem lesions'
    },
    'Late Blight': {
        'description': 'Late Blight, caused by Phytophthora infestans, is a devastating fungal disease that thrives in cool, moist environments.',
        'symptoms': '- Large, irregular grayish-green spots\n- White mold on leaf undersides\n- Rapid leaf and stem decay\n- Fruit rot'
    },
    'Leaf Mold': {
        'description': 'Leaf Mold, caused by Passalora fulva, is a fungal disease common in humid conditions, affecting leaf undersides.',
        'symptoms': '- Yellow spots on upper leaf surfaces\n- Grayish-white mold on leaf undersides\n- Leaf curling and drying\n- Reduced photosynthesis'
    },
    'Septoria Leaf Spot': {
        'description': 'Septoria Leaf Spot, caused by Septoria lycopersici, is a fungal disease causing numerous small spots on leaves.',
        'symptoms': '- Small, circular spots with gray centers\n- Yellow halos around spots\n- Leaf yellowing and drop\n- Lower leaf infection'
    },
    'Spider Mites': {
        'description': 'Spider Mites are tiny pests that feed on leaf sap, causing stippling and reduced plant vigor.',
        'symptoms': '- Tiny yellow or white speckles\n- Fine webbing on leaf undersides\n- Leaf bronzing and drying\n- Stunted plant growth'
    },
    'Target Spot': {
        'description': 'Target Spot, caused by Corynespora cassiicola, is a fungal disease with distinctive target-like spots on leaves.',
        'symptoms': '- Circular spots with concentric rings\n- Dark brown to black centers\n- Leaf yellowing and drop\n- Fruit lesions'
    },
    'Yellow Leaf Curl Virus': {
        'description': 'Yellow Leaf Curl Virus is transmitted by whiteflies, causing severe leaf curling and stunted growth.',
        'symptoms': '- Upward leaf curling\n- Yellowing along leaf margins\n- Stunted plant growth\n- Reduced fruit yield'
    },
    'Mosaic Virus': {
        'description': 'Mosaic Virus causes mottled leaves and reduced yield, spread by aphids or infected tools.',
        'symptoms': '- Mottled green and yellow leaves\n- Leaf distortion and wrinkling\n- Stunted growth\n- Poor fruit development'
    },
    'Healthy': {
        'description': 'The leaf is free from diseases or pest damage, indicating a healthy tomato plant.',
        'symptoms': '- Uniform green color\n- No spots or lesions\n- Normal leaf shape and size\n- Healthy plant growth'
    }
}

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### Upload Tomato Leaf Image 🍅")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a tomato leaf for accurate diagnosis."
    )

if uploaded_file is not None:
    with col2:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(img, caption='Uploaded Tomato Leaf 🍅', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Diagnose Leaf", key="diagnose"):
        with st.spinner("Analyzing leaf..."):
            time.sleep(1)
            img_array = image.img_to_array(img) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Step 1: Check if it's a leaf
            is_leaf_prob = leaf_detector.predict(img_batch, verbose=0)[0][0]

            if is_leaf_prob > 0.5:
                st.markdown(
                    "<p class='diagnosis'>❌ The uploaded image does not appear to be a tomato leaf. Please try again.</p>",
                    unsafe_allow_html=True
                )
            else:
                # Step 2: Predict disease
                preds = ensemble_predict(models, img_batch)
                class_idx = np.argmax(preds, axis=1)[0]
                class_labels = list(disease_info.keys())
                diagnosis = class_labels[class_idx]
                confidence = preds[0][class_idx] * 100

                if diagnosis.lower() == "healthy":
                    st.markdown(
                        f"<p class='diagnosis healthy'>The leaf is <strong>{diagnosis}</strong>! 🍅🌱</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<p class='diagnosis'>The leaf is diagnosed with <strong>{diagnosis}</strong> ({confidence:.2f}% confidence) 🍅</p>",
                        unsafe_allow_html=True
                    )

                with st.expander("Learn More About This Diagnosis"):
                    st.markdown(f"**Description**: {disease_info[diagnosis]['description']}")
                    st.markdown("**Symptoms**:")
                    for symptom in disease_info[diagnosis]['symptoms'].split('\n'):
                        st.markdown(f"- {symptom}")

# Footer
st.markdown("""
    <hr>
    <p class='footer'>
        Powered by CropConnect | Built with Deep Learning | 2025
    </p>
""", unsafe_allow_html=True)
