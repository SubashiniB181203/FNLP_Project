from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow opencv-python matplotlib

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load and preprocess dataset
def load_images_and_labels(dataset_path):
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append(idx)
    return np.array(images), np.array(labels), class_names

dataset_path = '/content/drive/MyDrive/kvasir-dataset'
images, labels, class_names = load_images_and_labels(dataset_path)
images = images.astype('float32') / 255.0

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# 3. Convert labels to categorical
num_classes = len(class_names)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# 4. Image augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# 5. Load DenseNet121 and build model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)


# 6. Compile model
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train model with early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                    validation_data=(X_test, y_test_cat),
                    epochs=25, callbacks=[early_stop])

# 8. Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# 9. Fine-tune the model (unfreeze last N layers)
fine_tune_at = 300  # Unfreeze from this layer index onward (can tune based on performance)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Train again (fine-tuning)
fine_tune_history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                              validation_data=(X_test, y_test_cat),
                              epochs=10,
                              callbacks=[early_stop])

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_pred_finetuned = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_finetuned, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

model.save('kvasir_model.h5')

!pip install streamlit opencv-python tensorflow

!pip install streamlit pyngrok

%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('/content/kvasir_model.h5')

# Define class names
class_names = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Gastrointestinal Image Classification")
st.write("Upload an image of a gastrointestinal tract for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"Predicted Class: **{predicted_class}**")

!pip install openai==0.28

import openai
openai.api_key = "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"

%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import openai
import os

# Load your API key from Streamlit secrets or environment variable
openai.api_key = "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"

# Load the trained model
model = tf.keras.models.load_model('/content/kvasir_model.h5')

# Define class names
class_names = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def ask_gpt(question, disease):
    prompt = f"You are a medical assistant. The disease detected is '{disease}'. Answer the user's question about this condition.\n\nUser: {question}\nAssistant:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

st.title("🩺 Gastrointestinal Image Classification + GPT Chatbot")
st.write("Upload an image of a gastrointestinal tract for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Class: **{predicted_class}**")

    # GPT chatbot section
    st.subheader("💬 Ask GPT about this condition")
    user_question = st.text_input("Ask a question about the detected condition:")

    if user_question:
        with st.spinner("Consulting medical assistant..."):
            response = ask_gpt(user_question, predicted_class)
        st.write(response)

!pip install transformers huggingface_hub

from huggingface_hub import login

# Authenticate with your Hugging Face account
login(token="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")

%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load Keras model
model = tf.keras.models.load_model('/content/kvasir_model.h5')

# Disease class names
class_names = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

# Preprocess input image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit interface
st.title("🩺 Gastrointestinal Disease Classifier + BioBERT Q&A")
st.write("Upload a GI tract image to classify and ask questions based on results.")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and display uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess and predict disease
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"✅ Predicted Disease: **{predicted_class}**")

    # Load BioBERT model for Q&A
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Q&A section
    st.header("Ask a medical question about the result")
    question = st.text_input("Type your question below:")

    if question:
        context = f"The predicted condition is {predicted_class}.Gastrointestinal (GI) diseases refer to disorders involving the digestive tract, including the esophagus, stomach, small intestine, large intestine, rectum, liver, gallbladder, and pancreas. Common symptoms include abdominal pain, bloating, diarrhea, constipation, nausea, and vomiting. Some prevalent GI disorders are irritable bowel syndrome (IBS), gastroesophageal reflux disease (GERD), and Crohn's disease."
        answer = qa_pipeline(question=question, context=context)
        st.info(f"💬 Answer: **{answer['answer']}**")

!ngrok config add-authtoken 2wPAqYdyaz3VAiydydWDg54gjm2_XRkEXg5zNK9kUD4DKeQn
from pyngrok import ngrok

# Kill all ngrok processes to ensure a clean start
!killall ngrok

# Start the Streamlit app in the background
!streamlit run app.py &>/content/log.txt &

# Connect to the Streamlit app using ngrok and get the public URL
url = ngrok.connect(8501)
print("Streamlit app running at:", url)  # Now 'url' is defined and can be used

# If you need to disconnect later, you can use:
# ngrok.disconnect(url.public_url)
