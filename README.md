# Image-Caption-Generator-using-CNN-LSTM-Encoder-Decoder-Architecture
## 📘 Overview
This project implements an image captioning model that automatically generates descriptive captions for input images.
It combines Convolutional Neural Networks (CNNs) for visual feature extraction and Long Short-Term Memory (LSTM) networks for sequential text generation - a classic Encoder-Decoder architecture widely used in modern vision-language models.
## 🎯 Objective
To build a deep learning model that learns to describe the content of an image in natural language, using the Flickr8k dataset (8,000 images and five captions each).
## Architecture
| Component          | Role              | Description                                                                                                          |
| ------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Encoder (CNN)**  | Feature extractor | A pretrained **VGG16** network (without top layers) is used to encode each image into a fixed-length feature vector. |
| **Decoder (LSTM)** | Caption generator | An **LSTM** network takes the encoded visual features and generates captions word by word.                           |
| **Tokenizer**      | Text processing   | Converts words to integer sequences for model input.                                                                 |
| **BLEU Score**     | Evaluation        | Measures the similarity between predicted captions and reference captions.                                           |
## ⚙️ Workflow
Data Preprocessing :

Clean and tokenize text captions.

Add startseq and endseq tags for sequence boundaries.

Extract image features using pretrained VGG16.

Pad sequences to fixed length (right-padding for cuDNN compatibility).

Model Training:

Train the CNN–LSTM model on image–caption pairs.

Use teacher forcing during training.

Evaluate using BLEU-1 and BLEU-2 metrics.

Caption Generation:

Input an image → Encoder extracts features → Decoder predicts caption tokens → Tokens converted back to text.
## Sample Results
Example Output:
--------------------- Actual Captions ---------------------
startseq man skis past another man displaying paintings in the snow endseq  
startseq man on skis looking at artwork for sale in the snow endseq  
...

-------------------- Predicted Caption --------------------
startseq man looks at the man in the snow endseq

BLEU Scores:

BLEU-1: 0.403

BLEU-2: 0.205

## 💡 What I Did
This repository is based on a Kaggle notebook on Image Captioning (CNN–LSTM).
I reproduced, debugged, and enhanced it with:
🧩 Fixed preprocessing issues (regex and sequence padding).

⚡ Enabled cuDNN optimization for faster GPU training.

🧹 Improved text cleaning and tokenization pipeline.

📈 Evaluated model performance using BLEU metrics.

🧠 Added explanations for Encoder–Decoder mechanisms and training logic.

Acknowledgment:
Original project concept and dataset source: Kaggle Image Caption Generator (Flickr8k).
This repository is a reproduction and enhancement of that project for learning and portfolio purposes.

## Technologies Used

Python

TensorFlow

NumPy, Pandas

Matplotlib, Pillow

NLTK
## Future Improvements

Implement Beam Search for improved caption decoding.

Experiment with Attention Mechanism (e.g., Bahdanau / Luong).

Train with larger datasets such as Flickr30k or MS COCO.

Deploy a web app for live image captioning.

## 📚 Learning Outcome

Through this project, I explored:

The inner workings of Encoder–Decoder deep learning models.

How CNNs and LSTMs can jointly model vision and language.

The importance of text preprocessing and sequence alignment.

Evaluation of NLP models using BLEU metrics.
##  Reference

Kaggle: Image Caption Generator using CNN–LSTM (Flickr8k Dataset)

Paper: Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2015)

## License

This project is licensed under the [MIT License](./LICENSE) © 2025 Rakibul Hasan.  
Feel free to use and modify the code for educational and research purposes, with proper attribution.
