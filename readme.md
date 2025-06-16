# Generating Descriptive Captions from Semantic Text Relationships

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras)](https://keras.io/)

This repository contains the source code for the fourth-year (2021-2022) project from the University of Aleppo, Faculty of Informatics. The project focuses on generating human-readable descriptive text from structured semantic data using an AI and NLP-based approach.


## 📌 Overview

The core objective of this project is to build a system that can translate structured textual relationships (e.g., `subject-predicate-object` triples) into natural, descriptive sentences. This is achieved by training a sequence-to-sequence (Seq2Seq) neural network.

For instance, given a set of simple relations like:
*   `man - wearing - sneakers`
*   `man - has - red shirt`
*   `shade - is along - street`

The model learns to generate a cohesive description, such as: **"A man in a red shirt is wearing sneakers on a street with a shadow."**

A primary application for this technology is to automatically generate descriptive `alt-text` for images on the web, enhancing accessibility and providing context in situations with slow internet connections or limited disk space.

## ✨ Key Features

*   **Data-Driven:** Utilizes the rich **Visual Genome** dataset, which contains dense annotations of objects, attributes, and relationships within images.
*   **NoSQL Database Integration:** Employs MongoDB to efficiently store, query, and merge the complex, nested JSON data from the dataset into a normalized format suitable for training.
*   **Advanced NLP Pipeline:**
    *   **Subword Tokenization:** Uses **Byte Pair Encoding (BPE)** to handle rare words and morphological variations effectively.
    *   **Semantic Embeddings:** Leverages **Word2Vec (Skip-gram)** to create dense vector representations of words that capture semantic meaning.
*   **Deep Learning Model:** Implements a **Sequence-to-Sequence (Seq2Seq)** architecture with **LSTM** (Long Short-Term Memory) units to handle variable-length input and output sequences.
*   **Efficient Training:** Employs the **Teacher Forcing** technique to stabilize and accelerate the training process of the recurrent neural network.

## 🏗️ System Architecture

The project is broken down into three main stages: Data Preparation, NLP Preprocessing, and Model Training.

### 1. Data Preparation

The Visual Genome dataset is composed of multiple large JSON files containing information about objects, attributes, relationships, and regions. The links to the necessary JSON files can be found in the `Semantic-Caption-Generation.gdrive` file.

1.  **Loading Data:** The raw JSON files are loaded into a MongoDB database.
2.  **Normalization & Merging:** A series of aggregation queries (see `appendix` of the report **in Arabic**) are run to merge these disparate sources. The goal is to create a single, unified document for each image that links every object to its attributes and its role in various relationships. This step is crucial for creating clean training pairs.

### 2. NLP Preprocessing Pipeline

Once the data is structured, it is fed through an NLP pipeline to prepare it for the model.

1.  **Training Pair Extraction:** For each image region, we extract the input (a concatenation of semantic triples like `"cow in grass <SEP> cow has head"`) and the target output (the human-written phrase like `"cow's head in the grass"`).
2.  **Tokenization:** A Byte Pair Encoding (BPE) tokenizer is trained on the corpus. This breaks words down into common subword units (e.g., `tokenization` -> `token`, `ization`), which helps the model generalize better. Special tokens like `<PAD>`, `<START>`, `<END>`, and `<SEP>` are added.
3.  **Word Embeddings:** A Word2Vec (Skip-gram) model is trained on the text corpus to generate 300-dimensional embeddings for each token in our vocabulary. These embeddings serve as the input to the neural network.

### 3. Model Architecture & Training

A Seq2Seq model with an Encoder-Decoder structure is used to learn the mapping from semantic relations to descriptive text.



*   **Encoder:** An LSTM layer processes the input sequence of embedded tokens and compresses its information into a fixed-size context vector (the final hidden state and cell state of the LSTM).
*   **Decoder:** A separate LSTM layer is initialized with the Encoder's context vector. During training, it uses **Teacher Forcing**, where the ground-truth token from the previous timestep is fed as input to predict the next token.
*   **Output Layer:** A Dense layer with a `tanh` activation function maps the Decoder's output to the dimensionality of the word embeddings, performing a regression task to predict the embedding of the next word in the sequence.

The model was trained for 200 epochs, achieving:
*   **Training Accuracy:** ~89.8%
*   **Validation/Testing Accuracy:** ~74.4%

## 🛠️ Technologies Used

*   **Language:** Python
*   **Deep Learning:** TensorFlow, Keras
*   **Database:** MongoDB
*   **NLP/Data Handling:** NLTK, Pandas, NumPy

## 🚀 Getting Started

### Prerequisites

*   Python 3.8 or higher
*   An active MongoDB instance
*   The [Visual Genome](https://visualgenome.org/api/v0/api_home.html) dataset. (Link to the required JSON files are in `Semantic-Caption-Generation.gdrive`).

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/semantic-caption-generation.git
    cd semantic-caption-generation
    ```

2.  **Set up a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Database Setup:**
    *   Download the Visual Genome dataset JSON files.
    *   Set up a MongoDB database and import the JSON files into separate collections.
    *   Run the data preparation scripts (containing the MongoDB aggregation queries) to create the final, normalized collection for training, already provided in the `Semantic-Caption-Generation.gdrive` file.

### Running the Model

1.  **Train the model:**
    ```sh
    python model_cat_entropy.py
    ```

2.  **Generate a description:**
    ```sh
    python model_cat_entropy2.py 
    ```

## 📊 Results & Examples

Here is an example from the validation set demonstrating the model's performance.

**Example 1:**

*   **Input Relations:** `['throwing', '</w>', 'a', '</w>', 'ball']`
*   **Desired Output (Ground Truth):** `['p', 'it', 'ch', 'er', '</w>', 'thr', 'ow', 'ing', '</w>', 'a', '</w>', 'b', 'all']`
    *   *Reconstructed Text:* "pitcher throwing a ball"
*   **Actual Model Output:** `['a', '</w>', 'skateboard', 're', '<PAD>', 'k', 'airplane', 'wearing', ...]`
    *   *Reconstructed Text:* "a skateboarder..." (The model shows some confusion here, which is expected given the data complexity and potential scarcity).

The discrepancy in results highlights the challenges of the task and suggests areas for future improvement.

## 🔮 Future Work

*   **Integrate Computer Vision:** Extend the project by building a CV model (e.g., YOLO, Faster R-CNN) to automatically detect objects and relationships from an image, creating a full end-to-end image-to-semantic-caption system.
*   **Explore Transformer Models:** Replace the LSTM-based Seq2Seq architecture with a Transformer-based one to potentially capture long-range dependencies more effectively.
*   **Increase Dataset Size:** Use a larger and more diverse dataset to improve the model's generalization and reduce errors.
*   **Hyperparameter Optimization:** Conduct a systematic search for optimal parameters, such as embedding dimensions, number of LSTM units, and learning rate.

## 👥 Acknowledgements

This project was developed under the supervision of:
*   **Dr. Fadel Sukkar**

We thank the Department of Artificial Intelligence for their guidance and support throughout this project.