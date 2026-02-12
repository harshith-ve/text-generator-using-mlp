# Text Generator Using MLP

A simple yet powerful text generation application built using Multi-Layer Perceptrons (MLP) and PyTorch. This project demonstrates how vanilla neural networks can capture language patterns and generate contextually relevant text, trained on "The Adventures of Sherlock Holmes" by Arthur Conan Doyle.

## 🌟 Features

- **Interactive Streamlit Web Interface**: User-friendly web application for text generation
- **Flexible Model Architecture**: Configurable embedding dimensions, context lengths, and activation functions
- **Word-Level Text Generation**: Generates text word-by-word based on learned patterns
- **Real-Time Text Typing Effect**: Displays generated text with a typewriter animation
- **Pre-trained Models**: Multiple trained models with different hyperparameter configurations
- **Reproducible Results**: Seed control for consistent text generation

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/harshith-ve/text-generator-using-mlp.git
cd text-generator-using-mlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Using the Application

1. **Configure Model Parameters**:
   - **Embedding Size**: Choose between 64 or 128 dimensions
   - **Context Length**: Select context window size (5, 10, or 15 words)
   - **Activation Function**: Choose between ReLU or Tanh

2. **Set Generation Parameters**:
   - **Number of Words**: Control the length of generated text (10-2000 words)
   - **Random Seed**: Set seed for reproducible generation

3. **Generate Text**:
   - Enter your seed text in the input box (or leave blank)
   - Click "Generate" to see the model complete your text

## 📁 Project Structure

```
text-generator-using-mlp/
│
├── app.py                    # Main Streamlit application
├── p1.py                     # Helper script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── trained_models/           # Pre-trained model weights
│   ├── model_emb64_ctx5_actReLU.pth
│   ├── model_emb64_ctx10_actReLU.pth
│   ├── model_emb64_ctx15_actReLU.pth
│   ├── model_emb128_ctx5_actReLU.pth
│   ├── model_emb128_ctx10_actReLU.pth
│   ├── model_emb128_ctx15_actReLU.pth
│   └── ... (Tanh variants)
│
└── Jupyter Notebooks/
    ├── Question1.ipynb       # Data preparation and exploration
    ├── Question2.ipynb       # Regularization techniques (L1/L2)
    └── Question3.ipynb       # Full model implementation and training
```

## 🧠 Model Architecture

The text generator uses a simple yet effective MLP architecture:

### NextWord Neural Network
```
Input (Context Words) 
    ↓
Embedding Layer (vocab_size → emb_dim)
    ↓
Flatten (block_size × emb_dim)
    ↓
Linear Layer 1 (block_size × emb_dim → 1024)
    ↓
Activation (ReLU or Tanh)
    ↓
Linear Layer 2 (1024 → vocab_size)
    ↓
Softmax (Probability Distribution)
```

### Key Components:

- **Embedding Layer**: Converts word indices to dense vector representations
- **Hidden Layer**: 1024 neurons with configurable activation function
- **Output Layer**: Produces probability distribution over vocabulary
- **Context Window**: Uses previous N words to predict the next word

## 📚 Training Details

### Dataset
- **Source**: "The Adventures of Sherlock Holmes" by Arthur Conan Doyle
- **URL**: https://www.gutenberg.org/files/1661/1661-0.txt
- **Preprocessing**:
  - Text tokenization into words
  - Vocabulary creation with string-to-index mappings
  - Context-target pair generation
  - Filtering of short sentences (< 2 words)

### Hyperparameters
- **Embedding Dimensions**: 64 or 128
- **Context Length**: 5, 10, or 15 words
- **Hidden Layer Size**: 1024 neurons
- **Activation Functions**: ReLU or Tanh
- **Training Device**: CPU

### Available Models
12 pre-trained models with different configurations:
- 2 embedding sizes × 3 context lengths × 2 activation functions = 12 models

## 📓 Jupyter Notebooks

The repository includes three educational Jupyter notebooks:

### Question1.ipynb - Data Preparation
- Loading and preprocessing text data
- Creating vocabulary mappings (stoi/itos)
- Tokenizing text into words
- Generating context-target pairs for training
- Data pipeline exploration

### Question2.ipynb - Regularization Techniques
- XOR classification problem demonstration
- MLP training without regularization (baseline)
- L1 regularization implementation (sparse weights)
- L2 regularization implementation (weight decay)
- Decision boundary visualization
- Comparison of regularization effects on overfitting

### Question3.ipynb - Model Implementation
- Complete `NextWord` neural network implementation
- Embedding layer integration
- Text generation pipeline
- Autoregressive sampling strategy
- Training loop and optimization
- Model evaluation and text generation examples

## 🛠️ Requirements

- Python 3.7+
- PyTorch
- Streamlit
- NumPy
- urllib (built-in)

See `requirements.txt` for specific versions.

## 🎯 Project Goals

This project demonstrates:
1. **Vanilla Neural Networks for NLP**: Shows that simple MLPs can learn language patterns
2. **Word-Level Generation**: Implements next-word prediction using context windows
3. **Interactive Deployment**: Provides user-friendly interface via Streamlit
4. **Educational Value**: Includes comprehensive notebooks explaining concepts
5. **Practical Application**: Real-time text generation with customizable parameters

## 📝 Model Capabilities & Limitations

### Capabilities:
- Generates contextually relevant words based on input
- Captures English language structure (capitalization, punctuation)
- Produces valid English words (or close approximations)
- Maintains sentence formatting with proper punctuation

### Limitations:
- **Not designed for meaningful sentences**: Focus is on format, not semantics
- **Limited computational resources**: Simple architecture for demonstration
- **Single-text training**: Trained only on Sherlock Holmes stories
- **No attention mechanism**: Uses fixed-size context windows

## 🤝 Acknowledgments

- Dataset: ["The Adventures of Sherlock Holmes" by Arthur Conan Doyle](https://www.gutenberg.org/files/1661/1661-0.txt) from Project Gutenberg
- Framework: PyTorch for deep learning implementation
- Interface: Streamlit for web application deployment

## 📜 License

This project is open source and available for educational purposes.

## 🔗 Repository

Visit the repository at: [harshith-ve/text-generator-using-mlp](https://github.com/harshith-ve/text-generator-using-mlp)

---

**Note**: This is an educational project demonstrating fundamental concepts of neural networks for text generation. The goal is to show how even simple architectures can learn and replicate language patterns, not to create state-of-the-art text generation models.