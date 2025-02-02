# Neural Machine Translation with Attention 🚀

A PyTorch implementation of a Sequence-to-Sequence model with Attention for English-Spanish translation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Features

- **Bidirectional GRU Encoder**: Captures context from both directions of the input sequence
- **Attention Mechanism**: Helps the model focus on relevant parts of the input sequence
- **Teacher Forcing**: Implements curriculum learning for better training stability
- **Dynamic Batching**: Efficient training with variable sequence lengths
- **Hugging Face Integration**: Uses MarianTokenizer for robust text processing

## 🏗️ Architecture

The model consists of three main components:

1. **Encoder**: Bidirectional GRU network that processes input sequences
2. **Attention**: Computes attention weights for each encoder state
3. **Decoder**: GRU network that generates translations using attention context

```plaintext
Input → Encoder → Attention → Decoder → Translation
      ↑          ↑          ↑
      Embeddings Context    Attention Weights
```

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nmt-attention.git
cd nmt-attention
```

2. Install dependencies:
```bash
pip install torch transformers datasets
```

3. Train the model:
```python
python train.py
```

4. Translate text:
```python
from translate import translate
text = "How are you?"
translated = translate(model, text, tokenizer)
print(translated)

# Loading a saved model
model = Seq2Seq(encoder, decoder, device)
model.load_state_dict(torch.load('LSTM_text_generator.pth'))
model.eval()
```

## 📊 Model Performance

Training metrics after 10 epochs:
- Initial Loss: 11.147
- Final Loss: 3.527
- Training Time: ~2 hours on NVIDIA V100

## 🔧 Hyperparameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CLIP = 1.0
N_EPOCHS = 10
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
```

## 📚 Dataset

Using the `loresiensis/corpus-en-es` dataset from Hugging Face Hub, which provides English-Spanish sentence pairs for training.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper
- Hugging Face for the transformers library and datasets
- PyTorch team for the amazing deep learning framework

---
⭐️ If you found this project helpful, please consider giving it a star!