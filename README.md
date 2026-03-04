---
title: PixelNur - CNN-Based Steganography
emoji: 🔒
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🔒 PixelNur

### Advanced CNN-Based Steganography System

*Hide secret messages inside images with military-grade encryption*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/imrushikesh09/pixelnur)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[Try It Now](https://huggingface.co/spaces/imrushikesh09/pixelnur) • [Report Bug](https://github.com/rushikeshxdev/pixelNur/issues) • [Documentation](#-how-it-works)

</div>

---

## ✨ Features

🧠 **CNN-Based Adaptive Embedding** • Intelligent texture analysis  
🔐 **AES-256 Encryption** • Military-grade security  
🛡️ **Attack Resistant** • Survives JPEG compression & resizing  
📊 **High Quality** • PSNR >40 dB, SSIM >0.99  
⚡ **GPU Accelerated** • Fast processing with CUDA support  
🎯 **Blind Extraction** • No original image needed

## 🚀 Quick Start

### Embed a Message
```bash
1. Upload your image (PNG/JPEG)
2. Type your secret message
3. Set a strong password (16+ chars)
4. Choose robustness level
5. Download the stego image
Extract a Message
1. Upload the stego image
2. Enter the password
3. Reveal the hidden message
🎚️ Robustness Levels
Level	Quality	Capacity	Resistance	Best For
None	⭐⭐⭐⭐⭐	100%	None	Trusted channels
Low	⭐⭐⭐⭐	80%	Light JPEG	Social media
Medium	⭐⭐⭐	27%	JPEG QF 75+	Messaging apps
High	⭐⭐	16%	Max protection	Hostile environments
🔬 How It Works
Image → LWT Transform → CNN Mask → Encrypt Message → LSB Embed → Stego Image
Wavelet Transform: Decomposes image into frequency bands
CNN Analysis: Generates texture-aware embedding mask
Encryption: AES-256 with PBKDF2 key derivation
Robustness: Error correction + spatial replication
Embedding: LSB matching in frequency domain
📊 Performance
Image Quality: PSNR 40-48 dB, SSIM 0.96-0.99
Capacity: 8-128 KB (depends on image size & robustness)
Speed: 2-15 seconds per image (GPU accelerated)
🛠️ Tech Stack
Deep Learning: PyTorch, CNN-based texture analysis
Image Processing: OpenCV, Lifting Wavelet Transform
Encryption: AES-256-CBC, PBKDF2-HMAC-SHA256
Interface: Gradio web UI
Deployment: Hugging Face Spaces
📦 Installation
# Clone the repository
git clone https://github.com/rushikeshxdev/pixelNur.git
cd pixelNur

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
🎓 Research Team
KIT's College of Engineering, Kolhapur

Rushikesh Randive (2223000930)
Aman Qureshi (2223000503)
Ankita Patil (2223000302)
Madhura Patil (2223000060)
📄 License
MIT License - see LICENSE for details

🙏 Acknowledgments
Built with ❤️ for privacy • Powered by Hugging Face Spaces

<div align="center">
⭐ Star this repo if you find it useful!

</div>