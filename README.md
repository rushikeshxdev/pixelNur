---
title: PixelNur Phase 2 - CNN-Based Steganography
emoji: 🔐
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# PixelNur Phase 2 🔐

**Advanced CNN-Based Steganography with Robustness Features**

PixelNur Phase 2 is a state-of-the-art steganography system that hides secret messages within images using deep learning and advanced signal processing techniques. The system provides multiple robustness levels to protect hidden data against common image attacks like JPEG compression, resizing, and noise.

## ✨ Key Features

- **🧠 CNN-Based Embedding**: Uses convolutional neural networks to generate adaptive embedding masks based on image texture
- **🔒 Military-Grade Encryption**: AES-256 encryption with password-based key derivation (PBKDF2)
- **🛡️ Robustness Levels**: Four levels of protection against image attacks (0-3)
- **📊 High Quality**: Maintains excellent visual quality (PSNR >40 dB, SSIM >0.99)
- **⚡ GPU Acceleration**: Automatic CUDA support for faster processing
- **🎯 Blind Extraction**: No original image needed for message recovery

## 🚀 Quick Start

### Embedding a Message

1. **Upload Cover Image**: Select a PNG, JPEG, or BMP image (minimum 256×256 pixels)
2. **Enter Secret Message**: Type the text you want to hide
3. **Set Password**: Enter a strong password (minimum 16 characters) - you'll need this to extract the message
4. **Select Robustness Level**: Choose based on your needs:
   - **None (0)**: Best quality, no attack resistance
   - **Low (1)**: Light protection, resists minor JPEG compression
   - **Medium (2)**: Moderate protection, resists JPEG QF 75+
   - **High (3)**: Maximum protection, resists JPEG, resizing, and noise
5. **Click "Embed Message"**: Download the resulting stego image

### Extracting a Message

1. **Upload Stego Image**: Select the image containing the hidden message
2. **Enter Password**: Use the same password from embedding
3. **Click "Extract Message"**: The hidden message will be revealed

## 🎚️ Robustness Levels Explained

| Level | Quality (PSNR) | Capacity | Attack Resistance | Use Case |
|-------|---------------|----------|-------------------|----------|
| **None (0)** | 44-48 dB | 100% | None | Maximum quality, trusted channels |
| **Low (1)** | 42-46 dB | 80% | Light JPEG (QF 90+) | Social media with minimal compression |
| **Medium (2)** | 40-44 dB | 27% | JPEG QF 75+, minor resizing | Email, messaging apps |
| **High (3)** | 38-42 dB | 16% | JPEG QF 75, resizing, noise | Hostile environments, maximum security |

**Trade-offs:**
- Higher robustness = Lower capacity and slightly lower image quality
- Lower robustness = Higher capacity and better image quality
- Choose based on expected image transformations

## 📈 Performance Metrics

### Image Quality
- **PSNR (Peak Signal-to-Noise Ratio)**: 40-48 dB depending on robustness level
  - Higher is better (>40 dB is excellent)
  - Measures pixel-level similarity
- **SSIM (Structural Similarity Index)**: 0.96-0.99+
  - Closer to 1.0 is better
  - Measures perceptual similarity

### Capacity
- **512×512 image**: ~8-32 KB depending on robustness level
- **1024×1024 image**: ~32-128 KB depending on robustness level
- Capacity scales with image size and decreases with higher robustness

### Processing Time
- **Embedding (512×512)**: 2-5 seconds (robustness 0), 5-15 seconds (robustness 3)
- **Extraction (512×512)**: 2-5 seconds
- GPU acceleration provides 2-3× speedup when available

## 🔬 Technical Architecture

### Core Technologies
- **CNN Module**: Generates texture-aware embedding masks using gradient analysis
- **LWT (Lifting Wavelet Transform)**: Decomposes images into frequency sub-bands
- **Robustness Layer**: Implements error correction coding (Reed-Solomon) and spatial replication
- **Encryption Service**: AES-256-CBC with PBKDF2 key derivation (100,000 iterations)
- **Metrics Service**: Calculates PSNR, SSIM, and capacity metrics

### Embedding Pipeline
1. Image decomposition via LWT (Haar wavelet)
2. CNN-based mask generation (texture analysis)
3. Message encryption (AES-256)
4. Robustness encoding (ECC + replication)
5. LSB matching in LH sub-band
6. Inverse LWT reconstruction

### Extraction Pipeline
1. Image decomposition via LWT
2. Blind extraction from LH sub-band
3. Robustness decoding (majority voting + ECC)
4. Message decryption (AES-256)
5. Validation and output

## 🖼️ Example Results

### Visual Quality Comparison

**Original Image** → **Stego Image (Robustness 0)** → **Stego Image (Robustness 3)**

The differences are imperceptible to the human eye. Even at robustness level 3, the image maintains excellent visual quality with PSNR >38 dB.

### Attack Resistance

| Attack Type | Robustness 0 | Robustness 1 | Robustness 2 | Robustness 3 |
|-------------|--------------|--------------|--------------|--------------|
| JPEG QF 95 | ❌ Fails | ✅ Recovers | ✅ Recovers | ✅ Recovers |
| JPEG QF 85 | ❌ Fails | ❌ Fails | ✅ Recovers | ✅ Recovers |
| JPEG QF 75 | ❌ Fails | ❌ Fails | ⚠️ Partial | ✅ Recovers |
| Resize 90% | ❌ Fails | ❌ Fails | ⚠️ Partial | ✅ Recovers |
| Gaussian Noise (σ=2) | ❌ Fails | ❌ Fails | ⚠️ Partial | ✅ Recovers |

## 🔐 Security Considerations

### Password Requirements
- **Minimum length**: 16 characters
- **Recommendation**: Use a strong, unique password with mixed case, numbers, and symbols
- **Key derivation**: PBKDF2-HMAC-SHA256 with 100,000 iterations
- **Storage**: Never store passwords - they are only used for key derivation

### Security Features
- ✅ AES-256-CBC encryption (military-grade)
- ✅ Random IV generation for each embedding
- ✅ HMAC authentication (prevents tampering)
- ✅ Secure key derivation (PBKDF2)
- ✅ No metadata leakage

### Important Notes
⚠️ **Research Prototype**: This is a research implementation. For production use, conduct thorough security audits.

⚠️ **Password Security**: If you forget the password, the message cannot be recovered. There is no password reset.

⚠️ **Image Modifications**: Avoid editing or re-saving stego images with image editors, as this may corrupt the hidden data.

⚠️ **Capacity Limits**: Messages that exceed image capacity will be rejected. Use larger images or lower robustness levels for longer messages.

## 📚 Documentation & Resources

- **GitHub Repository**: [PixelNur Phase 2](https://github.com/yourusername/pixelnur-phase2)
- **Research Paper**: [Coming Soon]
- **API Documentation**: [Coming Soon]
- **Issue Tracker**: [Report Bugs](https://github.com/yourusername/pixelnur-phase2/issues)

## 🎓 Academic Context

This project was developed as part of advanced steganography research at KIT's College of Engineering, Kolhapur. It builds upon classical steganography techniques by incorporating modern deep learning and robust signal processing methods.

### Research Team
- Aman Qureshi (2223000503)
- Rushikesh Randive (2223000930)
- Ankita Patil (2223000302)
- Madhura Patil (2223000060)

**Institution**: KIT's College of Engineering, Kolhapur  
**Year**: 2024-2025

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for providing free hosting infrastructure
- PyTorch and OpenCV communities for excellent tools
- Research community for steganography and watermarking techniques

---

**Note**: This application runs on Hugging Face Spaces with automatic GPU acceleration when available. Processing times may vary based on server load and hardware availability.
