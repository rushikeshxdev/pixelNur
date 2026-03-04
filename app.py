"""
PixelNur Phase 2 - Gradio Interface for Hugging Face Spaces

This module provides a web interface for the PixelNur steganography system
using Gradio. It exposes embedding and extraction functionality through
an intuitive tabbed interface.

Requirements: 2.1, 2.2, 3.5, 3.6
"""

import gradio as gr
import numpy as np
import cv2
import torch
import logging
import sys
from typing import Tuple, Optional

# Import Phase 2 modules
from src.pixelnur import PixelNur, PixelNurError, InsufficientCapacityError
from src.extraction_engine import ExtractionEngine, ExtractionError
from src.lwt_transform import LWTTransform
from src.encryption_service import EncryptionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('pixelnur-gradio')

# Constants
ROBUSTNESS_LEVELS = {
    "None (0) - Best quality, no attack resistance": "none",
    "Low (1) - Light ECC, minor attack resistance": "low",
    "Medium (2) - Moderate ECC + replication": "medium",
    "High (3) - Maximum protection, lower capacity": "high"
}

MIN_PASSWORD_LENGTH = 16
MIN_IMAGE_SIZE = 256
MAX_IMAGE_SIZE = 2048  # For Gradio deployment

# Initialize PixelNur system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Initializing PixelNur on device: {device}")

try:
    pixelnur = PixelNur(device=device)
    extraction_engine = ExtractionEngine()
    lwt_transform = LWTTransform()
    encryption_service = EncryptionService()
    logger.info("PixelNur system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PixelNur system: {str(e)}", exc_info=True)
    raise


def _validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate image format and dimensions.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None or image.size == 0:
        return False, "No image provided. Please upload a valid image."
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False, "Invalid image format. Please upload a color image (RGB)."
    
    h, w = image.shape[:2]
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        return False, f"Image too small: {w}×{h} pixels. Minimum: {MIN_IMAGE_SIZE}×{MIN_IMAGE_SIZE} pixels."
    
    if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE:
        return False, f"Image too large: {w}×{h} pixels. Maximum: {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} pixels."
    
    return True, ""


def _validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password meets minimum requirements.
    
    Args:
        password: Password string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required."
    
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password too short: {len(password)} characters. Minimum: {MIN_PASSWORD_LENGTH} characters."
    
    return True, ""


def _parse_robustness_level(level_str: str) -> str:
    """
    Convert UI string to robustness level value.
    
    Args:
        level_str: Robustness level string from UI dropdown
        
    Returns:
        Robustness level value ('none', 'low', 'medium', 'high')
    """
    return ROBUSTNESS_LEVELS.get(level_str, "none")


def embed_interface(
    cover_image: np.ndarray,
    message: str,
    password: str,
    robustness_level: str
) -> Tuple[Optional[np.ndarray], str, str, str]:
    """
    Gradio interface function for embedding.
    
    Args:
        cover_image: Cover image from Gradio Image component (RGB)
        message: Secret message text
        password: Encryption password
        robustness_level: Robustness level string from dropdown
    
    Returns:
        Tuple of (stego_image, psnr_text, ssim_text, capacity_text)
    """
    try:
        # Validate inputs
        valid, error = _validate_image(cover_image)
        if not valid:
            return None, f"❌ {error}", "", ""
        
        if not message or not message.strip():
            return None, "❌ Message cannot be empty.", "", ""
        
        valid, error = _validate_password(password)
        if not valid:
            return None, f"❌ {error}", "", ""
        
        # Convert RGB to BGR for OpenCV
        cover_bgr = cv2.cvtColor(cover_image, cv2.COLOR_RGB2BGR)
        
        # Parse robustness level
        robustness_value = _parse_robustness_level(robustness_level)
        
        logger.info(
            f"Embed request: image_shape={cover_image.shape}, "
            f"message_length={len(message)}, robustness_level={robustness_value}"
        )
        
        # Embed message
        stego_bgr, metrics = pixelnur.embed_message(
            cover_image=cover_bgr,
            message=message,
            encryption_key=password,
            robustness_level=robustness_value,
            check_capacity=True
        )
        
        # Convert BGR to RGB for Gradio
        stego_rgb = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2RGB)
        
        # Format outputs
        psnr_text = f"✅ PSNR: {metrics['psnr']:.2f} dB"
        ssim_text = f"✅ SSIM: {metrics['ssim']:.4f}"
        capacity_text = (
            f"✅ Capacity: {metrics['capacity_used_bytes']} / "
            f"{metrics['capacity_available_bytes']} bytes "
            f"({metrics['capacity_utilization']:.1f}% used)"
        )
        
        logger.info(
            f"Embed success: psnr={metrics['psnr']:.2f}, "
            f"ssim={metrics['ssim']:.4f}"
        )
        
        return stego_rgb, psnr_text, ssim_text, capacity_text
        
    except InsufficientCapacityError as e:
        logger.warning(f"Capacity error: {str(e)}")
        return None, f"❌ {str(e)}", "", ""
    
    except PixelNurError as e:
        logger.error(f"PixelNur error: {str(e)}")
        return None, f"❌ {str(e)}", "", ""
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return None, "❌ An unexpected error occurred. Please try again.", "", ""


def extract_interface(
    stego_image: np.ndarray,
    password: str
) -> str:
    """
    Gradio interface function for extraction.
    
    Args:
        stego_image: Stego image from Gradio Image component (RGB)
        password: Encryption password
    
    Returns:
        Extracted message text or error message
    """
    try:
        # Validate inputs
        valid, error = _validate_image(stego_image)
        if not valid:
            return f"❌ {error}"
        
        valid, error = _validate_password(password)
        if not valid:
            return f"❌ {error}"
        
        # Convert RGB to BGR for OpenCV
        stego_bgr = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Extract request: image_shape={stego_image.shape}")
        
        # Extract encrypted message
        encrypted_message = extraction_engine.extract(
            stego_image=stego_bgr,
            cover_image=None  # Blind extraction
        )
        
        # Decrypt message
        message_bytes = encryption_service.decrypt(
            encrypted_message,
            password
        )
        
        # Decode to string
        message = message_bytes.decode('utf-8')
        
        logger.info(f"Extract success: message_length={len(message)}")
        
        return f"✅ Extracted message:\n\n{message}"
        
    except ExtractionError as e:
        logger.warning(f"Extraction error: {str(e)}")
        return (
            f"❌ Extraction failed: {str(e)}\n\n"
            f"Please verify:\n"
            f"• Correct password\n"
            f"• Image hasn't been modified\n"
            f"• Image contains embedded data"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return "❌ An unexpected error occurred. Please try again."


def create_gradio_app() -> gr.Blocks:
    """
    Create and configure the Gradio interface.
    
    Returns:
        Gradio Blocks app ready to launch
    """
    
    with gr.Blocks(title="PixelNur Phase 2 - Steganography System") as app:
        gr.Markdown("""
        # 🔒 PixelNur Phase 2 - CNN-Based Steganography
        
        Hide secret messages in images using advanced steganography with CNN-based adaptive embedding.
        
        **Features:**
        - 🎯 High image quality (PSNR >40 dB, SSIM >0.99)
        - 🛡️ Configurable robustness against attacks
        - 🔐 Strong encryption (SHA-256 + XOR)
        - 🚀 GPU-accelerated processing
        """)
        
        with gr.Tabs():
            # Embed Tab
            with gr.Tab("📤 Embed Message"):
                gr.Markdown("### Hide a secret message in an image")
                
                with gr.Row():
                    with gr.Column():
                        embed_image = gr.Image(
                            label="Cover Image",
                            type="numpy",
                            height=300
                        )
                        embed_message = gr.Textbox(
                            label="Secret Message",
                            placeholder="Enter your secret message here...",
                            lines=3
                        )
                        embed_password = gr.Textbox(
                            label="Password (minimum 16 characters)",
                            placeholder="Enter a strong password...",
                            type="password"
                        )
                        embed_robustness = gr.Radio(
                            label="Robustness Level",
                            choices=list(ROBUSTNESS_LEVELS.keys()),
                            value=list(ROBUSTNESS_LEVELS.keys())[0]
                        )
                        embed_button = gr.Button("🔒 Embed Message", variant="primary")
                    
                    with gr.Column():
                        stego_image = gr.Image(
                            label="Stego Image (Download this!)",
                            type="numpy",
                            height=300
                        )
                        psnr_output = gr.Textbox(label="PSNR (Image Quality)", interactive=False)
                        ssim_output = gr.Textbox(label="SSIM (Structural Similarity)", interactive=False)
                        capacity_output = gr.Textbox(label="Capacity Usage", interactive=False)
                
                embed_button.click(
                    fn=embed_interface,
                    inputs=[embed_image, embed_message, embed_password, embed_robustness],
                    outputs=[stego_image, psnr_output, ssim_output, capacity_output]
                )
            
            # Extract Tab
            with gr.Tab("📥 Extract Message"):
                gr.Markdown("### Retrieve a hidden message from an image")
                
                with gr.Row():
                    with gr.Column():
                        extract_image = gr.Image(
                            label="Stego Image",
                            type="numpy",
                            height=300
                        )
                        extract_password = gr.Textbox(
                            label="Password",
                            placeholder="Enter the password used during embedding...",
                            type="password"
                        )
                        extract_button = gr.Button("🔓 Extract Message", variant="primary")
                    
                    with gr.Column():
                        extract_output = gr.Textbox(
                            label="Extracted Message",
                            placeholder="The hidden message will appear here...",
                            lines=10,
                            interactive=False
                        )
                
                extract_button.click(
                    fn=extract_interface,
                    inputs=[extract_image, extract_password],
                    outputs=extract_output
                )
        
        gr.Markdown("""
        ---
        ### 📖 Usage Instructions
        
        **Embedding:**
        1. Upload a cover image (PNG, JPEG, or BMP, minimum 256×256 pixels)
        2. Enter your secret message
        3. Create a strong password (minimum 16 characters)
        4. Select robustness level based on your needs
        5. Click "Embed Message" and download the stego image
        
        **Extraction:**
        1. Upload the stego image
        2. Enter the same password used during embedding
        3. Click "Extract Message" to reveal the hidden text
        
        **Robustness Levels:**
        - **None (0)**: Best quality, no attack resistance
        - **Low (1)**: Light error correction, resists minor JPEG compression
        - **Medium (2)**: Moderate protection with 3× replication
        - **High (3)**: Maximum protection with 5× replication, resists JPEG, resizing, noise
        
        ---
        
        **⚠️ Important Notes:**
        - Keep your password secure - it cannot be recovered if lost
        - Higher robustness levels reduce available capacity
        - Do not re-save or edit the stego image (use PNG for lossless storage)
        - This is a research prototype - use at your own risk
        
        **🔗 Links:**
        - [GitHub Repository](https://github.com/yourusername/pixelnur)
        - [Documentation](https://github.com/yourusername/pixelnur/blob/main/README.md)
        - [Report Issues](https://github.com/yourusername/pixelnur/issues)
        """)
    
    return app


# Launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
