"""
Metrics Service for PixelNur Phase 2

This module provides quality metrics calculation for steganography operations:
- PSNR (Peak Signal-to-Noise Ratio): Measures imperceptibility (target: 42-48 dB)
- SSIM (Structural Similarity Index): Measures perceptual quality (target: ≥0.91)

Supports both RGB and grayscale images with optimized NumPy vectorization.

Requirements: 8.2, 15.2
"""

import numpy as np
from typing import Tuple, Union
from numpy.typing import NDArray


class MetricsService:
    """Service for calculating image quality metrics between cover and stego images."""

    @staticmethod
    def calculate_psnr(
        cover_image: NDArray[np.uint8], stego_image: NDArray[np.uint8], max_pixel_value: int = 255
    ) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between cover and stego images.

        PSNR measures the ratio between the maximum possible signal power and the power
        of corrupting noise. Higher PSNR indicates better quality (less distortion).

        Formula: PSNR = 10 * log10(MAX^2 / MSE)
        where MAX is the maximum possible pixel value (255 for 8-bit images)
        and MSE is the Mean Squared Error.

        Args:
            cover_image: Original cover image as numpy array (H, W) or (H, W, C)
            stego_image: Stego image with embedded data as numpy array (H, W) or (H, W, C)
            max_pixel_value: Maximum pixel value (default: 255 for 8-bit images)

        Returns:
            PSNR value in decibels (dB). Target range: 42-48 dB

        Raises:
            ValueError: If images have different shapes or are empty
            ValueError: If MSE is zero (images are identical)
        """
        # Validate inputs
        if cover_image.shape != stego_image.shape:
            raise ValueError(
                f"Image shapes must match. Got cover: {cover_image.shape}, "
                f"stego: {stego_image.shape}"
            )

        if cover_image.size == 0:
            raise ValueError("Images cannot be empty")

        # Convert to float for accurate calculation
        cover_float = cover_image.astype(np.float64)
        stego_float = stego_image.astype(np.float64)

        # Calculate Mean Squared Error using vectorized operations
        mse = np.mean((cover_float - stego_float) ** 2)

        # Handle identical images (MSE = 0)
        if mse == 0:
            return float("inf")

        # Calculate PSNR
        psnr = 10.0 * np.log10((max_pixel_value**2) / mse)

        return float(psnr)

    @staticmethod
    def calculate_ssim(
        cover_image: NDArray[np.uint8],
        stego_image: NDArray[np.uint8],
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        max_pixel_value: int = 255,
    ) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between cover and stego images.

        SSIM measures perceptual similarity by comparing luminance, contrast, and structure.
        Uses a sliding window approach for local similarity assessment.

        Formula: SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
        where μ is mean, σ is variance, σxy is covariance, and C1, C2 are stability constants.

        Args:
            cover_image: Original cover image as numpy array (H, W) or (H, W, C)
            stego_image: Stego image with embedded data as numpy array (H, W) or (H, W, C)
            window_size: Size of the sliding window (default: 11, must be odd)
            k1: Constant for luminance stability (default: 0.01)
            k2: Constant for contrast stability (default: 0.03)
            max_pixel_value: Maximum pixel value (default: 255 for 8-bit images)

        Returns:
            SSIM value in range [-1, 1], where 1 means identical images. Target: ≥0.91

        Raises:
            ValueError: If images have different shapes or are empty
            ValueError: If window_size is even or too large
        """
        # Validate inputs
        if cover_image.shape != stego_image.shape:
            raise ValueError(
                f"Image shapes must match. Got cover: {cover_image.shape}, "
                f"stego: {stego_image.shape}"
            )

        if cover_image.size == 0:
            raise ValueError("Images cannot be empty")

        if window_size % 2 == 0:
            raise ValueError(f"Window size must be odd. Got: {window_size}")

        if window_size > min(cover_image.shape[:2]):
            raise ValueError(
                f"Window size ({window_size}) cannot be larger than "
                f"image dimensions ({cover_image.shape[:2]})"
            )

        # Handle multi-channel images (RGB) by processing each channel separately
        if len(cover_image.shape) == 3:
            ssim_values = []
            for channel in range(cover_image.shape[2]):
                ssim_channel = MetricsService._calculate_ssim_single_channel(
                    cover_image[:, :, channel],
                    stego_image[:, :, channel],
                    window_size,
                    k1,
                    k2,
                    max_pixel_value,
                )
                ssim_values.append(ssim_channel)
            # Return mean SSIM across all channels
            return float(np.mean(ssim_values))
        else:
            # Grayscale image
            return MetricsService._calculate_ssim_single_channel(
                cover_image, stego_image, window_size, k1, k2, max_pixel_value
            )

    @staticmethod
    def _calculate_ssim_single_channel(
        cover_channel: NDArray[np.uint8],
        stego_channel: NDArray[np.uint8],
        window_size: int,
        k1: float,
        k2: float,
        max_pixel_value: int,
    ) -> float:
        """
        Calculate SSIM for a single channel using sliding window approach.

        This is an internal method that implements the core SSIM algorithm
        for grayscale images or individual color channels.

        Args:
            cover_channel: Cover image channel (H, W)
            stego_channel: Stego image channel (H, W)
            window_size: Size of the sliding window
            k1: Luminance stability constant
            k2: Contrast stability constant
            max_pixel_value: Maximum pixel value

        Returns:
            SSIM value for the channel
        """
        # Convert to float for accurate calculation
        cover_float = cover_channel.astype(np.float64)
        stego_float = stego_channel.astype(np.float64)

        # Calculate stability constants
        c1 = (k1 * max_pixel_value) ** 2
        c2 = (k2 * max_pixel_value) ** 2

        # Create Gaussian window for weighted averaging
        window = MetricsService._create_gaussian_window(window_size)

        # Calculate local means using convolution
        mu_cover = MetricsService._apply_window(cover_float, window)
        mu_stego = MetricsService._apply_window(stego_float, window)

        # Calculate local variances and covariance
        mu_cover_sq = mu_cover**2
        mu_stego_sq = mu_stego**2
        mu_cover_stego = mu_cover * mu_stego

        sigma_cover_sq = MetricsService._apply_window(cover_float**2, window) - mu_cover_sq
        sigma_stego_sq = MetricsService._apply_window(stego_float**2, window) - mu_stego_sq
        sigma_cover_stego = (
            MetricsService._apply_window(cover_float * stego_float, window) - mu_cover_stego
        )

        # Calculate SSIM map
        numerator = (2 * mu_cover_stego + c1) * (2 * sigma_cover_stego + c2)
        denominator = (mu_cover_sq + mu_stego_sq + c1) * (sigma_cover_sq + sigma_stego_sq + c2)

        ssim_map = numerator / denominator

        # Return mean SSIM across the entire image
        return float(np.mean(ssim_map))

    @staticmethod
    def _create_gaussian_window(window_size: int, sigma: float = 1.5) -> NDArray[np.float64]:
        """
        Create a 2D Gaussian window for SSIM calculation.

        Args:
            window_size: Size of the window (must be odd)
            sigma: Standard deviation of the Gaussian distribution

        Returns:
            Normalized 2D Gaussian window
        """
        # Create 1D Gaussian kernel
        ax = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
        gauss_1d = np.exp(-(ax**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Create 2D Gaussian kernel using outer product
        gauss_2d = np.outer(gauss_1d, gauss_1d)

        return gauss_2d

    @staticmethod
    def _apply_window(
        image: NDArray[np.float64], window: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Apply a sliding window to an image using convolution.

        This method uses NumPy's correlate function for efficient sliding window operations.

        Args:
            image: Input image (H, W)
            window: Window kernel (K, K)

        Returns:
            Convolved image with same dimensions as input
        """
        from scipy.ndimage import convolve

        # Use scipy's convolve for efficient sliding window
        # mode='reflect' handles borders by reflecting the image
        return convolve(image, window, mode="reflect")

    @staticmethod
    def calculate_metrics(
        cover_image: NDArray[np.uint8], stego_image: NDArray[np.uint8]
    ) -> Tuple[float, float]:
        """
        Calculate both PSNR and SSIM metrics in a single call.

        This is a convenience method for calculating both metrics at once,
        which is more efficient than calling them separately.

        Args:
            cover_image: Original cover image as numpy array (H, W) or (H, W, C)
            stego_image: Stego image with embedded data as numpy array (H, W) or (H, W, C)

        Returns:
            Tuple of (psnr, ssim) values

        Raises:
            ValueError: If images have different shapes or are empty
        """
        psnr = MetricsService.calculate_psnr(cover_image, stego_image)
        ssim = MetricsService.calculate_ssim(cover_image, stego_image)

        return psnr, ssim

    @staticmethod
    def estimate_capacity(
        cnn_mask: NDArray[np.uint8],
        robustness_level: str = "none",
        header_overhead_pixels: int = 56
    ) -> int:
        """
        Estimate embedding capacity in bytes based on CNN mask and robustness level.

        This method calculates the maximum message size that can be embedded in an image
        by counting usable coefficients from the CNN mask, accounting for header overhead,
        and applying ECC overhead based on the robustness level.

        Capacity Calculation:
        1. Count usable pixels from CNN mask (binary mask with 1s indicating embeddable locations)
        2. Subtract header overhead (56 pixels reserved for version, robustness level, length)
        3. Convert pixels to bits (1 bit per pixel using LSB embedding)
        4. Apply ECC overhead based on robustness level
        5. Convert bits to bytes

        ECC Overhead by Robustness Level:
        - none: 0% overhead (no error correction)
        - low: 14.2% overhead (RS(255, 223) - corrects 16 byte errors)
        - medium: 33.5% overhead (RS(255, 191) - corrects 32 byte errors)
        - high: 300% overhead (RS(255, 127) + 3x replication - corrects 64 byte errors)

        Args:
            cnn_mask: Binary embedding mask from CNN Module (H/2, W/2) with values 0 or 1
            robustness_level: Robustness level ('none', 'low', 'medium', 'high')
            header_overhead_pixels: Number of pixels reserved for header (default: 56)

        Returns:
            Maximum embedding capacity in bytes

        Raises:
            ValueError: If cnn_mask is invalid or robustness_level is not recognized
            ValueError: If capacity is negative (image too small)

        Requirements:
        - 19.1: Calculate maximum embedding capacity in bytes
        - 19.2: Account for ECC overhead when robustness is enabled
        - 19.3: Account for redundant embedding when high robustness is selected
        - 19.4: Provide capacity estimates for different robustness levels
        - 24.6: Embedding capacity shall not exceed calculated maximum
        """
        # Validate inputs
        if cnn_mask is None or cnn_mask.size == 0:
            raise ValueError("CNN mask cannot be empty or None")

        if len(cnn_mask.shape) != 2:
            raise ValueError(
                f"CNN mask must be 2D array (H/2, W/2), got shape: {cnn_mask.shape}"
            )

        valid_robustness_levels = ["none", "low", "medium", "high"]
        if robustness_level not in valid_robustness_levels:
            raise ValueError(
                f"Invalid robustness level: {robustness_level}. "
                f"Must be one of {valid_robustness_levels}"
            )

        # Count usable pixels from CNN mask (pixels with value 1)
        # Treat any non-zero value as usable (convert to binary)
        binary_mask = (cnn_mask > 0).astype(np.uint8)
        usable_pixels = int(np.sum(binary_mask))

        # Subtract header overhead
        available_pixels = usable_pixels - header_overhead_pixels

        if available_pixels <= 0:
            raise ValueError(
                f"Image too small for embedding. Usable pixels: {usable_pixels}, "
                f"header overhead: {header_overhead_pixels}. "
                f"Need at least {header_overhead_pixels + 1} usable pixels."
            )

        # Convert pixels to bits (1 bit per pixel using LSB embedding)
        available_bits = available_pixels

        # Apply ECC overhead based on robustness level
        # ECC reduces capacity because parity bytes are added
        if robustness_level == "none":
            # No overhead
            capacity_bits = available_bits
        elif robustness_level == "low":
            # RS(255, 223): 223 data bytes + 32 parity bytes = 255 total
            # Overhead: 32/223 = 14.2%
            # Capacity factor: 223/255 = 0.8745
            capacity_bits = int(available_bits * 0.8745)
        elif robustness_level == "medium":
            # RS(255, 191): 191 data bytes + 64 parity bytes = 255 total
            # Overhead: 64/191 = 33.5%
            # Capacity factor: 191/255 = 0.7490
            capacity_bits = int(available_bits * 0.7490)
        elif robustness_level == "high":
            # RS(255, 127): 127 data bytes + 128 parity bytes = 255 total
            # Plus 3x replication (same data embedded 3 times)
            # Overhead: 128/127 = 100% ECC + 200% replication = 300% total
            # Capacity factor: 127/255 / 3 = 0.1660
            capacity_bits = int(available_bits * 0.1660)

        # Convert bits to bytes
        capacity_bytes = capacity_bits // 8

        return capacity_bytes

    @staticmethod
    def estimate_capacity_all_levels(
        cnn_mask: NDArray[np.uint8],
        header_overhead_pixels: int = 48
    ) -> dict:
        """
        Estimate embedding capacity for all robustness levels.

        This is a convenience method that calculates capacity for all four
        robustness levels (none, low, medium, high) in a single call.

        Args:
            cnn_mask: Binary embedding mask from CNN Module (H/2, W/2)
            header_overhead_pixels: Number of pixels reserved for header (default: 48)

        Returns:
            Dictionary with capacity in bytes for each robustness level:
            {
                'none': capacity_bytes,
                'low': capacity_bytes,
                'medium': capacity_bytes,
                'high': capacity_bytes
            }

        Raises:
            ValueError: If cnn_mask is invalid

        Requirements: 19.4 (provide capacity estimates for different robustness levels)
        """
        capacities = {}
        for level in ["none", "low", "medium", "high"]:
            try:
                capacities[level] = MetricsService.estimate_capacity(
                    cnn_mask, level, header_overhead_pixels
                )
            except ValueError as e:
                # If capacity calculation fails for any level, propagate the error
                raise ValueError(f"Failed to calculate capacity for level '{level}': {str(e)}")

        return capacities
