import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.fftpack import dct, idct


class DCTWatermark:
    def __init__(self, strength=25):
        self.strength = strength  # Watermark embedding strength

    def embed(self, host_img, watermark):
        """Embed watermark into host image"""
        # Convert to YUV color space, process only Y channel (luminance)
        yuv = cv2.cvtColor(host_img, cv2.COLOR_BGR2YUV)
        y_channel = np.float32(yuv[:, :, 0])

        # Split image into 8x8 blocks
        blocks = self.split_blocks(y_channel)

        # Ensure watermark is a binary image
        if watermark.dtype != np.uint8 or watermark.max() > 1:
            watermark = (watermark > 128).astype(np.uint8)

        # Embed watermark
        watermarked_blocks = []
        wm_idx = 0
        for block in blocks:
            # Apply DCT transform
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

            # Embed watermark in DCT coefficients (using mid-frequency)
            if wm_idx < watermark.size:
                dct_block[3, 4] += self.strength * (2 * watermark.flat[wm_idx] - 1)
                wm_idx += 1

            # Inverse DCT transform
            idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
            watermarked_blocks.append(idct_block)

        # Reconstruct Y channel
        watermarked_y = self.merge_blocks(watermarked_blocks, y_channel.shape)

        # Merge channels and convert back to BGR
        yuv[:, :, 0] = np.clip(watermarked_y, 0, 255)
        return cv2.cvtColor(np.uint8(yuv), cv2.COLOR_YUV2BGR)

    def extract(self, watermarked_img, watermark_shape):
        """Extract watermark from image"""
        yuv = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YUV)
        y_channel = np.float32(yuv[:, :, 0])

        # Split image into 8x8 blocks
        blocks = self.split_blocks(y_channel)

        # Extract watermark
        watermark = np.zeros(watermark_shape, dtype=np.uint8)
        wm_idx = 0

        for block in blocks:
            if wm_idx >= watermark.size:
                break

            # Apply DCT transform
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

            # Extract watermark bit
            watermark.flat[wm_idx] = 1 if dct_block[3, 4] > 0 else 0
            wm_idx += 1

        return watermark * 255  # Convert to 0-255 range

    def split_blocks(self, img, block_size=8):
        """Split image into 8x8 blocks"""
        h, w = img.shape
        blocks = []

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                blocks.append(img[y:y + block_size, x:x + block_size])

        return blocks

    def merge_blocks(self, blocks, shape, block_size=8):
        """Merge blocks into complete image"""
        h, w = shape
        merged = np.zeros(shape, dtype=np.float32)
        block_idx = 0

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                merged[y:y + block_size, x:x + block_size] = blocks[block_idx]
                block_idx += 1

        return merged


def create_text_watermark(text, size=(64, 64)):
    """Create text watermark image"""
    watermark = np.zeros(size, dtype=np.uint8)
    # Adjust font size and position to fit watermark dimensions
    font_scale = min(size) / 64  # Dynamically adjust font based on watermark size
    text_pos = (int(size[1] * 0.1), int(size[0] * 0.6))  # Text position
    cv2.putText(watermark, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, 2)
    return watermark


def robustness_test(original_img, watermarked_img, watermark, watermarker):
    """Perform robustness tests"""
    attacks = {
        'Rotate 15°': lambda img: rotate_image(img, 15),
        'Translate': lambda img: shift_image(img, 30, 30),
        'Crop 20%': lambda img: crop_image(img, 0.2),
        'Contrast Adjust': lambda img: adjust_contrast(img, 1.5),
        'Brightness Adjust': lambda img: adjust_brightness(img, 40),
        'Gaussian Noise': lambda img: add_gaussian_noise(img, 0.02),
        'JPEG Compression': lambda img: jpeg_compression(img, 80),
        'Scale 80%': lambda img: scale_image(img, 0.8)
    }

    results = []
    # Adjust subplot layout to 4 rows x 3 columns to fit all images
    plt.figure(figsize=(18, 20))

    # Display original and watermarked images
    plt.subplot(4, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(4, 3, 2)
    plt.imshow(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB))
    plt.title('Watermarked Image')
    plt.axis('off')

    plt.subplot(4, 3, 3)
    plt.imshow(watermark, cmap='gray')
    plt.title('Original Watermark')
    plt.axis('off')

    # Test various attacks
    for i, (attack_name, attack_func) in enumerate(attacks.items()):
        attacked_img = attack_func(watermarked_img.copy())
        extracted_wm = watermarker.extract(attacked_img, watermark.shape)

        # Calculate similarity
        similarity = calculate_similarity(watermark, extracted_wm)
        results.append((attack_name, similarity))

        # Visualize results
        plt.subplot(4, 3, i + 4)
        plt.imshow(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB))
        plt.title(f"{attack_name}\nSimilarity: {similarity:.2%}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('robustness_test_results.jpg')
    print("Robustness test results saved as 'robustness_test_results.jpg'")
    plt.show()
    return results


# ======== Image processing functions ========
def rotate_image(img, angle):
    """Rotate image"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))


def shift_image(img, dx, dy):
    """Translate image"""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(128, 128, 128))


def crop_image(img, ratio):
    """Crop image"""
    h, w = img.shape[:2]
    start_y, end_y = int(h * ratio / 2), int(h * (1 - ratio / 2))
    start_x, end_x = int(w * ratio / 2), int(w * (1 - ratio / 2))
    return img[start_y:end_y, start_x:end_x]


def adjust_contrast(img, factor):
    """Adjust image contrast"""
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def adjust_brightness(img, value):
    """Adjust image brightness"""
    return cv2.convertScaleAbs(img, alpha=1, beta=value)


def add_gaussian_noise(img, var):
    """Add Gaussian noise to image"""
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality):
    """Apply JPEG compression"""
    temp_path = 'temp_jpeg.jpg'
    cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    img = cv2.imread(temp_path)
    os.remove(temp_path)  # Clean up temporary file
    return img


def scale_image(img, factor):
    """Scale image"""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)


def calculate_similarity(original, extracted):
    """Calculate watermark similarity"""
    # Binarize images
    orig_bin = (original > 128).astype(np.uint8)
    extr_bin = (extracted > 128).astype(np.uint8)

    # Calculate ratio of matching pixels
    match = np.sum(orig_bin == extr_bin)
    total = orig_bin.size
    return match / total


# ======== Main function ========
def main():
    # Set image path to /kaggle/input/original/original.jpg
    img_path = '/kaggle/input/original/original.jpg'
    print(f"Using image path: {img_path}")

    # 1. Read original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Error: Could not read image at {img_path}, please check the path")
        print("Creating sample image as fallback...")
        # Create sample image
        original_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(original_img, 'Sample Image', (150, 256),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img_path = 'generated_sample.jpg'
        cv2.imwrite(img_path, original_img)

    print(f"Successfully loaded image: {img_path} (size: {original_img.shape[1]}x{original_img.shape[0]})")

    # 2. Create watermark (content: SDU2025)
    watermark_text = "SDU2025"
    watermark_size = (64, 64)
    watermark = create_text_watermark(watermark_text, watermark_size)
    cv2.imwrite('watermark.png', watermark)
    print(
        f"Created watermark image 'watermark.png' (text: {watermark_text}, size: {watermark_size[0]}x{watermark_size[1]})")

    # 3. Embed watermark
    watermarker = DCTWatermark(strength=30)
    watermarked_img = watermarker.embed(original_img, watermark)
    cv2.imwrite('watermarked.jpg', watermarked_img)
    print("Generated watermarked image 'watermarked.jpg'")

    # 4. Extract watermark (no attack)
    extracted_wm = watermarker.extract(watermarked_img, watermark.shape)
    cv2.imwrite('extracted_watermark.png', extracted_wm)
    print("Extracted original watermark 'extracted_watermark.png'")

    # 5. Calculate similarity
    similarity = calculate_similarity(watermark, extracted_wm)
    print(f"Original watermark similarity: {similarity:.2%}")

    # 6. Perform robustness tests
    print("Starting robustness tests...")
    test_results = robustness_test(original_img, watermarked_img, watermark, watermarker)

    # 7. Print test results
    print("\nRobustness test results:")
    print("-" * 40)
    print(f"{'Attack Type':<18} | {'Similarity'}")
    print("-" * 40)
    for name, similarity in test_results:
        print(f"{name:<18} | {similarity:.2%}")


if __name__ == "__main__":
    main()
