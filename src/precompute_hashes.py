import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

# Try to use OpenCV's optimized pHash.
USE_CV2_IMG_HASH = hasattr(cv2, "img_hash")
if not USE_CV2_IMG_HASH:
    # Fallback to imagehash (requires Pillow and imagehash).
    print(
        "Warning: cv2.img_hash not found. Falling back to imagehash. "
        "For better performance install opencv-contrib-python."
    )
    import imagehash
    from PIL import Image


class PHashComputer:
    def __init__(self, image_paths, pool_workers=None):
        self.image_paths = image_paths
        self.pool_workers = pool_workers
        self.phash_array = None  # Raw pHashes (each a 1D numpy array)
        self.bit_hashes = None  # Vectorized (binary) representation of hashes
        self.filenames = None

    @staticmethod
    def init_phash_worker():
        """Initializer for process pool workers.
        Only needed if using cv2's img_hash."""
        if USE_CV2_IMG_HASH:
            global phash_obj
            phash_obj = cv2.img_hash.PHash_create()

    @staticmethod
    def compute_phash(image_path):
        """Compute the pHash for a single image.
        Uses cv2.img_hash if available; otherwise falls back to imagehash."""
        if USE_CV2_IMG_HASH:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            # Optionally, you might resize here—but often cv2’s implementation performs its own normalization.
            hash_val = phash_obj.compute(image)
            # Return a flat numpy array of type uint8.
            return hash_val.flatten()
        else:
            try:
                image = Image.open(image_path)
            except Exception:
                return None
            # Compute pHash using imagehash (this returns an ImageHash object).
            hash_obj = imagehash.phash(image)
            # Convert the boolean hash array to a flattened uint8 array (0s and 1s).
            return np.array(hash_obj.hash.flatten(), dtype=np.uint8)

    def compute_all_phashes(self):
        """Compute and store the pHashes for all image paths with a progress bar."""
        with ProcessPoolExecutor(
            initializer=self.init_phash_worker, max_workers=self.pool_workers
        ) as executor:
            phashes = list(
                tqdm(
                    executor.map(self.compute_phash, self.image_paths),
                    total=len(self.image_paths),
                    desc="Computing pHashes",
                )
            )
        valid = [(p, h) for p, h in zip(self.image_paths, phashes) if h is not None]
        if not valid:
            raise ValueError("No valid images processed.")
        self.filenames, hash_list = zip(*valid)
        self.phash_array = np.stack(hash_list, axis=0)

    def vectorize_hashes(self):
        """Convert the pHashes into a binary bit representation for fast Hamming comparisons."""
        if self.phash_array is None:
            raise ValueError("pHash array is empty; run compute_all_phashes first.")
        if USE_CV2_IMG_HASH:
            self.bit_hashes = np.unpackbits(self.phash_array.astype(np.uint8), axis=1)
        else:
            self.bit_hashes = self.phash_array.astype(np.uint8)

    def compute_hamming_distance_matrix(self):
        """Compute and return the full NxN Hamming distance matrix using vectorized NumPy operations."""
        if self.bit_hashes is None:
            self.vectorize_hashes()
        sums = self.bit_hashes.sum(axis=1, keepdims=True)
        dot_products = self.bit_hashes @ self.bit_hashes.T
        hamming_matrix = sums + sums.T - 2 * dot_products
        return hamming_matrix


def fast_ssim(img1, img2, kernel_size=7):
    """
    Compute a fast SSIM between two grayscale images using box filters.
    This version uses OpenCV’s highly optimized boxFilter function to compute local
    means, variances, and covariance. The kernel_size controls the window size.

    Constants are based on the dynamic range of the images (assumed to be 0-255).
    """
    # Convert images to float32 for precision.
    I1 = img1.astype(np.float32)
    I2 = img2.astype(np.float32)
    # Constants (these assume L=255)
    C1 = 6.5025
    C2 = 58.5225
    # Compute local means via box filter.
    mu1 = cv2.boxFilter(I1, ddepth=-1, ksize=(kernel_size, kernel_size))
    mu2 = cv2.boxFilter(I2, ddepth=-1, ksize=(kernel_size, kernel_size))
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    # Compute local variances and covariance.
    sigma1_sq = (
        cv2.boxFilter(I1 * I1, ddepth=-1, ksize=(kernel_size, kernel_size)) - mu1_sq
    )
    sigma2_sq = (
        cv2.boxFilter(I2 * I2, ddepth=-1, ksize=(kernel_size, kernel_size)) - mu2_sq
    )
    sigma12 = (
        cv2.boxFilter(I1 * I2, ddepth=-1, ksize=(kernel_size, kernel_size)) - mu1_mu2
    )
    # Compute SSIM map.
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


class SSIMComputer:
    def __init__(self, filenames, pool_workers=None, target_size=None, kernel_size=7):
        """
        :param filenames: List of image file paths.
        :param pool_workers: Number of threads for SSIM comparisons.
        :param target_size: Tuple (width, height) to resize images (reduces computation & memory).
        :param kernel_size: Window size for the box filter in fast_ssim.
        """
        self.filenames = filenames
        self.pool_workers = pool_workers
        self.target_size = target_size
        self.kernel_size = kernel_size
        self.gray_images = {}  # Cache for preloaded images.
        self.preload_images()

    def load_and_convert(self, image_path):
        """Load an image, optionally resize it, and convert it to grayscale."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        if self.target_size is not None:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def preload_images(self):
        """Preload and convert all images once (with progress bar)."""
        for path in tqdm(self.filenames, desc="Preloading images"):
            gray = self.load_and_convert(path)
            if gray is not None:
                self.gray_images[path] = gray

    def compute_ssim_for_pair(self, pair):
        """Compute SSIM for a given image pair using fast_ssim."""
        i, j = pair
        path1, path2 = self.filenames[i], self.filenames[j]
        img1 = self.gray_images.get(path1)
        img2 = self.gray_images.get(path2)
        if img1 is None or img2 is None:
            return (pair, 0.0)
        score = fast_ssim(img1, img2, kernel_size=self.kernel_size)
        return (pair, 1 - score)

    def compute_full_ssim_matrix(
        self, memmap_filename="assets/results/ssim_matrix.dat"
    ):
        """
        Compute and return the full NxN SSIM matrix using a memory-mapped file.
        This implementation processes the matrix row-by-row using threads with the fast_ssim function.
        """
        n = len(self.filenames)
        ssim_matrix = np.memmap(
            memmap_filename, dtype="float32", mode="w+", shape=(n, n)
        )
        ssim_matrix[:] = 0.0
        for i in range(n):
            ssim_matrix[i, i] = 1.0
        for i in tqdm(range(n), desc="Computing full SSIM matrix"):

            def compute_ssim_j(j):
                return self.compute_ssim_for_pair((i, j))

            with ThreadPoolExecutor(max_workers=self.pool_workers) as executor:
                results = list(executor.map(compute_ssim_j, range(i + 1, n)))
            for j, score in results:
                ssim_matrix[i, j] = score
                ssim_matrix[j, i] = score
            ssim_matrix.flush()
        return ssim_matrix


class DistanceMatrixComputer:
    def __init__(
        self,
        image_paths,
        phash_workers=None,
        ssim_workers=None,
        ssim_target_size=None,
        ssim_kernel_size=7,
    ):
        self.image_paths = image_paths
        self.phash_workers = phash_workers
        self.ssim_workers = ssim_workers
        self.ssim_target_size = ssim_target_size
        self.ssim_kernel_size = ssim_kernel_size
        self.phash_computer = PHashComputer(image_paths, pool_workers=phash_workers)
        self.ssim_computer = None

    def compute_full_distance_matrices(self):
        # Compute pHash-based Hamming distance matrix.
        print("Computing pHashes and full Hamming distance matrix...")
        self.phash_computer.compute_all_phashes()
        self.phash_computer.vectorize_hashes()
        hamming_matrix = self.phash_computer.compute_hamming_distance_matrix()
        np.save("assets/results/phash_matrix.npy", hamming_matrix)

        # Compute full SSIM matrix using the fast SSIM implementation.
        print("Preloading images for SSIM computation (with optional resizing)...")
        self.ssim_computer = SSIMComputer(
            self.phash_computer.filenames,
            pool_workers=self.ssim_workers,
            target_size=self.ssim_target_size,
            kernel_size=self.ssim_kernel_size,
        )
        print("Computing full SSIM matrix (row-by-row memmapped computation)...")
        ssim_matrix = self.ssim_computer.compute_full_ssim_matrix()
        np.save("assets/results/ssim_matrix.npy", ssim_matrix)


if __name__ == "__main__":
    image_paths = glob.glob("data/fitzpatrick17k/images/*.jpg")
    if not image_paths:
        print("No images found. Check your image path and file extension.")
        exit()

    finder = DistanceMatrixComputer(
        image_paths,
        phash_workers=8,
        ssim_workers=8,
        ssim_target_size=(64, 64),
        ssim_kernel_size=7,
    )
    finder.compute_full_distance_matrices()
