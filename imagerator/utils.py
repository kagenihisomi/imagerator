import pickle

import numpy as np
from PIL import Image, ImageOps
from transformers import pipeline


def mask_colorer(my_mask: np.ndarray) -> Image.Image:
    """Colorizes a mask for display."""
    colorized_mask = ImageOps.colorize(
        ImageOps.grayscale(Image.fromarray(my_mask)),
        black="black",
        white=tuple(np.random.randint(0, 256, size=3)),
    )
    colorized_mask = colorized_mask.convert("RGBA")
    colorized_mask.putalpha(120)
    return colorized_mask


def sepia(input_img):
    sepia_filter = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


def generate_image_sections(
    image_np: np.ndarray,
) -> tuple[Image.Image, list[tuple[Image.Image, str]]]:
    generator = pipeline(
        "mask-generation",
        model="facebook/sam-vit-base",
        device="cpu",
        points_per_batch=512,
    )
    image = Image.fromarray(image_np)
    outputs = generator(image)

    masks = outputs["masks"]

    sections = [(mask, f"{i}") for i, mask in enumerate(masks)]

    return (image, sections)


def get_example_sections(_):
    sections = pickle.load(open("outputs.pkl", "rb"))
    return sections


def image_from_mask(img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Generate a new image from the masked areas of the original image.
    :param img: _description_
    :type img: Image
    :param mask: _description_
    :type mask: Image
    :return: _description_
    :rtype: Image
    """

    # Create a boolean mask that represents the masked areas of the image
    mask_array = np.array(mask)
    image_array = np.array(img)

    mask_bool = mask_array[:, :, 3] > 0

    # Select only the masked areas of the image
    img_masked_np = np.zeros_like(image_array)
    img_masked_np[mask_bool] = image_array[mask_bool]
    # Convert the masked image back to a PIL image
    img_masked = Image.fromarray(img_masked_np)
    x0, y0, x1, y1 = Image.fromarray(mask_bool.astype(np.uint8)).getbbox()

    # Display the original image
    # img_trimmed = img.crop((x0, y0, x1, y1))

    # Display the masked image
    img_trimmed = img_masked.crop((x0, y0, x1, y1))

    return img_trimmed
