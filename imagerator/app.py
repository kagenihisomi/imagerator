import gradio as gr
from PIL import Image

from imagerator.utils import generate_image_sections, image_from_mask


def select_section(evt: gr.SelectData, imgs: gr.AnnotatedImage) -> Image.Image:
    """
    Select a section from the image.

    :param evt: _description_
    :type evt: gr.SelectData
    :param imgs: _description_
    :type imgs: gr.AnnotatedImage
    :return: _description_
    :rtype: Image.Image
    """

    # Need to extract from gr.AnnotatedImage images and masks
    img_path_original = imgs[0]
    img_path_masks = imgs[1]

    section = img_path_masks[evt.index]
    img_path_mask = section[0]

    img_original = Image.open(img_path_original["name"])
    img_mask = Image.open(img_path_mask["name"])

    img_selected = image_from_mask(img_original, img_mask)
    # Return the selected section
    return img_selected


def main():
    with gr.Blocks() as demo:
        # UI Components
        with gr.Row():
            img_input = gr.Image(label="Base Image")
            with gr.Column():
                section_btn = gr.Button("Click to Segment Image")
        with gr.Row(equal_height=False):
            img_segmented = gr.AnnotatedImage(label="Segmented Image")
            img_segmented_selected = gr.Image(label="Selected Segment")

        section_btn.click(generate_image_sections, img_input, img_segmented)
        img_segmented.select(select_section, img_segmented, img_segmented_selected)

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
