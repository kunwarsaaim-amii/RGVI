import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from anvil.common.image.exr.exr import write as write_exr
from anvil.common.image.exr.read_rgb_or_mask import read_mask as read_exr_mask_anvil
from anvil.common.image.exr.read_rgb_or_mask import read_rgb
from diffusers import FluxFillPipeline

# Initialize the model pipeline
pipe = None

MASK_THRESHOLD = 0.01
OUTPUT_DIR = Path("gradio_temp")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_temp_image(image, suffix=".png", is_mask=False):
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=OUTPUT_DIR
        )

        arr = image.cpu().numpy() if hasattr(image, "cpu") else np.array(image)

        if is_mask:
            write_exr(temp_file.name, arr, ["A"], compression="none")
        else:
            write_exr(temp_file.name, arr, ["R", "G", "B"], compression="none")
        return str(temp_file.name)
    except Exception as e:
        print(f"ERROR: Failed to save temporary image: {e}")
        # Clean up temp file if creation succeeded but writing failed
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception as cleanup_error:
                print(
                    f"WARNING: Failed to cleanup temp file {temp_file.name}: {cleanup_error}"
                )
        return None


def initialize_model():
    """Initialize the FLUX Fill pipeline"""
    global pipe
    if pipe is None:
        print("Loading FLUX Fill model...")
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
        print("Model loaded successfully!")
    return pipe


def convert_mask_to_binary(mask_image):

    if mask_image.ndim == 3:
        # Convert to grayscale if mask is RGB
        mask_image = mask_image.squeeze()

    unique_vals = set(np.unique(mask_image))
    bool_vals_01 = set(np.array([0, 1]).astype(np.float32))
    bool_vals_0255 = set(np.array([0, 255]).astype(np.float32))
    is_bool_mask = unique_vals.issubset(bool_vals_01) or unique_vals.issubset(
        bool_vals_0255
    )

    if not is_bool_mask:
        # Apply threshold to convert to binary (assuming input is in 0-1 range)
        mask_image = (mask_image > MASK_THRESHOLD).astype(np.float32)
    elif unique_vals.issubset(bool_vals_0255):
        # Convert from 0-255 range to 0-1 range
        mask_image = (mask_image / 255.0).astype(np.float32)

    return mask_image


def process_image(
    input_image,
    mask_image,
    prompt,
    height,
    width,
    guidance_scale,
    num_inference_steps,
    max_sequence_length,
    seed,
):
    """Process the image using FLUX Fill model"""

    global pipe
    if pipe is None:
        return None, "Model not loaded. Please wait for the model to initialize."

    if input_image is None:
        return None, "Please upload an input image."

    if mask_image is None:
        return None, "Please upload a mask image."

    if not prompt or prompt.strip() == "":
        return None, "Please enter a prompt."

    try:
        # Read EXR images using the anvil read_rgb function
        input_array = read_rgb(input_image.name)
        mask_array = read_exr_mask_anvil(mask_image.name)
        mask_array = convert_mask_to_binary(mask_array)

        # Set up generator with seed
        generator = torch.Generator("cpu").manual_seed(seed)

        # Generate the image
        result = pipe(
            prompt=prompt,
            image=input_array,
            mask_image=mask_array,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
            output_type="pil",
        )

        output_image = result.images[0]

        # Save the output as EXR file
        # output_exr_path = save_temp_image(output_image, suffix=".exr", is_mask=False)
        if output_image is None:
            return None, "Error saving output EXR file."

        return output_image, "Image generated successfully!"

    except Exception as e:
        return None, f"Error generating image: {str(e)}"


# Create the Gradio interface
# Initialize the model when the app starts
print("Initializing FLUX Fill model at startup...")
initialize_model()


# def cleanup_outputs():
#     shutil.rmtree(OUTPUT_DIR)
#     OUTPUT_DIR.mkdir()


with gr.Blocks(title="FLUX Fill - Image Inpainting") as demo:
    gr.Markdown(
        """
        # FLUX Fill - AI Image Inpainting
        
        **Instructions:**
        1. Upload your input EXR image
        2. Upload a mask EXR image (white areas will be filled, black areas will be preserved)
        3. Enter a descriptive prompt for what you want to fill in
        4. Adjust parameters as needed
        5. Click "Generate" to create the filled image
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")

            input_image = gr.File(label="Input Image (.exr)", file_types=[".exr"])

            mask_image = gr.File(label="Mask Image (.exr)", file_types=[".exr"])

            prompt = gr.Textbox(
                label="Prompt",
                value="Empty background, high resolution",
                placeholder="Describe what you want to fill in the masked area...",
                lines=3,
            )
            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")

            output_image = gr.Image(label="Generated Image", type="pil")

            status_text = gr.Textbox(label="Status", interactive=False, lines=2)

    with gr.Row():
        gr.Markdown("### Parameters")

    with gr.Row():
        with gr.Column():
            height = gr.Number(label="Height", value=1200, minimum=256, maximum=2048)

            width = gr.Number(label="Width", value=2160, minimum=256, maximum=4096)

            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=50.0,
                step=1,
                value=30.0,
                info="Higher values follow the prompt more closely",
            )

        with gr.Column():
            num_inference_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=10,
                maximum=200,
                step=1,
                value=50,
                info="More steps = higher quality but slower",
            )

            max_sequence_length = gr.Slider(
                label="Max Sequence Length",
                minimum=128,
                maximum=1024,
                step=64,
                value=512,
                info="Maximum length for text encoding",
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                value=0,
                info="Random seed for reproducible results",
            )

    with gr.Row():
        clear_btn = gr.Button("Clear", variant="secondary")

    # Event handlers
    generate_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            mask_image,
            prompt,
            height,
            width,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
            seed,
        ],
        outputs=[output_image, status_text],
    )

    def clear_all():
        return None, None, "", None, "Cleared all inputs."

    clear_btn.click(
        fn=clear_all,
        outputs=[input_image, mask_image, prompt, output_image, status_text],
    )

    # Examples section
    gr.Markdown("### Tips:")
    gr.Markdown(
        """
        - **Input Image**: The original EXR image you want to modify
        - **Mask Image**: Black and white EXR image where white areas indicate what to fill/replace
        - **Prompt**: Describe what you want to appear in the masked areas
        - **Guidance Scale**: Higher values (20-40) follow your prompt more strictly
        - **Steps**: 30-50 steps usually provide good results
        - **Seed**: Use the same seed to reproduce results
        """
    )
    # demo.unload(cleanup_outputs)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=["gradio_temp"])
