import math
import dnnlib
import dnnlib.tflib as tflib
import pickle
import config
from BabyGAN.encoder.generator_model import Generator
import subprocess
import PIL.Image
from PIL import Image

def initialize_generator():
    tflib.init_tf()
    URL_FFHQ = "BabyGAN/karras2019stylegan-ffhq-1024x1024.pkl"
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    model_scale = int(2*(math.log(1024,2)-1))
    return generator

def encode_images_script(aligned_images_dir, generated_images_dir, latent_representations_dir):
    script_path = 'encode_images.py'  # Replace this with the actual path to the encode_images.py script

    # Prepare the command to run the encode_images.py script with the provided arguments
    command = (
        f'python {script_path} --early_stopping False --lr 0.25 --batch_size 2 --iterations 100 --output_video False '
        f'{aligned_images_dir} {generated_images_dir} {latent_representations_dir}'
    )

    # Run the command using subprocess
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the script: {e}")
    else:
        print("encode_images.py script executed successfully!")

def align_images_script(src_dir, aligned_images_dir):
    script_path = 'align_images.py'  # Replace this with the actual path to the align_images.py script

    # Prepare the command to run the align_images.py script with the provided arguments
    command = f'python {script_path} {src_dir} {aligned_images_dir}'

    # Run the command using subprocess
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    
    except subprocess.CalledProcessError as e:
        print(f"Error while running the script: {e}")
    else:
        print("align_images.py script executed successfully!")


    aligned_images_dir = "/content/BabyGAN/aligned_images"
    generated_images_dir = "/content/BabyGAN/generated_images"
    latent_representations_dir = "/content/BabyGAN/latent_representations"
    initialize_generator()
    encode_images_script(aligned_images_dir, generated_images_dir, latent_representations_dir)

def generate_final_image(latent_vector, direction, coeffs):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs*direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    Generator.set_dlatents(new_latent_vector)
    img_array = Generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    img.save("child.png")
    return "child.png"