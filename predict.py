import os
import subprocess
import shutil
import gdown
import tqdm
from cog import Path, Input, List
import subprocess
from utils import align_images_script, initialize_generator, encode_images_script, generate_final_image
import numpy as np

class AiChildGenerator():
    def __init__(self):
           # Step 1: Clone the repository and navigate to the project directory
        subprocess.run(["git", "clone", "https://github.com/tg-bomze/BabyGAN.git"])
        os.chdir("BabyGAN")

        # Step 2: Create some required directories
        os.makedirs("aligned_images", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("father_image", exist_ok=True)
        os.makedirs("mother_image", exist_ok=True)

        # Step 3: Download files using gdown
        files_to_download = [
            "https://drive.google.com/uc?id=1Kxen0y7qKRFXfkaMHMJCHeDitLdgsbzh",
            "https://drive.google.com/uc?id=1dtGZ4hvr-WYt8R75U94kR50jc_sidMe3",
            "https://drive.google.com/uc?id=1IZbZPfbxg7jjOdxNb_AGU2WoIl9Ayw_H",
            "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ",
            "https://drive.google.com/uc?id=1XRLsVq0ULt0sR3cmgjMTqwSgRTQh-GJH" 
        ]

        for file_url in tqdm.tqdm(files_to_download):
            gdown.download(file_url, quiet=True)

        # Step 4: Move 'finetuned_resnet.h5' to the 'data' folder
        source_file = "finetuned_resnet.h5"
        destination_folder = "data"
        shutil.move(source_file, os.path.join(destination_folder, "finetuned_resnet.h5"))


    def predict(
        self,
        father_image: Path = Input(
            description="image of the father",
        ),
        mother_image: Path = Input(
            description="image of the mother",
        )) -> List[Path]:
        
        FATHER_FILENAME = "father." + father_image.split(".")[-1]
        os.rename(father_image, FATHER_FILENAME)

        src_dir = "BabyGAN/father_image"
        aligned_images_dir = "BabyGAN/aligned_images"
        align_images_script(src_dir, aligned_images_dir)   

        MOTHER_FILENAME = "mother." + mother_image.split(".")[-1]
        os.rename(mother_image, MOTHER_FILENAME)

        src_dir = "BabyGAN/mother_image"
        aligned_images_dir = "BabyGAN/aligned_images"
        align_images_script(src_dir, aligned_images_dir)  


        aligned_images_dir = "/content/BabyGAN/aligned_images"
        generated_images_dir = "/content/BabyGAN/generated_images"
        latent_representations_dir = "/content/BabyGAN/latent_representations"
        initialize_generator()
        encode_images_script(aligned_images_dir, generated_images_dir, latent_representations_dir)

        first_face = np.load('BabyGAN/latent_representations/father_01.npy')
        second_face = np.load('BabyGAN/latent_representations/mother_01.npy')
        age_direction = np.load('BabyGAN/ffhq_dataset/latent_directions/age.npy')
        genes_influence = 0.3 
        hybrid_face = ((1-genes_influence)*first_face)+(genes_influence*second_face)
        person_age = 10 #@param {type:"slider", min:10, max:50, step:1}
        intensity = -((person_age/5)-6)
        output_path = generate_final_image(hybrid_face, age_direction, intensity)

        return output_path












