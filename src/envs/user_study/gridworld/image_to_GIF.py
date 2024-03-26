from PIL import Image

for j in range(1,13):
    directory = f'src/envs/user_study/gridworld/game_frames/trajectory_{j}'

    # File names of the images to combine
    image_files = [f'frame_{i}.png' for i in range(1, 6)]

    # Load the images
    images = [Image.open(f'{directory}/{img}') for img in image_files]

    # Save the images as a gif
    output_path = f'{directory}/output.gif'
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=500, loop=0)

    print(f"GIF saved at {output_path}")