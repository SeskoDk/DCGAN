import os
import imageio


def get_numeric_name(filename):
    first_split = filename.split("_")[1]
    second_split = int(first_split.split(".")[0])
    return second_split


def main():
    path = "gen_images"
    files = [file for file in os.listdir(path)]
    sorted_files = sorted(files, key=get_numeric_name)
    sorted_files = [os.path.join(path, file) for file in sorted_files]

    nth_files = [sorted_files[idx] for idx in range(0, len(sorted_files), 50)]

    with imageio.get_writer("pokemon_animation.gif", mode="I") as writer:
        for file in nth_files:
            image = imageio.imread(file)
            writer.append_data(image)


if __name__ == "__main__":
    main()
