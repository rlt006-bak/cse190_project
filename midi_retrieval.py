"""
Obtain the midi files for the prepreprocessing step
"""

import os
import numpy as np

def response():
    
    get_response = input()
    if get_response == "Y":
        return 1
    else:
        return 0

def get_directories():
    """
    Choose the directories containing the midi files we want to include
    """

    # Specify genres to include
    genres = ["classical", "jazz", "halo"]
    int_genre_dict = {index:genre for index, genre in enumerate(genres)}
    genre_path_dict = {"classical": "training_data/classical",
                       "jazz": "training_data/jazz",
                       "halo": "training_data/halo"}
    include_genre_bools = np.array(range(len(genres)))

    for i in range(len(genres)):
        print("Include {genre}? Y/N".format(genre=int_genre_dict[i]))

        if response() == 1:
            include_genre_bools[i] = 1
        else:
            include_genre_bools[i] = 0

    # Get all the leaf directories of included genres
    leaf_directories = []
    for index, include in enumerate(include_genre_bools):
        if include == 1:
            genre = int_genre_dict[index]
            genre_path = genre_path_dict[genre]

            # choose which leaf directory to include
            for root, dirs, files in os.walk(genre_path):
                if not dirs:
                    print("Include: {}? Y/N".format(root))
                    
                    if response() == 1:
                        leaf_directories.append(root)

    return leaf_directories


def get_files(directory):
    """
    From the chosen directory, returns a list of all midi file paths
    """

    midi_files = []
    for file in os.listdir(directory):
        if file.endswith(".mid"):
            midi_files.append(os.path.join(directory, file))

    return midi_files
