import glob
from shutil import copyfile, copy, copy2, copytree, copyfile
import os

base_dir = os.path.dirname(os.path.realpath(__file__))

emotions_list = ["neutral", "anger", "contempt",
                 "disgust", "fear", "happy", "sadness", "surprise"]

# emotions_list = ["neutral", "anger", "happy", "sadness"]
# Returns a list of all folders with participant numbers
emotions_folders = glob.glob("emotions\\*")

# print(glob.glob('images\\%s\\%s\\*' % ("S005", "001")))


def imageWithEmotionExtraction():
    for x in emotions_folders:
        participant = "%s" % x[-4:]  # store current participant number
        for sessions in glob.glob("%s\\*" % x):
            for files in glob.glob("%s\\*_emotion.txt" % sessions):
                current_session = files[23:26]
                file = open(files, 'r+')
                emotion = int(float(file.readline()))
                file.close()

                # print(participant, sessions, files, emotion, current_session)

                # get path for last image in sequence, which contains the emotion
                sourcefile_emotion = glob.glob(
                    'images\\%s\\%s\\*' % (participant, current_session))[-1]
                # do same for neutral image
                sourcefile_neutral = glob.glob(
                    'images\\%s\\%s\\*' % (participant, current_session))[0]
                # Generate path to put neutral image
                dest_neut = 'selected_set\\neutral\\%s' % sourcefile_neutral[16:]
                # Do same for emotion containing image
                dest_emot = 'selected_set\\%s\\%s' % (
                    emotions_list[emotion], sourcefile_emotion[16:])
                print(sourcefile_neutral, sourcefile_emotion,
                      dest_neut, dest_emot)
                copyfile(sourcefile_neutral, dest_neut)  # Copy file
                copyfile(sourcefile_emotion, dest_emot)  # Copy file


if __name__ == '__main__':
    imageWithEmotionExtraction()
