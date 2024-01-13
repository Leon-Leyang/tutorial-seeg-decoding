import bz2
import numpy as np
import _pickle as pickle
from tqdm import tqdm
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def preprocess_seeg():
    chapter_start_end_timestep = get_start_end_timestep_4_chapters()

    # The names of the contacts
    contact_names = ["LAH1", "LAH2", "LAH3", "LAH4", "LAH5", "LAH6", "LAH7", "LAH8", "LAH9", "LAH10", "LAH11", "LAH12",
                     "LAH13", "LAH14","LAMY1", "LAMY2", "LAMY3", "LAMY4", "LAMY5", "LAMY6", "LAMY7", "LAMY8", "LAMY9",
                     "LAMY10", "LAMY11", "LAMY12", "LAMY13", "LAMY14", "LPH1", "LPH2", "LPH3", "LPH4", "LPH5", "LPH6",
                     "LPH7", "LPH8", "LPH9", "LPH10", "LPH11", "LPH12", "LPH13", "LPH14", "RPH1", "RPH2", "RPH3", "RPH4",
                     "RPH5", "RPH6", "RPH7", "RPH8", "RPH9", "RPH10", "RPH11", "RPH12", "RPH13", "RPH14", "RAMY1", "RAMY2",
                     "RAMY3", "RAMY4", "RAMY5", "RAMY6", "RAMY7", "RAMY8", "RAMY9", "RAMY10", "RAMY11", "RAMY12", "RAMY13",
                     "RAMY14", "RAH1", "RAH2", "RAH3", "RAH4", "RAH5", "RAH6", "RAH7", "RAH8", "RAH9", "RAH10", "RAH11",
                     "RAH12", "RAH13", "RAH14"]

    # The list to store the data
    # Should be of shape (num_contacts, num_timesteps_in_total)
    seeg = []

    # For each contact, concatenate the data from chapters
    for contact in tqdm(contact_names):
        # Load data for the contact
        filename = f"../data/e0006KR/2019-06-04_e0006KR_{contact}.pbz2"
        orig_data_contact = bz2.BZ2File(filename, "rb")
        orig_data_contact = pickle.load(orig_data_contact)['signal']

        # Extract data according to each chapter's start and end timestep and concatenate them
        data_contact = []
        for start_timestep, end_timestep in chapter_start_end_timestep:
            data_contact.append(orig_data_contact[start_timestep:end_timestep])
        data_contact = np.concatenate(data_contact)
        seeg.append(data_contact)

    seeg = np.array(seeg)
    return seeg


def get_start_end_timestep_4_chapters():
    # The list to store the start and end timestep of each chapter
    chapter_start_end_timestep = []

    # Load the event file
    event_file = "../data/e0006KR/2019-06-04_e0006KR_Events.pbz2"
    data_event = bz2.BZ2File(event_file, "rb")
    data_event = pickle.load(data_event)

    # Get the start and end timestep of each chapter(9 signals a chapter start, 18 signals a chapter end)
    for i in range(len(data_event['signal'])):
        if data_event['signal'][i][0] == 9:
            start_timestep = data_event['signal'][i][1]
        elif data_event['signal'][i][0] == 18:
            end_timestep = data_event['signal'][i][1]
            chapter_start_end_timestep.append((start_timestep, end_timestep))

    # Drop the last chapter
    chapter_start_end_timestep = chapter_start_end_timestep[:-1]

    return chapter_start_end_timestep

def getDuration(filepath):

    # Open the video file
    cap = cv2.VideoCapture(filepath)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get the frames per second (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the duration of the video in seconds
        duration_seconds = total_frames / fps
     
        cap.release()
        return total_frames
def getFrameTotal():
    file_list = sorted(os.listdir("/media/data_cifs_lrs/projects/prj_tutorial_seeg-decoding/green_book_movie"))
    filtered_file_list = [f for f in file_list if f.endswith("avi") and f != ".DS_Store"]

    frame_total=[]
    for f in filtered_file_list:
        
        filepath= "/media/data_cifs_lrs/projects/prj_tutorial_seeg-decoding/green_book_movie/"+f

        
        total_frames = getDuration(filepath)
        frame_total.append(total_frames)
    return frame_total
def getXMLfiles():
    foldername ="/media/data_cifs_lrs/projects/prj_tutorial_seeg-decoding/greenbook/raw/face/"
    xml_file_list = sorted(os.listdir(foldername))
    xml_files =[]
    for f in xml_file_list :
        xml_files.append("/media/data_cifs_lrs/projects/prj_tutorial_seeg-decoding/greenbook/raw/face/"+f)
    return xml_files

def extract_tony_frames(xml_file, frame_offset):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tony_frames = []
    for box in root.findall('.//box'):
        character = box.find('attribute[@name="character"]')
        if character is not None and character.text == "Tony":
            frame = int(box.get('frame')) + frame_offset
            if frame not in tony_frames:
                tony_frames.append(frame)
    return sorted(tony_frames)

#

def create_tony_frames():
    # Total frames in each XML file
    total_frames_each_file =  getFrameTotal()
    xml_files =  getXMLfiles()
    # Calculate the accumulated frame count
    tony_frames_dict = {}
    frame_offset = 0
    for index, (xml_file, total_frames) in enumerate(zip(xml_files, total_frames_each_file)):
        tony_frames = extract_tony_frames(xml_file, frame_offset)
        tony_frames_dict[index] = tony_frames
        frame_offset += total_frames
    combined_frames = []
    for key in tony_frames_dict:
        combined_frames.extend(tony_frames_dict[key])

    return combined_frames


# Function to create a list indicating which seconds contain Tony
def seconds_with_tony(frames_list, frames_per_second):
    # Create a dictionary to count frames of Tony in each second
    frame_count_per_second = {}
    for frame in frames_list:
        second = frame // frames_per_second
        frame_count_per_second[second] = frame_count_per_second.get(second, 0) + 1

    # Initialize an empty list for the final output
    seconds_with_tony_list = [0] * (total_frames // frames_per_second + 1)

    # Mark the seconds with more than 10 frames of Tony as 1
    for second, count in frame_count_per_second.items():
        if count > 10:
            seconds_with_tony_list[second] = 1

    return seconds_with_tony_list




if __name__ == "__main__":
    seeg = preprocess_seeg()
    np.save("../data/seeg.npy", seeg)
    # Video frame length and frames per second
    total_frames = 234267
    frames_per_second = 30

    # List of frame numbers that contain the character Tony
    frames_with_tony = create_tony_frames()
   
    seconds_with_tony_array = seconds_with_tony(frames_with_tony,30)
    np.save("../data/seconds_with_tony.npy", seconds_with_tony_array )


