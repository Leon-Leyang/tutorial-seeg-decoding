import bz2
import numpy as np
import _pickle as pickle
from tqdm import tqdm


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


if __name__ == "__main__":
    seeg = preprocess_seeg()
    np.save("../data/seeg.npy", seeg)
