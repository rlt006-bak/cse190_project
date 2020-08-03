"""
Preprocess a database so it can be used by the model
"""

import numpy as np
from math import ceil
from pretty_midi import PrettyMIDI, Note
from keras.utils import np_utils

# We use 100 time-shifts instead of 127, every time step will represent 10 ms
# So the dimension of our one-hot vector is 388
NUM_PITCHES = 128
NUM_TIME_SHIFTS = 100
NUM_VELOCITY = 32

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = NUM_PITCHES
TIME_SHIFT_OFFSET = (NOTE_OFF_OFFSET + NUM_PITCHES) - 1
VELOCITY_OFFSET = TIME_SHIFT_OFFSET + NUM_TIME_SHIFTS
ONE_HOT_DIM = NUM_PITCHES + NUM_PITCHES + NUM_TIME_SHIFTS + NUM_VELOCITY

# Constants for encoding
# stacks = [end_stack, start_stack]
START_STACK = 1
END_STACK = 0

# Constant for training
SEQUENCE_LENGTH = 25

class OneHot:

    def __init__(self, note_on, note_off, time_shift, velocity):
        """
        Create hot-vector encoding. Only one of the parameters is nonnegative. The rest are -1.
        - note_on is a pitch [0,127]
        - note_off is a pitch [0,127]
        - time_shift [1,100] is how many units of 10ms to shift by
        - velocity normalizes the 128 velocities into groups of 4
        """

        vec = np.zeros(ONE_HOT_DIM)
        if note_on != -1:
            self.uid = NOTE_ON_OFFSET + note_on
            vec[self.uid] = 1

        elif note_off != -1:
            self.uid = NOTE_OFF_OFFSET + note_off
            vec[self.uid] = 1

        elif time_shift != -1:
            self.uid = TIME_SHIFT_OFFSET + time_shift
            vec[self.uid] = 1

        elif velocity != -1:
            self.uid = VELOCITY_OFFSET + velocity
            vec[self.uid] = 1

        self.vec = vec

def decode_uid_tuple(uid):
    """
    decode uid back into tuple representation
    """

    # what the one-hot encoding looks like with the offset positions
    # [ ON . . . . | OFF . . . TIME | . . . . VEL | . . . . .]
    (note_on, note_off, time_shift, velocity) = (-1, -1, -1, -1)
    
    if uid < NOTE_OFF_OFFSET:
        note_on = uid
    elif uid <= TIME_SHIFT_OFFSET:
        note_off = uid - NOTE_OFF_OFFSET
    elif uid <=  VELOCITY_OFFSET:
        time_shift = uid - TIME_SHIFT_OFFSET
    else:
        velocity = uid - VELOCITY_OFFSET

    return (note_on, note_off, time_shift, velocity)

def convert_ms(seconds):
    """
    Convert seconds to milliseconds and round to nearest 10
    """

    return ceil(int(seconds * 1000) / 10) * 10

def convert_vel(velocity):
    """
    Convert the velocity to its corresponding bin [1,32]
    """

    return ceil((velocity + 1) / 4)

def convert_file(file_path):
    """
    Extract the notes from the midi file and convert into one-hot vector encodings
    Append to events_arr as an array, to keep midi files separate
    """
    
    print("Parsing {}".format(file_path))
    midi = PrettyMIDI(file_path)

    # check for multiple instruments
    events = []
    notes = []
    for instrument in midi.instruments:
        # Skip drums
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            # Starting and ending items are converted to multiples of 10ms
            notes.append(Note(start=convert_ms(note.start), end=convert_ms(note.end),
                              pitch=note.pitch, velocity=convert_vel(note.velocity)))

    # Convert the notes into one-hot vectors: [note on | note off | time shift | velocity]
    # Sort the notes by start time and by end time (view as a stack)
    # The first stack is for note on events; the second is for note off events
    notes_start_sort = sorted(notes, key=lambda x:x.start, reverse=True)
    notes_end_sort = sorted(notes, key=lambda x:x.end, reverse=True)
    stacks = [notes_end_sort, notes_start_sort]

    # curr_time is in ms
    curr_time = 0
    curr_velocity = -1
    while len(notes_start_sort) != 0 or len(notes_end_sort) != 0:

        # Prioritize the note at top of stacks with earliest end/start time
        # Time shift events take first priority + velocity must be before note on
        if len(notes_start_sort) != 0 and len(notes_end_sort) != 0:
            priority_stack = np.argmin((stacks[END_STACK][-1].end, 
                                        stacks[START_STACK][-1].start))
        elif len(notes_start_sort) != 0:
            priority_stack = START_STACK
        else:
            priority_stack = END_STACK
            
        note = stacks[priority_stack][-1]

        # Create time shift event
        if priority_stack == START_STACK:
            time_diff = note.start - curr_time
            curr_time = note.start
        else:
            time_diff = note.end - curr_time
            curr_time = note.end

        while time_diff > 0:
            time = 1000 if time_diff > 1000 else time_diff
            time_diff -= time

            # Get time in terms of time shifts [1, ..., 100]
            time_shift = int(time / 10)
            events.append(OneHot(-1, -1, time_shift, -1))
                 
        # Create note on event, first checking for velocity
        if priority_stack == START_STACK:
            if note.velocity != curr_velocity:
                events.append(OneHot(-1, -1, -1, note.velocity))
                curr_velocity = note.velocity
            
            events.append(OneHot(note.pitch, -1, -1, -1))

        # Create note off event, first checking for velocity
        else:
            events.append(OneHot(-1, note.pitch, -1, -1))

        # Pop note
        stacks[priority_stack].pop()

    # Return the list of events
    return events 

def prepare_sequences(events_arr):
    """
    Prepare input sequences for the network after we finish parsing the midi files.
    Each input sequence will be the uid encoding of each vector
    """
   
    network_input = []
    network_output = []

    # create input sequences with corresponding network_outputs
    for events in events_arr:
        for i in range(0, len(events) - SEQUENCE_LENGTH):
            sequence_in = [hot_vec.uid for hot_vec in events[i:i + SEQUENCE_LENGTH]]
            sequence_out = events[i + SEQUENCE_LENGTH].vec
            network_input.append(sequence_in)
            network_output.append(sequence_out)

    # reshape input for network and normalize
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    network_input = network_input / float(ONE_HOT_DIM)

    return (network_input, np.array(network_output))

def prepare_sequences_predict(events_arr):
    """
    Prepare input sequences for the prediction.
    Each input sequence will be the uid encoding of each vector
    """
    
    network_input = []

    # create input sequences with corresponding network_outputs
    for events in events_arr:
        for i in range(0, len(events) - SEQUENCE_LENGTH):
            sequence_in = [hot_vec.uid for hot_vec in events[i:i + SEQUENCE_LENGTH]]
            network_input.append(sequence_in)


    # reshape input for network and normalize
    n_patterns = len(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    normalized_input = normalized_input / float(ONE_HOT_DIM)

    return (network_input, normalized_input)
