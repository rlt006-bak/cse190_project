import numpy as np
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint
from pretty_midi import PrettyMIDI, Instrument, Note

from midi_retrieval import get_directories, get_files
from preprocess import convert_file, prepare_sequences, prepare_sequences_predict
from preprocess import decode_uid_tuple, NUM_PITCHES, ONE_HOT_DIM


def create_network(network_input):
    
    model = Sequential()
    model.add(LSTM( 
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]), 
        return_sequences=True
    ))
    model.add(LSTM(512, recurrent_dropout=0.3))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(ONE_HOT_DIM))
    model.add(Lambda(lambda x: x / 0.6))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )

    return model


def train_network():
    
    # Get all the midi files to include in training
    dir_list = get_directories()
    midi_files = []
    for directory in dir_list:
        midi_files.extend(get_files(directory))

    # Parse the midi files
    events_arr = []
    for midi in midi_files:
        events_arr.append(convert_file(midi))
    pickle.dump(events_arr, open("events_arr.p", "wb"))

    # Prepare the input sequences
    (network_input, network_output) = prepare_sequences(events_arr)

    # Start training the network
    model = create_network(network_input)
    checkpoint = ModelCheckpoint(
        "weights.hdf5",
        monitor="loss",
        verbose=0,
        save_best_only=True,
        mode="min"
    )

    callbacks_list = [checkpoint]
    model.fit(
        x=network_input,
        y=network_output,
        batch_size=64,
        epochs=50,
        callbacks=callbacks_list
    )


def get_next_note(prediction):

    # 0) Choose next note by taking top k values and then choosing randomly
    # 1) Choose with respect to relative probability
    # 2) Choose max probability
    option = 0
    if option == 0:
        k = 3
        index = np.random.choice(np.argsort(prediction[0])[-k:])
    elif option == 1:
        index = np.random.choice(a=range(len(prediction[0])), p=prediction[0])
    elif option == 2:
        index = np.argmax(prediction[0])

    return index


def generate_notes(model, network_input):
    
    # Choose starting event sequence at random
    print("Generating music...")
    start = np.random.randint(0, len(network_input)-1) 
    pattern = network_input[start]
    prediction_output = []

    # Generate the next sequence of events
    num_generations = 1000
    for note_index in range(num_generations):

        # get pattern of events and normalize before predicting
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / ONE_HOT_DIM
        prediction = model.predict(prediction_input)

        # get the next event (uid) from probabilities
        index = get_next_note(prediction)
        prediction_output.append(index)

        # prepare for next generation
        pattern.append(index)
        pattern = pattern[1:len(pattern)]


    # decode the uids into format (note_on, note_off, time_shift, velocity)
    return [decode_uid_tuple(uid) for uid in prediction_output]


def create_midi(prediction_output, gen_num):
    """
    From the prediction output of form (note_on, note_off, time_shift, velocity),
    construct the midi file with the following rules:
        1) time_shift event adds onto the offset of midi events
        2) velocity events set the velocity of all notes appearing after
    """

    print("Creating midi file...")
    pm = PrettyMIDI()
    piano = Instrument(0)
    pm.instruments.append(piano)

    # keeps track of which notes are on and off
    # so we turn a note on iff if it was off previously
    note_arr_bool = np.zeros(NUM_PITCHES)
    note_start_time = dict((pitch, 0) for pitch in range(NUM_PITCHES))

    # current time in milliseconds, default velocity bin at mezzoforte
    curr_time = 0
    curr_velocity = 6

    # bool: if 0, do nothing to note_on notes with no corresponding note_off
    #       if 1, use reservoir sampling to add note_on notes after we finish parsing
    # time_shift_i = the number of time_shift events seen so far
    fix_note_on = 1
    time_shift_i = 0
    time_shift_reservoir = 0

    # also keep a velocity reservoir
    velocity_i = 0
    velocity_reservoir = 0

    # 0 - normal
    # 1 - split notes on consecutive note_on events
    option = 0

    # construct the sequence of note events
    total_missed = 0
    for (note_on, note_off, time_shift, velocity) in prediction_output:

        if note_on != -1:
            # only create note_on events if the note is not currently on
            if option == 0 and note_arr_bool[note_on] == 0:
                note_arr_bool[note_on] = 1
                note_start_time[note_on] = curr_time

            # if the note is already on and we encounter a note_on event, split the note
            elif option == 1 and note_arr_bool[note_on] == 1:
                piano.notes.append(Note(start=note_start_time[note_on] / 1000,
                                        end=curr_time / 1000,
                                        pitch=note_on,
                                        velocity=curr_velocity))
                note_start_time[note_on] = curr_time

        elif note_off != -1 and note_arr_bool[note_off] == 1:
            # convert times into seconds
            piano.notes.append(Note(start=note_start_time[note_off] / 1000,
                                    end=curr_time / 1000,
                                    pitch=note_off,
                                    velocity=curr_velocity))

            note_arr_bool[note_off] = 0

        elif time_shift != -1:
            # update time_shift num seen and choose to switch or not
            # on the ith shift, replace with probabiliy 1/i, otherwise stay
            time_shift_i += 1
            time_shift_reservoir = np.random.choice(a=[time_shift, time_shift_reservoir], 
                                                    p=[1/time_shift_i, 1-1/time_shift_i])

            # each time_shift unit represents 10 ms
            curr_time += time_shift * 10


        elif velocity != -1:
            # scale the velocity bins to actual velocity 
            velocity_i += 1
            curr_velocity = (velocity * 4) - 1
            velocity_reservoir = np.random.choice(a=[curr_velocity, velocity_reservoir],
                                                  p=[1/velocity_i, 1-1/velocity_i])

        # represents skipped events
        else:
            total_missed += 1

    # reconstruct remaining note_on events, convert the time_shift to ms
    if fix_note_on == 1:
        print("Used reservoir")
        time_shift = time_shift_reservoir * 10
        for curr_pitch, state in enumerate(note_start_time):
            # if the note is on
            if state == 1:
                piano.notes.append(Note(start=note_start_time[curr_pitch] / 1000,
                                        end=(note_start_time[curr_pitch] + time_shift) / 1000,
                                        pitch=curr_pitch,
                                        velocity=velocity_reservoir))
    # output midi
    pm.write("generated_music/output{}.mid".format(gen_num))
    print("Total events skipped: {}".format(total_missed))

def generate():
    
    # Get the processed notes
    events_arr = pickle.load(open("events_arr.p", "rb"))
   
    # Set up the network with final weights
    (network_input, normalized_input) = prepare_sequences_predict(events_arr)
    model = create_network(normalized_input)
    model.load_weights("weights.hdf5")

    # Generate midi
    print("How many files do you want?")
    num = int(input())
    for i in range(num):
        print("---File {}---".format(i + 1))
        prediction_output = generate_notes(model, network_input)
        create_midi(prediction_output, i)
