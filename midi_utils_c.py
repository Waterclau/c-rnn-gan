import midi
import numpy as np

def midi_to_input(midi_data, num_outputs):
    """
    Converts MIDI data to input format for the model.
    """
    # Create piano roll representation
    piano_roll = midi_data.get_piano_roll(fs=50)
    piano_roll = piano_roll[:, :num_outputs]
    piano_roll = np.transpose(piano_roll)

    # Convert to binary input
    piano_roll[piano_roll > 0] = 1

    # Add time dimension
    piano_roll = np.expand_dims(piano_roll, axis=0)

    return piano_roll


def input_to_midi(piano_roll, filename, program=0):
    """
    Converts model output to a MIDI file.
    """
    # Remove time dimension
    piano_roll = np.squeeze(piano_roll, axis=0)

    # Convert to MIDI format
    piano_roll = np.transpose(piano_roll)
    piano_roll[piano_roll < 0.5] = 0
    piano_roll[piano_roll >= 0.5] = 1
    piano_roll = np.concatenate((piano_roll, np.zeros((128, 2))), axis=1)

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    for i, time_step in enumerate(piano_roll):
        note_on = []
        note_off = []
        for j, note in enumerate(time_step):
            if note == 1 and j not in note_on:
                note_on.append(j)
            elif note == 0 and j not in note_off:
                note_off.append(j)
        for note in note_on:
            note_start = midi.NoteOnEvent(tick=0, velocity=80, pitch=note+21, channel=0)
            track.append(note_start)
        for note in note_off:
            note_stop = midi.NoteOffEvent(tick=0, velocity=80, pitch=note+21, channel=0)
            track.append(note_stop)

    # Add program change event
    program_change = midi.ProgramChangeEvent(tick=0, channel=0, data=[program])
    track.append(program_change)

    # Save MIDI file
    midi.write_midifile(filename, pattern)


class MIDIFile:
    """
    Class for loading MIDI files and generating input for the model.
    """
    def __init__(self, filename, num_outputs=88, normalize=False, denoise=False):
        self.filename = filename
        self.num_outputs = num_outputs
        self.normalize = normalize
        self.denoise = denoise

    def preprocess_data(self, data):
        if self.normalize:
            data = normalize_data(data)
        if self.denoise:
            data = denoise_data(data)
        return data

    def get_sequence(self, one_hots=False):
        midi_data = midi.read_midifile(self.filename)
        midi_data = self.preprocess_data(midi_data)
        sequence = midi_to_input(midi_data, self.num_outputs)
        if one_hots:
            sequence = np.eye(2)[sequence.reshape(-1)].reshape(sequence.shape[0], sequence.shape[1], -1)
        return sequence



def normalize_data(midi_data):
    # Normalize velocity values between 0 and 1
    for track in midi_data:
        for event in track:
            if isinstance(event, midi.NoteOnEvent) or isinstance(event, midi.NoteOffEvent):
                event.velocity /= 127.0
                
def remove_noise(midi_data, velocity_threshold=0.1):
    # Remove any notes with velocity below a threshold
    for track in midi_data:
        notes_to_remove = []
        for i, event in enumerate(track):
            if isinstance(event, midi.NoteOnEvent):
                if event.velocity < velocity_threshold:
                    notes_to_remove.append(i)
        for i in reversed(notes_to_remove):
            del track[i]

def correct_errors(midi_data):
    # Correct any out-of-range notes
    note_range = range(21, 109)
    for track in midi_data:
        for event in track:
            if isinstance(event, midi.NoteOnEvent) or isinstance(event, midi.NoteOffEvent):
                if event.pitch not in note_range:
                    if event.pitch < note_range.start:
                        event.pitch = note_range.start
                    elif event.pitch >= note_range.stop:
                        event.pitch = note_range.stop - 1
def preprocess_midi_file(filename):
    midi_data = midi.read_midifile(filename)
    normalize_data(midi_data)
    remove_noise(midi_data)
    correct_errors(midi_data)
    return midi_data

