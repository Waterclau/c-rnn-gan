import tensorflow as tf
import numpy as np
import os
import argparse
from midi_util import MIDIFile
save_dir = "results\model2-1\model-results"


# Model parameters
BATCH_SIZE = 1
SEQ_LENGTH = 32
DATA_TYPE = np.uint8
NUM_OUTPUTS = 128

# Sample an index from a probability distribution
def sample_from_probabilities(probabilities, temperature=1.0):
    # Convert to float64 to avoid numpy/Python int32 overflow errors
    probabilities = np.asarray(probabilities).astype('float64')
    # Apply temperature
    if temperature != 1.0:
        probabilities = np.power(probabilities, 1.0 / temperature)
        probabilities = probabilities / np.sum(probabilities)
    # Sample
    return np.random.choice(len(probabilities), p=probabilities)

# Generate a music sequence
def generate_music(session, input_ph, probabilities_op, initial_state, final_state,
                   initial_sequence, length, temperature):
    sequence = initial_sequence
    # Initialize state
    state = session.run(initial_state, {input_ph: np.zeros((BATCH_SIZE, SEQ_LENGTH, NUM_OUTPUTS)),
                                        initial_state: np.zeros((BATCH_SIZE, 2 * NUM_OUTPUTS))})
    # Generate sequence
    for i in range(length):
        # Compute probabilities for next note
        feed = {input_ph: sequence[-1:, np.newaxis], initial_state: state, temperature_ph: temperature}
        probabilities, state = session.run([probabilities_op, final_state], feed)
        # Sample next note
        index = sample_from_probabilities(probabilities[0], temperature=temperature)
        # Add to sequence
        new_note = np.zeros((1, NUM_OUTPUTS))
        new_note[0, index] = 1
        sequence = np.concatenate([sequence, new_note], axis=0)
    return sequence

# Save a music sequence as a MIDI file
def save_midi(filename, sequence):
    # Convert to MIDI events
    events = []
    current_time = 0
    for note in sequence:
        if np.max(note) > 0:
            pitch = np.argmax(note)
            events.append({'type': 'note_on', 'time': current_time, 'pitch': pitch, 'velocity': 64})
            current_time = 0
        else:
            current_time += 1
    # Write MIDI file
    midi = MIDIFile()
    midi.add_track(events)
    with open(filename, 'wb') as output_file:
        midi.write(output_file)

# Parse arguments
parser = argparse.ArgumentParser(description='Generate music using a trained C-RNN-GAN model')
parser.add_argument('--save-dir', type=str, default='save',
                    help='directory to load model from')
parser.add_argument('--length', type=int, default=1000,
                    help='length of music sequence to generate')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature of the softmax distribution used for sampling')
parser.add_argument('--seed', type=str, default=None,
                    help='path to MIDI file to use as initial sequence')
parser.add_argument('--output', type=str, default='output.mid',
                    help='filename to save the generated MIDI file')
args = parser.parse_args()

# Load model
#meta_path = os.path.join(args.save_dir, 'D:\Comillas\4º\TFG\v2\c-rnn-gan\results\model2-1\model.ckpt-16.meta')
meta_path = os.path.join(save_dir, "model.ckpt-12.meta")
checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
#saver = tf.compat.v1.train.import_meta_graph(meta_path)
saver = tf.train.import_meta_graph(meta_path)
#print(tf.get_collection('inputs'))
input_ph = tf.get_collection('inputs')
initial_state = tf.get_collection('initial_state')
final_state = tf.get_collection('final_state')
probabilities_op = tf.get_collection('probabilities_op')
temperature_ph = tf.get_collection('temperature_ph')
session = tf.Session()
print(session.run(input_ph))
saver.restore(session, checkpoint_path)

# Load initial sequence
if args.seed is not None:
    # Load from MIDI file
    sequence = MIDIFile(args.seed).get_sequence(one_hots=True)
else:
    # Use silence as initial sequence
    sequence = np.zeros((1, NUM_OUTPUTS))

# Generate music
generated = generate_music(session, input_ph, probabilities_op, initial_state, final_state,
                           sequence, length=args.length, temperature=args.temperature)

# Save as MIDI file
save_midi(args.output, generated)
print('Saved generated music to', args.output)



