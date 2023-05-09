import json
import librosa
import librosa.display
import argparse
import os

from tqdm import tqdm
from scipy import signal
from scipy.signal import butter
from madmom.features.beats import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


ap = argparse.ArgumentParser()
ap.add_argument('-build', '--build', required=False, action='store_true')
ap.add_argument('-test', '--test', required=False, action='store_true')
args = ap.parse_args()

train_path = './train'


def peak_picking(beat_times, total_samples, kernel_size, offset):
    # smoothing the beat function
    cut_off_norm = len(beat_times) / total_samples * 100 / 2
    ([b, a]) = butter(1, cut_off_norm)
    beat_times = signal.filtfilt(b, a, beat_times)

    # creating a list of samples for the rnn beats
    beat_samples = np.linspace(0, total_samples, len(beat_times), endpoint=True, dtype=int)

    n_t_medians = signal.medfilt(beat_times, kernel_size=kernel_size)
    offset = 0.01
    peaks = []

    for i in range(len(beat_times) - 1):
        if beat_times[i] > 0:
            if beat_times[i] > beat_times[i - 1]:
                if beat_times[i] > beat_times[i + 1]:
                    if beat_times[i] > (n_t_medians[i] + offset):
                        peaks.append(int(beat_samples[i]))
    return peaks


def analyze(y, sr):
    data = {}

    # sample rate
    data['sample_rate'] = sr

    # getting duration in seconds
    data['duration'] = librosa.get_duration(y=y, sr=sr)

    rnn_processor = RNNBeatProcessor(post_processor=None)
    predictions = rnn_processor(y)
    mm_processor = MultiModelSelectionProcessor(num_ref_predictions=None)
    beats = mm_processor(predictions)

    data['beat_samples'] = peak_picking(beats, len(y), 5, 0.01)

    if len(data['beat_samples']) < 3:
        data['beat_samples'] = peak_picking(beats, len(y), 25, 0.01)

    if data['beat_samples'] == []:
        data['beat_samples'] = [0]

    data['number_of_beats'] = len(data['beat_samples'])

    # tempo
    data['tempo_float'] = (len(data['beat_samples']) - 1) * 60 / data['duration']
    data['tempo_int'] = int(data['tempo_float'])

    # spectral features
    notes = []

    try:
        chroma = librosa.feature.chroma_cqt(y, sr=sr, n_chroma=12, bins_per_octave=12, n_octaves=8, hop_length=512)

        # CONVERSION TABLE
        # 0     c	  261.63
        # 1     c#	  277.18
        # 2	    d	  293.66
        # 3	    d#	  311.13
        # 4	    e	  329.63
        # 5	    f	  349.23
        # 6	    f#	  369.99
        # 7	    g	  392.00
        # 8	    g#	  415.30
        # 9	    a	  440.00
        # 10	a#	  466.16
        # 11	b	  493.88

        for col in range(chroma.shape[1]):
            notes.append(int(np.argmax(chroma[:, col])))

        data['notes'] = notes
        data['notes_freq'] = {
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,
            '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0
        }
        for note in notes:
            data['notes_freq'][str(note)] += 1
        data['dominant_note'] = int(np.argmax(np.bincount(np.array(notes))))
    except:
        data['notes'] = [0]
        data['dominant_note'] = 0

    return data


def one_batch(path):
    y, sr = librosa.load(path=path)
    cnt_samples = len(y)

    # trim for 10 seconds
    lp = 10 * sr
    hp = cnt_samples - 10 * sr

    # ROI in the whole signals
    result = analyze(y[lp:hp], sr)

    # get beat tendency
    beats = result['beat_samples']
    beat_tendency = list()
    for index in range(1, len(beats)):
        beat_tendency.append(round(beats[index] / beats[index - 1], 2))

    database = {
        'path': path,
        'tempo_float': round(result['tempo_float'], 2),
        'notes_freq': result['notes_freq'],
        'dominant_note': result['dominant_note'],
        'beat_tendency': beat_tendency,
        'num_beat': len(beat_tendency)
    }

    return database


if args.build:
    db = list()
    files = os.listdir(train_path)
    for each_file in tqdm(files):
        f_path = os.path.join(train_path, each_file)
        db.append(one_batch(path=f_path))

    with open('db.json', 'w') as f:
        json.dump(db, f)

elif args.test:
    # loading data
    with open('db.json', 'r') as f:
        db = json.load(f)

    path = './data/24K Magic_Instrumental.mp3'
    test_data = one_batch(path=path)
    print(test_data)

    for each in db:
        print(each['path'])

        # tempo
        if each['tempo_float'] > test_data['tempo_float']:
            big_tempo = test_data['tempo_float'] / each['tempo_float']
        else:
            big_tempo = each['tempo_float'] / test_data['tempo_float']

        print('tempo: {}%'.format(round(big_tempo * 100, 2)))

        # note freq
        db_norm = round(np.linalg.norm(list(each['notes_freq'].values())), 3)
        test_norm = round(np.linalg.norm(list(test_data['notes_freq'].values())), 3)

        sum = 0
        for k, v in each['notes_freq'].items():
            sum += (each['notes_freq'][k] * test_data['notes_freq'][k])

        cosin_sim = sum / (db_norm * test_norm)
        print('chord tendency: {}%'.format(round(cosin_sim * 100, 2)))

        # dominant note
        if each['dominant_note'] == test_data['dominant_note']:
            print('dominant note: Matched')
        else:
            print('dominant note: Mis-matched')

        # beat tendency
        if len(each['beat_tendency']) > len(test_data['beat_tendency']):
            a = each['beat_tendency']
            b = test_data['beat_tendency']
        else:
            a = test_data['beat_tendency']
            b = each['beat_tendency']

        w_size = len(b)
        dist_list = list()
        for index in range(0, len(a) - len(b) + 1):
            distance, path = fastdtw(a[index:index + w_size], b, dist=euclidean)
            dist_list.append(round(distance, 2))

        print('beat tendency distance: {}(0.00+)'.format(min(dist_list)))
        print('----------------------------------------------------------------------')

