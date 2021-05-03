import os
import webrtcvad

import numpy as np
import scipy.io.wavfile as wavfile
import glob
from gammatone.gtgram import gtgram


def filter_voice(signal, sample_rate, mode=3, threshold_voice=80):

    signal = np.array(signal, dtype=np.int16)
    signal = np.ascontiguousarray(signal)
    vad = webrtcvad.Vad(mode)
    frames = frame_generator(10, signal, sample_rate)
    frames = list(frames)

    if len(frames) == 0:
        return 0

    match = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            match += 1

    percentage_voice = match * 100 / len(frames)
    return percentage_voice > threshold_voice


def split_audio_chunks(audio_filename, size_chunks=500, overlap=500):
    """
    Split an binaural audio file into several chunks with overlap
    :param audio_filename: path to the audio file
    :param size_chunks: size of chunks in milliseconds
    :param overlap: overlap between consecutive chunks in milliseconds
    :return: list of chunks for each channels
    """

    fs, signal = wavfile.read(audio_filename)
    assert signal.shape[1] == 2

    signal1 = signal[:, 0]
    signal2 = signal[:, 1]

    length_chunk = int((fs * size_chunks) / 1000)
    overlap = int((fs * overlap) / 1000)

    index_start = 0
    chunk_signal1 = []
    chunk_signal2 = []

    stop_length = len(signal)
    while (index_start + length_chunk) < stop_length:

        stop_index = index_start + length_chunk
        chunck_s1 = signal1[index_start:stop_index]
        chunck_s2 = signal2[index_start:stop_index]

        if len(chunck_s1) == len(chunck_s2) == length_chunk:
            chunk_signal1.append(chunck_s1)
            chunk_signal2.append(chunck_s2)
        else:
            break
        index_start += overlap

    return fs, chunk_signal1, chunk_signal2


def save_gammatone(list_audio, output_dir, fs):
    """
    Save a list of chunks audio as gammagram
    :param list_audio: list of audio chunks
    :param output_dir: path of the output_dir
    :param fs: sampling rate
    :return:
    """

    for i, f in enumerate(list_audio):
        waveform = gtgram(f, fs, window_time=0.04, hop_time=0.02,
                          channels=128, f_min=120)
        wave_tmp = np.expand_dims(waveform, axis=0)
        filename = os.path.join(output_dir, f"chunk_{i}.npy")
        np.save(filename, wave_tmp)


def save_audio(list_audio, output_dir, fs):
    """
    Save list of chunks with fs sampling rate in output_dir
    :param list_audio: list of chunks audio
    :param output_dir:  path of the output_dir
    :param fs: sampling rate
    :return:
    """
    nb_keep = 0
    for i, sig in enumerate(list_audio):
        if filter_voice(sig, fs):
            filename = os.path.join(output_dir, f"chunk_{i}.wav")
            wavfile.write(filename, fs, sig.astype(np.int16))
            nb_keep += 1

    print(f"Saved {nb_keep} audio files discarded {len(list_audio) - nb_keep}")


def process_dataset(data_dir, output_dir, length_chunk, overlap, mean=False):
    """

    :param data_dir: Root dir of the dataset of speakers
    :param output_dir: Path to the output_dir
    :param length_chunk: Desired chunk length of audio
    :param overlap: Overlap between sucessive chunks
    :param mean: Avegrage the channels
    :return:
    """

    # Process train folder
    dirs = os.listdir(data_dir)
    os.mkdir(output_dir)

    for d in dirs:
        audio_files = glob.glob(os.path.join(data_dir, d, "*.wav"))
        os.mkdir(os.path.join(output_dir, d))
        chunks = []
        fs = 0
        for f in audio_files:
            fs, chunk_c1, chunk_c2 = split_audio_chunks(f, length_chunk, overlap)
            if mean:
                chunks = np.array(chunk_c1) + np.array(chunk_c2) / 2.0
            else:
                chunks += chunk_c1 + chunk_c2

        save_audio(chunks, os.path.join(output_dir, d), fs)


#----------------------------------------------------------- VAD CLASS ------------------------------------------------

class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio) -1:
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])