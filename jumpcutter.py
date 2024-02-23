#!/usr/bin/env python
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
import soundfile as sf
from shutil import rmtree
from tqdm import tqdm as std_tqdm
from functools import partial
from multiprocessing import Pool, Array
import numpy as np
import subprocess
import argparse
import re
import math
import os
import time

FFMPEG_PATH = 'ffmpeg'

tqdm = partial(std_tqdm,
               bar_format=('{desc:<22} {percentage:3.0f}%'
                           '|{bar:10}|'
                           ' {n_fmt:>6}/{total_fmt:>6} [{elapsed:^5}<{remaining:^5}, {rate_fmt}{postfix}]'))
# tqdm = std_tqdm


def _get_max_volume(s):
    return max(-np.min(s), np.max(s))


def _is_valid_input_file(filename) -> bool:
    """
    Check wether the input file is one that ffprobe recognizes, i.e. a video / audio / ... file.
    If it does, check whether there exists an audio stream, as we could not perform the dynamic shortening without one.

    :param filename: The full path to the input that is to be checked
    :return: True if it is a file with an audio stream attached.
    """

    command = [
        'ffprobe', '-i', filename, '-hide_banner', '-loglevel', 'error',
        '-select_streams', 'a', '-show_entries', 'stream=codec_type'
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outs, errs = None, None
    try:
        outs, errs = p.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        p.kill()
        outs, errs = p.communicate()
    finally:
        # If the file is no file that ffprobe recognizes we will get an error in the errors
        # else wise we will obtain an output in outs if there exists at least one audio stream
        return len(errs) == 0 and len(outs) > 0


def _input_to_output_filename(filename):
    dot_index = filename.rfind(".")
    return filename[:dot_index] + "_ALTERED" + filename[dot_index:]


def _create_path(s):
    # assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."
    try:
        os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory failed." \
                      " (The TEMP folder may already exist. Delete or rename it, and try again.)"


def _delete_path(s):  # Dangerous! Watch out!
    try:
        rmtree(s, ignore_errors=False)
        for i in range(5):
            if not os.path.exists(s):
                return
            time.sleep(0.01 * i)
    except OSError:
        print('Deletion of the directory {} failed'.format(s))
        print(OSError)


# TODO maybe transition to use the time=... instead of frame=... as frame is not accessible when exporting audio only
def _run_timed_ffmpeg_command(command, **kwargs):
    p = subprocess.Popen([FFMPEG_PATH, *command], stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)

    with tqdm(**kwargs) as t:
        while p.poll() is None:
            line = p.stderr.readline()
            m = re.search(r'frame=.*?(\d+)', line)
            if m is not None:
                new_frame = int(m.group(1))
                if t.total < new_frame:
                    t.total = new_frame
                t.update(new_frame - t.n)
        t.update(t.total - t.n)


def _get_tree_expression(chunks) -> str:
    return '{}/TB/FR'.format(_get_tree_expression_rec(chunks))


def _get_tree_expression_rec(chunks) -> str:
    """
    Build a 'Binary Expression Tree' for the ffmpeg pts selection

    :param chunks: List of chunks that have the format [oldStart, oldEnd, newStart, newEnd]
    :return: Binary tree expression to calculate the speedup for the given chunks
    """
    if len(chunks) > 1:
        split_index = int(len(chunks) / 2)
        center = chunks[split_index]
        return 'if(lt(N,{}),{},{})'.format(center[0],
                                           _get_tree_expression_rec(chunks[:split_index]),
                                           _get_tree_expression_rec(chunks[split_index:]))
    else:
        chunk = chunks[0]
        local_speedup = (chunk[3] - chunk[2]) / (chunk[1] - chunk[0])
        offset = - chunk[0] * local_speedup + chunk[2]
        return 'N*{}{:+}'.format(local_speedup, offset)

def _get_shared_audio_data():
    global shared_audio_buffer
    return np.frombuffer(shared_audio_buffer[0], dtype=shared_audio_buffer[1]).reshape(shared_audio_buffer[2])

def _init_shared_audio_data(audio):
    global shared_audio_buffer
    
    shared_audio_buffer = (
        Array(np.ctypeslib.as_ctypes_type(audio.dtype), audio.shape[0] * audio.shape[1], lock=False),
        audio.dtype,
        audio.shape
    )

    audio_buffer = _get_shared_audio_data()

    np.copyto(audio_buffer, audio)
    return audio_buffer

def _preprocess_filter_vocals(chunk):
    import librosa

    audio_buf = librosa.util.buf_to_float(_get_shared_audio_data()[chunk[0]:chunk[0]+chunk[1], 0].copy(order='C'), n_bytes=2)
    # audio_buf = librosa.to_mono(audio_buf)

    S_full, phase = librosa.magphase(librosa.stft(audio_buf))

    S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=chunk[2])))
    
    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_foreground = mask_v * S_full
    # S_background = mask_i * S_full

    audio_buf_filtered = librosa.istft(S_foreground * phase)

    return audio_buf_filtered

def _process_chunk(chunk):
        samples_per_frame, new_speeds, audio_fade_envelope_size = chunk[3:6]
        audio_chunk = _get_shared_audio_data()[int(chunk[0] * samples_per_frame):int(chunk[1] * samples_per_frame)]

        reader = ArrayReader(np.transpose(audio_chunk))
        writer = ArrayWriter(reader.channels)
        tsm = phasevocoder(reader.channels, speed=new_speeds[int(chunk[2])])
        tsm.run(reader, writer)
        altered_audio_data = np.transpose(writer.data)

        # smooth out transition's audio by quickly fading in/out
        if altered_audio_data.shape[0] < audio_fade_envelope_size:
            altered_audio_data[:] = 0  # audio is less than 0.01 sec, let's just remove it.
        else:
            premask = np.arange(audio_fade_envelope_size) / audio_fade_envelope_size
            mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
            altered_audio_data[:audio_fade_envelope_size] *= mask
            altered_audio_data[-audio_fade_envelope_size:] *= 1 - mask
        
        return altered_audio_data

def speed_up_video(
        input_file: str,
        output_file: str = None,
        frame_rate: float = None,
        sample_rate: int = 48000,
        enable_librosa: bool = False,
        silent_speed: float = 5.0,
        sounded_speed: float = 1.0,
        threshold_method: str = None,
        silent_threshold: float = 0.03,
        webrtc_mode: int = 1,
        frame_spreadage: int = 1,
        audio_fade_envelope_size: int = 400,
        librosa_preprocess: list = [],
        threads_num: int = None,
        temp_folder: str = 'TEMP') -> None:
    """
    Speeds up a video file with different speeds for the silent and loud sections in the video.

    :param input_file: The file name of the video to be sped up.
    :param output_file: The file name of the output file. If not given will be 'input_file'_ALTERED.ext.
    :param frame_rate: The frame rate of the given video. Only needed if not extractable through ffmpeg.
    :param sample_rate: The sample rate of the audio in the video.
    :param threshold_method: The method used for separating sounded and silent frames
    :param silent_threshold: The threshold when a chunk counts towards being a silent chunk.
                             Value ranges from 0 (nothing) - 1 (max volume).
    :param webrtc_mode: mode used for the Webrtc VAD, sets how aggressivelly it removes non-voice audio
    :param silent_speed: The speed of the silent chunks.
    :param sounded_speed: The speed of the loud chunks.
    :param frame_spreadage: How many silent frames adjacent to sounded frames should be included to provide context.
    :param audio_fade_envelope_size: Audio transition smoothing duration in samples.
    :param threads_num: Number of threads used for phasevocoding step. By default equal to the number of hardware threads.
    :param temp_folder: The file path of the temporary working folder.
    """
    # Set output file name based on input file name if none was given
    if output_file is None:
        output_file = _input_to_output_filename(input_file)

    # Create Temp Folder
    if os.path.exists(temp_folder):
        _delete_path(temp_folder)
    _create_path(temp_folder)

    # Find out framerate and duration of the input video
    command = [
        'ffprobe', '-i', input_file, '-hide_banner', '-loglevel', 'error',
        '-select_streams', 'v', '-show_entries', 'format=duration:stream=avg_frame_rate'
    ]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    std_out, err = p.communicate()
    match_frame_rate = re.search(r'frame_rate=(\d*)/(\d*)', str(std_out))
    if match_frame_rate is not None:
        frame_rate = float(match_frame_rate.group(1)) / float(match_frame_rate.group(2))
        # print(f'Found Framerate {frame_rate}')
    
    if frame_rate is None:
        print("\033[93;40mWARNING\033[0m: Failed to detect frame rate and --frame_rate argument was not passed, assuming 30 fps. If your output has desynced audio, this is why!!!")
        frame_rate = 30

    match_duration = re.search(r'duration=([\d.]*)', str(std_out))
    original_duration = 0.0
    if match_duration is not None:
        original_duration = float(match_duration.group(1))
        # print(f'Found Duration {original_duration}')

    # Extract the audio
    command = [
        '-i', input_file,
        '-ab', '160k',
        '-ac', '2',
        '-ar', str(sample_rate),
        '-vn', temp_folder + '/audio.wav',
        '-hide_banner'
    ]

    _run_timed_ffmpeg_command(command, total=int(original_duration * frame_rate), unit='frames',
                              desc='Extracting audio:')

    audio_data, wav_sample_rate = sf.read(temp_folder + "/audio.wav", dtype='int16')

    audio_data = _init_shared_audio_data(audio_data)

    audio_sample_count = audio_data.shape[0]
    max_audio_volume = _get_max_volume(audio_data)
    samples_per_frame = wav_sample_rate / frame_rate
    audio_frame_count = int(math.ceil(audio_sample_count / samples_per_frame))

    # Preprocess using librosa

    preprocessed_audio_buffer = None

    if enable_librosa:
        try:
            import librosa
        except ImportError:
            print("Enabled librosa advanced audio processing without installing the required librosa module!")
            raise

        p = Pool(threads_num)

        preprocessed_audio_buffer = np.zeros((audio_data.shape[0],), dtype=audio_data.dtype)
        preprocessed_pointer = 0
        
        chunk_size = 10 * sample_rate # 10s of audio

        if "filter-vocals" in librosa_preprocess:
            chunk_begin = 0
            in_chunks = []

            while chunk_begin < audio_sample_count:
                in_chunks.append((chunk_begin, chunk_size, sample_rate))
                
                chunk_begin += chunk_size

            for chunk in tqdm(p.imap(_preprocess_filter_vocals, in_chunks), desc="Filtering Vocals:", unit="chunks", total=len(in_chunks)):
                preprocessed_audio_buffer[preprocessed_pointer:preprocessed_pointer + chunk.shape[0]] = chunk * np.iinfo(preprocessed_audio_buffer.dtype).max
                preprocessed_pointer += chunk.shape[0]
    else:
        preprocessed_audio_buffer = audio_data

    # Find frames with loud audio
    has_loud_audio = np.zeros(audio_frame_count, dtype=bool)

    if threshold_method == None:
        threshold_method = "silence"

    threshold_methods = {
        "silence": 0,
        "webrtc-voice": 1
    }

    threshold_index = threshold_methods[threshold_method]

    webvad = None
    webvad_length = 0
    if threshold_index == 1:
        try:
            import webrtcvad
        except ImportError:
            print("Selected webrtc-voice as a threshold method without installing the required webrtcvad module!")
            quit(-1)

        webvad = webrtcvad.Vad(webrtc_mode)
        
        # webrtc vad only supports 10, 20 or 30 ms audio chunks so we need to pick the biggest audio size which fits inside our audio frame
        if samples_per_frame > int(sample_rate * 0.03):
            webvad_length = int(sample_rate * 0.03)
        elif samples_per_frame > int(sample_rate * 0.02):
            webvad_length = int(sample_rate * 0.02)
        elif samples_per_frame > int(sample_rate * 0.01):
            webvad_length = int(sample_rate * 0.01)
        else:
            print("Too high frame rate for webrtc to handle! frame time must be at least 10 ms.")
            quit(-1)

    for i in tqdm(range(audio_frame_count), desc="Thresholding audio:", unit='frames'):
        start = int(i * samples_per_frame)
        end = min(int((i + 1) * samples_per_frame), audio_sample_count)
        audio_chunk = preprocessed_audio_buffer[start:end]

        if threshold_index == 0: # silence method
            chunk_max_volume = float(_get_max_volume(audio_chunk)) / max_audio_volume
            has_loud_audio[i] = chunk_max_volume >= silent_threshold
        elif threshold_index == 1: # webrtc-voice method
            try:
                if len(audio_chunk) < webvad_length: # end of audio file, not enough samples, skip as silent
                    has_loud_audio[i] = False
                    continue

                # note: webrtc only accepts mono, only scans for left channel
                has_loud_audio[i] = webvad.is_speech(audio_chunk.tobytes(order='C'), sample_rate, webvad_length)

            except:
                print("\033[93;40mWARNING\033[0m: Webrtc Vad failed! Webrtc only supports 8000, 16000, 32000 or 48000 Hz sample rate, make sure you set your sample_rate argument accordingly.")
                raise

    del preprocessed_audio_buffer

    # Chunk the frames together that are quiet or loud
    new_speeds = [silent_speed, sounded_speed]
    chunks = [[0, 0, 0]]
    should_include_frame = np.zeros(audio_frame_count, dtype=bool)
    for i in tqdm(range(audio_frame_count), desc='Finding chunks:', unit='frames'):
        start = int(max(0, i - frame_spreadage))
        end = int(min(audio_frame_count, i + 1 + frame_spreadage))
        should_include_frame[i] = np.any(has_loud_audio[start:end])
        if i >= 1 and should_include_frame[i] != should_include_frame[i - 1]:  # Did we flip?
            chunks.append([chunks[-1][1], i, should_include_frame[i - 1], samples_per_frame, new_speeds, audio_fade_envelope_size])

    chunks.append([chunks[-1][1], audio_frame_count, should_include_frame[audio_frame_count - 1], samples_per_frame, new_speeds, audio_fade_envelope_size])
    chunks = chunks[1:]

    # Generate audio data with varying speed for each chunk
    output_audio_data = np.zeros(audio_data.shape, dtype=audio_data.dtype)
    output_pointer = 0

    p = Pool(threads_num)
    processed_chunks = p.imap(_process_chunk, chunks, chunksize=64)

    for index, altered_audio_data in tqdm(enumerate(processed_chunks), total=len(chunks), desc='Phasevocoding chunks:', unit='chunks'):
        end_pointer = output_pointer + altered_audio_data.shape[0]

        output_audio_data[output_pointer:end_pointer] = altered_audio_data # / max_audio_volume

        start_output_frame = int(math.ceil(output_pointer / samples_per_frame))
        end_output_frame = int(math.ceil(end_pointer / samples_per_frame))
        chunks[index] = chunks[index][:2] + [start_output_frame, end_output_frame]

        output_pointer = end_pointer

    # Postprocess using librosa

    if enable_librosa:
        pass

    sf.write(temp_folder + "/audioNew.wav", output_audio_data, sample_rate)

    del output_audio_data
    del p

    # Cut the video parts to length
    expression = _get_tree_expression(chunks)

    filter_graph_file = open(temp_folder + "/filterGraph.txt", 'w')
    filter_graph_file.write(f'fps=fps={frame_rate},setpts=')
    filter_graph_file.write(expression.replace(',', '\\,'))
    filter_graph_file.close()

    command = [
        '-i', input_file,
        '-i', temp_folder + '/audioNew.wav',
        '-filter_script:v', temp_folder + '/filterGraph.txt',
        '-map', '0',
        '-map', '-0:a',
        '-map', '1:a',
        '-c:a', 'aac',
        '-t', str(chunks[-1][3] / frame_rate), # fix incorrect video duration
        output_file,
        '-loglevel', 'warning',
        '-stats',
        '-y',
        '-hide_banner'
    ]

    _run_timed_ffmpeg_command(command, total=chunks[-1][3], unit='frames', desc='Generating final:')

    _delete_path(temp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modifies a video file to play at different speeds when there is sound vs. silence.')

    parser.add_argument('-i', '--input_file', type=str, dest='input_file', nargs='+', required=True,
                        help='The video file(s) you want modified.'
                             ' Can be one or more directories and / or single files.')
    parser.add_argument('-o', '--output_file', type=str, dest='output_file',
                        help="The output file. Only usable if a single file is given."
                             " If not included, it'll just modify the input file name by adding _ALTERED.")
    parser.add_argument('-j', '--jobs', type=int, dest='threads_num',
                        help="Number of threads used for phasevocoding the audio."
                        " By default equal to the number of hardware threads in your computer.")

    common_group = parser.add_argument_group('common audio options')

    common_group.add_argument('-S', '--sounded_speed', type=float, dest='sounded_speed',
                        help="The speed that sounded (spoken) frames should be played at. Defaults to 1.")
    common_group.add_argument('-s', '--silent_speed', type=float, dest='silent_speed',
                        help="The speed that silent frames should be played at. Defaults to 5")
    common_group.add_argument('-fm', '--frame_margin', type=float, dest='frame_spreadage',
                        help="Some silent frames adjacent to sounded frames are included to provide context."
                             " This is how many frames on either the side of speech should be included. Defaults to 1")
    common_group.add_argument('-sr', '--sample_rate', type=float, dest='sample_rate',
                        help="Sample rate of the input and output videos. FFmpeg tries to extract this information."
                             " Thus only needed if FFmpeg fails to do so.")
    common_group.add_argument('-fr', '--frame_rate', type=float, dest='frame_rate',
                        help="Frame rate of the input and output videos. FFmpeg tries to extract this information."
                             " Thus only needed if FFmpeg fails to do so.")
    common_group.add_argument('-r', '--librosa', dest='enable_librosa', action='store_true',
                        help="Enables advanced librosa based audio processing."
                        " See the librosa help section for librosa specific options. (requires librosa module)")
    
    threshold_arguments = parser.add_argument_group('audio thresholding options')

    threshold_arguments.add_argument('-tm', '--threshold_method', type=str, dest='threshold_method',
                        help="Thresholding method used for determining if an audio is sounded or silent."
                        " Currently support \"silence\" (default) and \"webrtc-voice\" (requires webrtcvad module) modes.")
    threshold_arguments.add_argument('-t', '--silent_threshold', type=float, dest='silent_threshold',
                        help='The volume amount that frames\' audio needs to surpass to be consider "sounded".'
                             ' It ranges from 0 (silence) to 1 (max volume). Defaults to 0.03 (silence method only)')
    threshold_arguments.add_argument('-va', '--voice-aggressiveness-mode', type=int, dest='webrtc_mode',
                        help="Voice Activity Detection aggresiveness mode. Sets how aggresive is the voice detector about determining if a voice is present."
                        " Must be between 0 and 3. 0 is least aggresive about filtering non-speech, 3 is the most aggresive. Defaults to 1. (webrtc-voice method only)")
    
    librosa_group = parser.add_argument_group('librosa options')

    librosa_group.add_argument('-pre', '--pre_process', type=str, action='append', dest='librosa_preprocess',
                        help="Set what preprocessing stages are enabled before thresholding audio."
                        " Currently supported stage is only \"filter-vocals\" and \"enhance-vocals\"")

    files = []
    for input_file in parser.parse_args().input_file:
        if os.path.isfile(input_file):
            files += [os.path.abspath(input_file)]
        elif os.path.isdir(input_file):
            files += [os.path.join(input_file, file) for file in os.listdir(input_file)]

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    del args['input_file']
    if len(files) > 1 and 'output_file' in args:
        del args['output_file']

    # It appears as though nested progress bars are deeply broken
    # with tqdm(files, unit='file') as progress_bar:
    for index, file in enumerate(files):
        if not _is_valid_input_file(file):
            print(f"Skipping file {index + 1}/{len(files)} '{os.path.basename(file)}' as it is not a valid input file.")
            continue
        # progress_bar.set_description("Processing file '{}'".format(os.path.basename(file)))
        print(f"Processing file {index + 1}/{len(files)} '{os.path.basename(file)}'")
        local_options = dict(args)
        local_options['input_file'] = file
        speed_up_video(**local_options)
