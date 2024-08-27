from pydub import AudioSegment

def play_wav_files(file_paths, output_path, volumes=None, durations=None):
    if volumes is None:
        volumes = [0.5] * len(file_paths)
    if durations is None:
        durations = [500] * len(file_paths)

    combined_audio = AudioSegment.silent(duration=500)

    for i, file_path in enumerate(file_paths):
        if file_path == '<EOS>':
            combined_audio += AudioSegment.silent(duration=1000)
            continue

        audio = AudioSegment.from_wav(f'/workspace/piano_notes/{file_path}.wav')
        audio = audio + (volumes[i] - 1) * 10  # adjusting volume based on a scale
        audio = audio[:durations[i]]

        combined_audio += audio

    combined_audio.export(output_path, format="wav")