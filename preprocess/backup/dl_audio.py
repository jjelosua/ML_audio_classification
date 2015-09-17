import os
import subprocess
import pandas as pd

data = pd.read_csv('labeled_audios_urls.csv')

paths = []
for i, url in enumerate(data['Document Url'].values):
    print('------', i, '------')

    fname = os.path.basename(url)
    mp3_path = os.path.join('mp3s/labeled/', fname)
    wav_path = os.path.join('wavs/labeled/', fname.replace('mp3', 'wav'))
    subprocess.call(['wget', url, '-O', mp3_path])

    # y: overwrite w/o asking (there are a few duplicates)
    # ar: sample rate: 8000
    # ac: channels: 1 (mono)
    subprocess.call(['ffmpeg', '-y', '-i', mp3_path, '-ar', '8000', '-ac', '1', wav_path])

    paths.append({
        'mp3_path': mp3_path,
        'wav_path': wav_path
    })

paths_df = pd.DataFrame(paths)
data = pd.concat([data, paths_df], axis=1)
data.to_csv('labeled_audios_urls_with_file_paths.csv')