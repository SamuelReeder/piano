import pandas as pd

class Notes(object):
    def __init__(self):
        self.ntoi = {}
        self.iton = []

    def add_note(self, note):
        if note not in self.ntoi:
            self.iton.append(note)
            self.ntoi[note] = len(self.iton) - 1
        return self.ntoi[note]

    def __len__(self):
        return len(self.iton)

class Tokenizer(object):
    def __init__(self, base_path):
        self.notes = Notes()
        self.end_of_song_token = "<EOS>"
        self.base_path = base_path
        self.notes.add_note(self.end_of_song_token)

    def tokenize(self, path):
        assert os.path.exists(self.base_path + path)
        
        data = pd.read_csv(self.base_path + path)
        
        ids = []
        durations = []
        velocities = []
        
        for _, row in data.iterrows():
            ids.append(self.notes.add_note(row['note_name']))
            durations.append(row['duration'])
            velocities.append(row['velocity'])
        ids.append(self.notes.ntoi[self.end_of_song_token])
        durations.append(0.0) # EOS
        velocities.append(0.0) # EOS
        return ids, durations, velocities

    def tokenize_multiple_files(self, paths):
        combined_ids = []
        combined_durations = []
        combined_velocities = []
        for path in paths:
            if '.csv' not in path or 'midi_notes' in path:
                continue
            song_ids, durations, velocities = self.tokenize(path)
            combined_ids.extend(song_ids)
            combined_durations.extend(durations)
            combined_velocities.extend(velocities)
        return combined_ids, combined_durations, combined_velocities
    
    
    def decode(self, ids):
        notes = [self.notes.iton[i] for i in ids]
        return notes
