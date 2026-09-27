"""The generic instrument-activity layout (index -> group) and the AudioSet class sets per group.

The activity vector is hierarchical: parent groups (guitar, synth) are listed next to their
children so a preset gene can target either. Teacher sources per group:
  'stem'  demucs6 stem energy relative to the mix (the more reliable source)
  'stem+tag'  demucs6 stem energy, gated by AudioSet evidence (the guitar/piano stems bleed)
  'tag'   AudioSet (PANNs Cnn14) class probabilities only
  'heur'  heuristic split (synth lead/pad/pluck from the 'other' stem envelope statistics)
"""

GROUPS = [
    # name,            source,      AudioSet classes (display names) used by the tag / gate
    ('vocals',          'stem',     ['Singing', 'Male singing', 'Female singing', 'Child singing', 'Synthetic singing', 'Rapping',
                                     'Humming', 'Yodeling', 'Chant', 'Vocal music', 'A capella', 'Choir', 'Speech']),
    ('vocals_backing',  'tag',      ['Choir']),
    ('drums',           'stem',     ['Drum kit', 'Drum machine', 'Drum', 'Snare drum', 'Rimshot', 'Drum roll', 'Bass drum', 'Cymbal', 'Hi-hat', 'Percussion']),
    ('perc_hand',       'tag',      ['Clapping', 'Hands', 'Finger snapping', 'Tambourine', 'Maraca', 'Rattle (instrument)', 'Wood block', 'Cowbell', 'Tabla']),
    ('bass',            'stem',     ['Bass guitar', 'Double bass']),
    ('guitar',          'stem+tag', ['Guitar', 'Electric guitar', 'Acoustic guitar', 'Steel guitar, slide guitar', 'Tapping (guitar technique)', 'Strum',
                                     'Plucked string instrument', 'Banjo', 'Mandolin', 'Ukulele', 'Sitar']),
    ('guitar_electric', 'stem+tag', ['Electric guitar', 'Steel guitar, slide guitar', 'Tapping (guitar technique)']),
    ('guitar_acoustic', 'stem+tag', ['Acoustic guitar', 'Strum', 'Banjo', 'Mandolin', 'Ukulele']),
    ('keys_piano',      'stem+tag', ['Piano', 'Electric piano', 'Keyboard (musical)', 'Harpsichord']),
    ('organ',           'tag',      ['Organ', 'Electronic organ', 'Hammond organ', 'Accordion']),
    ('synth',           'tag',      ['Synthesizer', 'Sampler', 'Theremin']),
    ('synth_lead',      'heur',     []),
    ('synth_pad',       'heur',     []),
    ('synth_pluck',     'heur',     []),
    ('brass',           'tag',      ['Brass instrument', 'French horn', 'Trumpet', 'Trombone', 'Saxophone']),
    ('strings',         'tag',      ['Bowed string instrument', 'String section', 'Violin, fiddle', 'Pizzicato', 'Cello', 'Orchestra']),
    ('woodwinds',       'tag',      ['Wind instrument, woodwind instrument', 'Flute', 'Clarinet', 'Harmonica', 'Bagpipes', 'Didgeridoo', 'Shofar']),
    ('mallets_bells',   'tag',      ['Harp', 'Zither', 'Mallet percussion', 'Marimba, xylophone', 'Glockenspiel', 'Vibraphone', 'Steelpan', 'Bell',
                                     'Church bell', 'Jingle bell', 'Chime', 'Wind chime', 'Tubular bells', 'Tuning fork', 'Singing bowl']),
    ('fx',              'tag',      ['Whoosh, swoosh, swish', 'Sound effect', 'Noise', 'White noise', 'Pink noise', 'Zing', 'Sine wave', 'Chirp tone',
                                     'Scratching (performance technique)', 'Beep, bleep', 'Siren', 'Rumble']),
]
NAMES = [g[0] for g in GROUPS]
G = len(GROUPS)
ELECTRONIC = ['Electronic music', 'House music', 'Techno', 'Dubstep', 'Drum and bass', 'Electronica', 'Electronic dance music',
              'Trance music', 'Synthesizer', 'Sampler', 'Disco', 'Dance music']
STEMS6 = ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other']


def class_index(labels):
    """{group: [indices]} for a label list (AudioSet names; YAMNet lacks a few, which are skipped)."""
    pos = {l: i for i, l in enumerate(labels)}
    return {n: [pos[c] for c in cls if c in pos] for n, _, cls in GROUPS}
