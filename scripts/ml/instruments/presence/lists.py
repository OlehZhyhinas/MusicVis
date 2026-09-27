"""Print id=path lists for a corpus/split (used to drive dsp_dump.ts and the other stages).
  python lists.py test|own-eval|own-train|fma [--max N]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common as C

# Owner tracks that duplicate a test song (same recording under another name): never train on them.
TEST_DUPES = ['saxobeat', 'waiting-for-love', 'ghosts-n-stuff', 'turn-down-for-what', 'thinking-out-loud', 'radioactive',
              'morenito', 'thrift-shop', 'lean-on', 'garrix-animals', 'fireflies', 'alors-on-danse', 'all-the-things-she-said']


def split(name):
    if name == 'test':
        return [('test', t, p) for t, p in C.tracks('test')]
    if name.startswith('own'):
        ev = name == 'own-eval'
        xs = [('own', t, p) for t, p in C.tracks('own') if C.is_eval('own', t, p) == ev]
        return xs if ev else [x for x in xs if not any(d in x[1] for d in TEST_DUPES)]
    if name == 'fma':
        return [('fma', t, p) for t, p in C.tracks('fma')]
    raise ValueError(name)

if __name__ == '__main__':
    xs = split(sys.argv[1])
    if '--max' in sys.argv:
        xs = xs[:int(sys.argv[sys.argv.index('--max') + 1])]
    for c, t, p in xs:
        print(f'{t}={p}')
