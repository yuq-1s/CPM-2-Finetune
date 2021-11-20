import json

def _load(path):
    with open(path) as f:
        yield '['
        for line in f:
            if line == '}\n':
                yield '},'
            else:
                yield line

foo = json.loads(''.join(_load('data/Math_23K.json'))[:-1] + ']')
with open('data/math23k/train.json', 'w') as f:
    json.dump([{'text': x['original_text'], 'equation': x['equation']} for x in foo], f, ensure_ascii=False)
