#!/usr/bin/env python3
"""Descriptive local screens and paired utterance-level timing bootstrap."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'runs' / (name + '.json')).read_text())


def score(name):
    rows = load(name)['attempts']
    passed = [x for x in rows if x['outcome'] == 'passed']
    return dict(attempts=len(rows), passed=len(passed), failures=len(rows) - len(passed),
                corpusCER=sum(x['characterEdits'] for x in rows) / sum(x['referenceCharacters'] for x in rows))


def warm(name):
    rows = load(name)['attempts']
    groups = {}
    for x in rows:
        if x.get('modelWasAlreadyLoaded') and x['outcome'] == 'passed':
            groups.setdefault(x['caseID'], []).append(x['result'])
    units = {key: {field: statistics.mean(x[field] for x in values)
                   for field in ['first_text_seconds', 'compute_rtf', 'post_audio_finalization_seconds']}
             for key, values in groups.items()}
    result = dict(totalAttempts=len(rows), totalFailures=sum(x['outcome'] != 'passed' for x in rows),
                  successfulUtterances=len(units), timingUnit='utterance mean of two loaded-model repeats',
                  conditionalOnSuccessfulNonemptyFinal=True,
                  units=units)
    for field in ['first_text_seconds', 'compute_rtf', 'post_audio_finalization_seconds']:
        values = sorted(x[field] for x in units.values())
        result[field] = dict(mean=statistics.mean(values), p50=statistics.median(values),
                             p95=values[math.ceil(len(values) * .95) - 1])
    result['maximumQueuedAudioSeconds'] = max(x['result']['maximum_queued_samples'] / 16000 for x in rows if 'result' in x)
    detected = [x['result']['first_text_seconds'] - x['result']['first_vad']['observedAudioSeconds']
                for x in rows if x.get('modelWasAlreadyLoaded') and 'first_vad' in x['result']]
    if detected:
        result['afterNativeVadObservationSeconds'] = dict(p50=statistics.median(detected), maximum=max(detected),
            note='After causal detector observation, not annotated acoustic speech onset; first text may be wrong.')
    return result


def main():
    report = dict(schemaVersion=1, analysisSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  selection='Exploratory source/conversion/configuration screen; no held-out or population accuracy claim.')
    report['quality'] = {name: score(name) for name in [
        'mimi-moonshine-direct-v2', 'parakeet-bf16-silero-direct-v1',
        'parakeet-silero-direct-v1', 'parakeet-silero-lookbehind2s-direct-v2',
        'parakeet-encoder-q4-silero-direct-v3']}
    report['moonshineWarm'] = warm('mimi-moonshine-paced-repeat-v2')
    report['parakeetWarm'] = warm('parakeet-encoder-q4-prewarm-paced-repeat-final-v5')
    report['phononEnglishWarm'] = warm('mimi-phonon-paced-repeat-v2')
    report['phononEnglishWarm']['comparisonLimit'] = 'Different corpus/language; four of eight utterances produced blank finals in all three trials. Conditional timings are not overall system performance.'
    a = report['moonshineWarm']['units']
    b = report['parakeetWarm']['units']
    ids = sorted(set(a) & set(b))
    differences = [b[key]['first_text_seconds'] - a[key]['first_text_seconds'] for key in ids]
    rng = random.Random(42)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    report['pairedJapaneseFirstTextDifferenceSeconds'] = dict(units=len(ids), parakeetMinusMoonshineMean=statistics.mean(differences),
        bootstrap95PercentCI=[samples[249], samples[9749]], method='Paired bootstrap of eight utterances; each value averages its two loaded-model trials. Descriptive selected-screen comparison.')
    report['payloadBytes'] = dict(bf16Source=1245321423, allModuleQ4=474668178, encoderOnlyQ4=481857364)
    (ROOT / 'runs' / 'summary-final-v5.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['moonshineWarm','parakeetWarm','phononEnglishWarm']},indent=2))


if __name__ == '__main__':
    main()
