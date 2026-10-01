#!/usr/bin/env python3
"""Production audio-queue replay; retain every attempt and artifact identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import unicodedata


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if unicodedata.category(c)[0] not in 'PSZC')


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = row
    return previous[-1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('suite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', choices=['moonshine', 'phonon', 'parakeet'], required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--unpaced', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.suite.read_text().splitlines() if line]
    if args.subset:
        rows = [rows[i] for i in range(0, 24, 3)]
    if args.limit:
        rows = rows[:args.limit]
    report = dict(schemaVersion=1, candidate=args.candidate,
                  executableSHA256=digest(args.executable), suiteSHA256=digest(args.suite),
                  harnessSHA256=digest(Path(__file__)),
                  executableSourceIdentity=subprocess.check_output([str(args.executable.resolve()), "--print-adaptive-segmentation-build-identity"], text=True).strip(),
                  audioDeliveryBoundary='end of each 50ms buffer',
                  mode='direct' if args.unpaced else 'paced', attempts=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flag = {'moonshine': '--smoke-moonshine-ja-live', 'phonon': '--smoke-phonon2-live', 'parakeet': '--smoke-parakeet-ja-live'}[args.candidate]
    temporary = args.output.with_suffix('.current.json')
    for row in rows:
        audio = (args.suite.parent / row['audio']).resolve()
        actual_hash = digest(audio)
        if actual_hash != row['audioSha256']:
            raise RuntimeError(f"Audio hash changed: {row['caseID']}")
        reference = row['reference']
        common = dict(caseID=row['caseID'], reference=reference, audioSHA256=actual_hash,
                      timeoutSeconds=120, modelInstanceReusedWithinUtterance=True,
                      protectedTerms=row.get("protectedTerms", []), protectedTermAliases=row.get("protectedTermAliases", {}))
        if temporary.exists():
            temporary.unlink()
        command = [str(args.executable.resolve()), flag, str(args.model.resolve()),
                   '--audio', str(audio), '--output', str(temporary.resolve()),
                   '--warm-runs', str(args.repeats)]
        if args.unpaced:
            command.append('--unpaced')
        started = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            common.update(exitCode=completed.returncode, stderr=completed.stderr, stdout=completed.stdout)
            if temporary.exists():
                container = json.loads(temporary.read_text())
                results = container.get('attempts', [container])
                for repeat, result in enumerate(results):
                    attempt = dict(common, repeat=repeat, modelWasAlreadyLoaded=repeat > 0, result=result)
                    hypothesis = normalize(result['text'])
                    normalized_reference = normalize(reference)
                    attempt.update(characterEdits=distance(hypothesis, normalized_reference),
                                   referenceCharacters=len(normalized_reference))
                    attempt['outcome'] = ('passed' if hypothesis and not result.get('warnings')
                                          else 'runtime-or-quality-failure')
                    report['attempts'].append(attempt)
            else:
                report['attempts'].append(dict(common, outcome='missing-report'))
        except subprocess.TimeoutExpired:
            report['attempts'].append(dict(common, outcome='timeout', elapsedSeconds=120))
        report['attempts'][-1]['processElapsedSeconds'] = time.monotonic() - started
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(row['caseID'], report['attempts'][-1]['outcome'], flush=True)
    if temporary.exists():
        temporary.unlink()


if __name__ == '__main__':
    main()
