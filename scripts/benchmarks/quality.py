"""Reference-based translation overlap, using the pinned canonical implementation."""

def chrf_quality(rows):
    if not rows: return None
    import sacrebleu
    from sacrebleu.metrics import CHRF
    metric = CHRF(char_order=6, word_order=0, beta=2)
    references = [[row["references"][index] if index < len(row["references"]) else row["references"][0] for row in rows]
                  for index in range(max(len(row["references"]) for row in rows))]
    score = metric.corpus_score([row["result"]["translation"] for row in rows], references)
    return {"score": score.score, "signature": str(metric.get_signature()), "sacrebleuVersion": sacrebleu.__version__,
            "scoredAttempts": len(rows), "conditionalOn": "completed outputs including empty_output; exceptions and timeouts are reported separately", "independentUnits": len({row["groupID"] for row in rows}),
            "interpretation": "character overlap on a 0-100 scale, not human semantic adequacy"}


def paired_quality(left, right):
    identity = lambda row: (row["caseID"], row["attemptIndex"])
    a, b = {identity(row): row for row in left}, {identity(row): row for row in right}
    speech_pairs = [(a[key], b[key]) for key in a.keys() & b.keys() if a[key].get("speechQuality") and b[key].get("speechQuality")]
    translation_pairs = [(a[key], b[key]) for key in a.keys() & b.keys()
                         if a[key].get("references") and b[key].get("references")
                         and a[key].get("lane") != "speech"
                         and a[key]["result"]["translationStatus"] in {"success", "empty_output"}
                         and b[key]["result"]["translationStatus"] in {"success", "empty_output"}]
    speech = None
    if speech_pairs:
        denominator = sum(x[0]["speechQuality"]["referenceUnits"] for x in speech_pairs)
        if any(x[0]["speechQuality"]["referenceUnits"] != x[1]["speechQuality"]["referenceUnits"] for x in speech_pairs):
            raise ValueError("Paired ASR references differ")
        rates = [sum(pair[index]["speechQuality"]["edits"] for pair in speech_pairs) / denominator if denominator else None for index in (0, 1)]
        speech = {"pairedAttempts": len(speech_pairs), "referenceUnits": denominator, "baselineErrorRate": rates[0],
                  "candidateErrorRate": rates[1], "delta": rates[1] - rates[0] if denominator else None}
    translation = None
    if translation_pairs:
        scores = [chrf_quality([pair[index] for pair in translation_pairs]) for index in (0, 1)]
        translation = {"pairedAttempts": len(translation_pairs), "baselineChrF2": scores[0]["score"],
                       "candidateChrF2": scores[1]["score"], "delta": scores[1]["score"] - scores[0]["score"],
                       "signature": scores[0]["signature"]}
    return {"speech": speech, "translation": translation,
            "plannedPairedAttempts": len(a.keys() & b.keys()),
            "note": "Quality uses only identically paired scored attempts. Read unmatched failures and coverage alongside these scores."}
