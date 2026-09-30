#!/usr/bin/env python3
"""Fast size, model inventory and pinned-manifest rejection contracts."""

import tempfile
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

import verify_shipped_translation_pack as verifier


class AppPayloadBudgetTests(unittest.TestCase):
    def test_stable_and_development_verified_models_have_separate_budgets(self):
        verifier.verify_app_size(659_115_028, {"mimi-phonon2": 424_191_836})
        verifier.verify_app_size(1_150_000_000, {
            "mimi-phonon2": 424_191_836,
            "mimi-whisper-large-v3-turbo-q4": 468_150_715,
        })

    def test_total_core_and_per_model_ceilings_are_enforced(self):
        cases = [
            (750_000_001, {"mimi-phonon2": 424_191_836}),
            (1_250_000_001, {"mimi-phonon2": 450_000_000,
                              "mimi-whisper-large-v3-turbo-q4": 450_000_000}),
            (600_000_000, {"mimi-phonon2": 100_000_000}),
            (600_000_000, {"mimi-phonon2": 500_000_000}),
            (1, {"mimi-phonon2": 2}),
        ]
        for app_bytes, speech_bytes in cases:
            with self.subTest(app_bytes=app_bytes), self.assertRaises(SystemExit):
                verifier.verify_app_size(app_bytes, speech_bytes)

    def test_unknown_models_cannot_be_excluded_from_core_size(self):
        with self.assertRaises(SystemExit):
            verifier.verify_app_size(600_000_000, {"unknown-model": 400_000_000})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unknown-model").mkdir()
            with self.assertRaises(SystemExit):
                verifier.verified_speech_pack_bytes(root)

    def test_integrity_failure_cannot_grant_extra_app_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mimi-phonon2").mkdir()
            with mock.patch.dict(verifier.SPEECH_PACK_VERIFIERS, {
                "mimi-phonon2": mock.Mock(side_effect=SystemExit("digest mismatch")),
            }), self.assertRaises(SystemExit):
                verifier.verified_speech_pack_bytes(root)

    def test_phonon_manifest_cannot_redeclare_its_own_weight_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            with self.assertRaises(SystemExit):
                verifier.verify_phonon2_pack.verify(root)

    def test_phonon_weight_mutation_is_rejected_after_manifest_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"tampered")
            (root / "config.json").write_bytes(b"{}")
            manifest = {
                "format": "mimi-phonon2-two-plane-v1",
                "repository": "FermionResearch/Phonon-2",
                "revision": "1c388bcec35d19740bf36b0b675718223fa7904e",
                "license": "CC-BY-4.0", "languages": ["en"],
                "weight_equivalence": {"modules": 264, "max_absolute_error": 0.0},
                "modules": [{"path": str(index)} for index in range(264)],
                "files": {
                    "model.safetensors": {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()},
                    "config.json": {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
                },
            }
            encoded = json.dumps(manifest).encode()
            (root / "manifest.json").write_bytes(encoded)
            with mock.patch.object(verifier.verify_phonon2_pack, "EXPECTED_MANIFEST_SHA256", hashlib.sha256(encoded).hexdigest()):
                with self.assertRaisesRegex(AssertionError, "Hash mismatch"):
                    verifier.verify_phonon2_pack.verify(root)


if __name__ == "__main__":
    unittest.main()
