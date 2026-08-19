#!/usr/bin/env bash
# Every suite, in the order that fails fastest.
set -e
cd "$(dirname "$0")/../.."
python scripts/build_extension.py > /dev/null
echo "--- unit"
python -m pytest scripts/tests -q
echo "--- inference queue"; node scripts/tests/test_serialize.mjs | tail -1
echo "--- content script"; node scripts/tests/test_content_smoke.mjs | tail -1
echo "--- scenarios";      node scripts/tests/test_scenarios.mjs | tail -1
echo "--- tokenizer parity"; python scripts/tests/test_tokenizer_parity.py --n 2000 | tail -1
echo "--- proposer parity";  python scripts/tests/test_proposer_parity.py --n 1888 | tail -1
echo "--- inference parity"; python scripts/tests/test_inference_parity.py --n 40 | tail -1
echo
echo "all suites passed"
