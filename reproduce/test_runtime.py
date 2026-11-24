# SPDX-License-Identifier: AGPL-3.0-only
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'expression'))
import numpy as np
from runtime import ExpressionModel, ExpressionPolicy, accepted, probabilities, quality_reason


class RuntimeTests(unittest.TestCase):
    def test_policy_and_quality_rejection(self):
        with self.assertRaises(ValueError):
            ExpressionPolicy.from_json('{"temperature":0,"min_probability":0.63,"min_margin":0.05}')
        self.assertEqual(quality_reason(np.zeros((48, 48), dtype=np.uint8)), 'low_contrast')
        self.assertFalse(accepted(probabilities(np.zeros((1, 7)), 1.231144413344916), ExpressionPolicy(1.231144413344916, .63, .05))[0])

    @unittest.skipUnless(os.environ.get('CORE_CV_LOCAL_MODEL_DIR'), 'requires existing authorized local models')
    def test_actual_e1_cpu_logits_match_recorded_oracle(self):
        oracle = json.loads((Path(__file__).parents[1] / 'tests/fixtures/oracle.json').read_text())
        runtime = ExpressionModel.from_file(Path(os.environ['CORE_CV_LOCAL_MODEL_DIR']) / oracle['e1']['file'])
        self.assertEqual(runtime.version, oracle['e1']['checkpoint_sha256'])
        y, x = np.indices((48, 48), dtype=np.uint32)
        arrays = {'checker3': (x // 3 + y // 3) % 2 * 255, 'mixed': (13*x + 31*y + (x*y)%251)%256,
                  'ramp': (5*x + 3*y)%256, 'vertical': 13*y%256}
        for example in oracle['e1']['examples']:
            value = arrays[example['name']].astype(np.float32)[None, None] / np.float32(255)
            np.testing.assert_allclose(runtime.logits(value)[0], example['logits'], rtol=1e-4, atol=1e-5)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                runtime.logits(np.full((1, 1, 48, 48), value, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
