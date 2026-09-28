import sys
from absl import flags
from absl.testing import absltest
flags.FLAGS(['validation'])
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
