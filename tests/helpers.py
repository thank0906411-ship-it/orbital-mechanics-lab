"""
pytest 공통 설정.

이 프로젝트의 스크립트는 숫자로 시작하는 파일명(예: 01_Kepler_orbit_propagation.py)이라
일반 `import` 문법으로 불러올 수 없고, 각각 `if __name__ == "__main__":` 가드로 실행부를
감싸고 있어 import해도 시뮬레이션이 저절로 돌지 않는다 — 뒤 번호 스크립트가 앞 번호를
재사용할 때도 이 파일과 동일한 importlib 로드 패턴을 쓴다.
"""

import importlib.util
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(filename):
  """PROJECT_ROOT의 파이썬 파일을 이름과 무관하게 모듈로 로드한다"""
  path = os.path.join(PROJECT_ROOT, filename)
  spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module
