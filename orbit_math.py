"""
02, 04, 05, 07번이 공유하는 3축 회전행렬과 지구 반지름 상수.

04번을 개발하며 SEZ 좌표변환에서 흔히 인용되는 회전 부호(Ry(90-lat)*Rz(+theta))가
이 프로젝트의 회전행렬 정의(능동회전 관례)와 맞지 않아 천정 고도각이 90도로 나오지
않는 버그를 실제로 겪었다 — 회전행렬 자체를 여러 파일에 복사해두면 이런 부호
문제를 한 곳에서만 고치고 다른 복사본은 놓치기 쉽다. 그래서 회전행렬 구현을
이 모듈 하나로 모으고, 02/04/05/07은 전부 여기서 import해서 쓴다.
"""

import numpy as np

EARTH_RADIUS_KM = 6378.137  # WGS84 적도 반지름 근사


def rotation_matrix_x(angle_rad):
  c, s = np.cos(angle_rad), np.sin(angle_rad)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle_rad):
  c, s = np.cos(angle_rad), np.sin(angle_rad)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle_rad):
  c, s = np.cos(angle_rad), np.sin(angle_rad)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
