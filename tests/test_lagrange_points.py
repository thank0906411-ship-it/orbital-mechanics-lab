"""lagrange_points.py 검증: 콜린스 라그랑주점이 알려진 극한(mu=0.5 대칭)에서
타당한지, 삼각 라그랑주점이 실제로 두 주천체와 정삼각형을 이루는지, 콜린스
라그랑주점이 모두 불안정한지, 삼각 라그랑주점이 지구-달 질량비에서 안정한지,
CR3BP 가속도가 라그랑주점 자체에서는 거의 0인지, 라그랑주점 근처 섭동 궤적이
L4에서는 유계, L1에서는 훨씬 크게 벗어나는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/lagrange_points.py")


def test_collinear_points_symmetric_for_equal_mass_ratio():
  """mu=0.5(두 주천체 질량이 같음)이면 L1(두 천체 사이)은 정확히 x=0(두
  천체의 중점)에 있어야 하고, L2/L3는 중점 기준으로 대칭이어야 한다."""
  x_l1, x_l2, x_l3 = m.find_collinear_lagrange_points(0.5)
  assert x_l1 == pytest.approx(0.0, abs=1e-6)
  assert x_l2 == pytest.approx(-x_l3, abs=1e-6)


def test_collinear_points_ordered_l3_lt_m1_lt_l1_lt_m2_lt_l2():
  """L1은 두 주천체 사이, L2는 M2(부천체) 바깥쪽, L3는 M1(주천체) 바깥쪽에
  있어야 한다 - x좌표 순서가 L3 < -mu(M1) < L1 < 1-mu(M2) < L2."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, x_l2, x_l3 = m.find_collinear_lagrange_points(mu)
  assert x_l3 < -mu < x_l1 < (1 - mu) < x_l2


def test_triangular_points_equidistant_from_both_primaries():
  """L4/L5는 두 주천체로부터 항상 거리 1(무차원, 정삼각형의 변)이어야 한다."""
  mu = m.EARTH_MOON_MASS_RATIO
  (x4, y4), (x5, y5) = m.triangular_lagrange_points(mu)
  m1 = np.array([-mu, 0.0])
  m2 = np.array([1 - mu, 0.0])
  for point in [np.array([x4, y4]), np.array([x5, y5])]:
    assert np.linalg.norm(point - m1) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(point - m2) == pytest.approx(1.0, abs=1e-9)


def test_triangular_points_symmetric_about_x_axis():
  """L4와 L5는 x축에 대해 대칭이어야 한다(y좌표 부호만 반대)."""
  mu = m.EARTH_MOON_MASS_RATIO
  (x4, y4), (x5, y5) = m.triangular_lagrange_points(mu)
  assert x4 == pytest.approx(x5, abs=1e-6)
  assert y4 == pytest.approx(-y5, abs=1e-6)


def test_cr3bp_acceleration_near_zero_at_lagrange_point():
  """라그랑주점은 정의상 평형점이므로, 그 점에서 정지 상태(속도 0)로 시작하면
  가속도가 거의 0이어야 한다."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  state = np.array([x_l1, 0.0, 0.0, 0.0, 0.0, 0.0])
  derivative = m.cr3bp_acceleration(state, mu)
  assert derivative[3] == pytest.approx(0.0, abs=1e-9)
  assert derivative[4] == pytest.approx(0.0, abs=1e-9)


def test_collinear_points_all_unstable():
  """L1/L2/L3는 모두 최대 고유값 실수부가 뚜렷한 양수여야 한다(불안정)."""
  mu = m.EARTH_MOON_MASS_RATIO
  x_l1, x_l2, x_l3 = m.find_collinear_lagrange_points(mu)
  for x in [x_l1, x_l2, x_l3]:
    eigenvalues = m.stability_eigenvalues(x, 0.0, mu)
    assert np.max(eigenvalues.real) > 0.01


def test_triangular_points_stable_for_earth_moon_ratio():
  """지구-달 질량비에서 L4는 모든 고유값의 실수부가 0에 가까워야 한다(안정,
  질량비가 루스 임계값 약 0.0385보다 작으므로)."""
  mu = m.EARTH_MOON_MASS_RATIO
  (x4, y4), _ = m.triangular_lagrange_points(mu)
  eigenvalues = m.stability_eigenvalues(x4, y4, mu)
  assert abs(np.max(eigenvalues.real)) < 1e-6


def test_triangular_points_unstable_above_routh_critical_ratio():
  """질량비가 루스 임계값(약 0.0385)보다 크면 L4/L5도 불안정해져야 한다 -
  안정성이 지구-달처럼 작은 질량비에서만 성립하는 특수한 경우임을 확인한다."""
  mu = 0.1
  (x4, y4), _ = m.triangular_lagrange_points(mu)
  eigenvalues = m.stability_eigenvalues(x4, y4, mu)
  assert np.max(eigenvalues.real) > 0.01


def test_perturbed_trajectory_bounded_near_l4_diverges_near_l1():
  """같은 크기의 초기 섭동을 줬을 때, L4 근처 궤적은 유계로 남지만 L1 근처
  궤적은 훨씬 크게 벗어나야 한다(선형 안정성 분석과 일치하는 비선형 결과)."""
  mu = m.EARTH_MOON_MASS_RATIO
  (x_l4, y_l4), _ = m.triangular_lagrange_points(mu)
  x_l1, _, _ = m.find_collinear_lagrange_points(mu)
  perturbation = (0.001, 0.001, 0.0, 0.0)

  l4_result = m.propagate_near_lagrange_point((x_l4, y_l4), perturbation, 5000, 0.001, mu)
  l1_result = m.propagate_near_lagrange_point((x_l1, 0.0), perturbation, 5000, 0.001, mu)

  assert l1_result["max_distance_from_point"] > l4_result["max_distance_from_point"] * 3
