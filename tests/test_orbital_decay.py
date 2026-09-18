"""orbital_decay.py 검증: 대기밀도 함수가 고도가 높을수록 작아지는지(단조
감소), 항력 감속이 항상 속도 반대방향인지, 하강이 재진입 고도에 도달하는지,
후반부 하강이 전반부보다 빠른지, 탄도계수가 클수록 수명이 길어지는지, 초기
고도가 높을수록 수명이 기하급수적으로 늘어나는지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("missions/orbital_decay.py")


def test_atmospheric_density_decreases_with_altitude():
  """고도가 높을수록 대기밀도는 작아야 한다(단조 감소)."""
  low = m.atmospheric_density(100.0)
  high = m.atmospheric_density(400.0)
  assert low > high


def test_atmospheric_density_matches_reference_at_reference_altitude():
  """기준 고도에서는 밀도가 정확히 기준 밀도와 같아야 한다."""
  density = m.atmospheric_density(m.REFERENCE_ALTITUDE_KM)
  assert density == pytest.approx(m.REFERENCE_DENSITY_KG_M3)


def test_drag_deceleration_opposes_velocity_direction():
  """항력 가속도의 방향은 속도 방향과 정확히 반대여야 한다."""
  velocity = np.array([5.0, 5.0, 0.0])
  accel = m.drag_deceleration(velocity, 200.0, 50.0)
  velocity_unit = velocity / np.linalg.norm(velocity)
  accel_unit = accel / np.linalg.norm(accel)
  assert accel_unit == pytest.approx(-velocity_unit)


def test_drag_deceleration_zero_for_zero_velocity():
  """속도가 0이면 항력 가속도도 0이어야 한다(방향 미정의를 안전하게 처리)."""
  accel = m.drag_deceleration(np.zeros(3), 200.0, 50.0)
  assert accel == pytest.approx(np.zeros(3))


def test_drag_stronger_at_lower_altitude():
  """같은 속도라면 고도가 낮을수록(밀도가 높을수록) 항력 가속도 크기가
  커야 한다."""
  velocity = np.array([7.5, 0.0, 0.0])
  accel_low = m.drag_deceleration(velocity, 100.0, 50.0)
  accel_high = m.drag_deceleration(velocity, 400.0, 50.0)
  assert np.linalg.norm(accel_low) > np.linalg.norm(accel_high)


def test_decay_reaches_reentry_altitude():
  """이 스크립트의 핵심 주장: 대기항력만으로 궤도가 재진입 고도까지 하강해야
  한다."""
  result = m.demo_decay_reaches_reentry_altitude()
  assert result["lifetime_sec"] > 0


def test_descent_accelerates_toward_reentry():
  """이 스크립트의 핵심 주장: 지수함수 밀도 모델 때문에 궤적 후반부의 고도
  감소율이 전반부보다 훨씬 커야 한다(가속 하강)."""
  result = m.demo_descent_accelerates_near_reentry()
  assert result["second_half_rate_km_day"] > result["first_half_rate_km_day"] * 2


def test_larger_ballistic_coefficient_extends_lifetime():
  """탄도계수가 클수록 궤도 수명이 길어져야 한다."""
  rows = m.demo_larger_ballistic_coefficient_extends_lifetime()
  assert rows[-1]["lifetime_days"] > rows[0]["lifetime_days"]


def test_lifetime_grows_faster_than_linear_with_altitude():
  """이 스크립트의 핵심 주장: 초기 고도가 선형으로 늘어나도 궤도 수명은
  기하급수적으로(선형보다 훨씬 빠르게) 늘어나야 한다."""
  rows = m.demo_lifetime_grows_exponentially_with_initial_altitude()
  assert rows[-1]["lifetime_days"] > rows[0]["lifetime_days"] * 5


def test_decay_transfer_raises_when_drag_too_weak_for_max_steps():
  """항력이 극단적으로 약하면(탄도계수가 극단적으로 크면) max_steps 안에
  재진입 고도에 도달하지 못해 명시적 오류를 내야 한다."""
  with pytest.raises(RuntimeError):
    m.decay_transfer(200.0, 100.0, 1e9, 30.0, max_steps=50)
