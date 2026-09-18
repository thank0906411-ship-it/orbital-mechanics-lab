"""torque_free_rigid_body.py 검증: 토크 없는 자유회전에서 회전 에너지/각운동량이
보존되는지, 최대축/최소축 정렬 회전은 작은 섭동에 안정적인지, 중간축 정렬 회전은
같은 섭동에 크게 발산하는지(중간축 정리), 세 축의 섭동 성장 배율을 비교하면
중간축이 압도적으로 큰지."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("attitude/torque_free_rigid_body.py")


def test_spherically_symmetric_body_has_constant_angular_velocity():
  """모든 관성모멘트가 같으면(구대칭) 오일러 방정식의 우변이 항상 0이어야
  한다 — 알려진 특수해로 구현 자체를 검증한다."""
  omega = np.array([0.5, 0.8, 0.3])
  derivative = m.euler_equations_derivative(omega, inertia_diag=(2.0, 2.0, 2.0))
  assert derivative == pytest.approx(np.zeros(3), abs=1e-12)


def test_rotational_energy_formula():
  omega = np.array([1.0, 2.0, 3.0])
  inertia_diag = (1.0, 2.0, 3.0)
  expected = (1.0 * 1.0 ** 2 + 2.0 * 2.0 ** 2 + 3.0 * 3.0 ** 2) / 2
  assert m.rotational_energy(omega, inertia_diag) == pytest.approx(expected)


def test_angular_momentum_magnitude_formula():
  omega = np.array([1.0, 0.0, 0.0])
  inertia_diag = (2.0, 3.0, 4.0)
  assert m.angular_momentum_magnitude(omega, inertia_diag) == pytest.approx(2.0)


def test_energy_and_momentum_conserved_during_free_rotation():
  """이 스크립트의 핵심 주장: 토크 없는 자유회전에서 회전 에너지와 각운동량
  크기는 수치오차 수준 안에서 보존되어야 한다."""
  result = m.demo_conservation_of_energy_and_momentum()
  assert result["energy_drift"] < 1e-6
  assert result["momentum_drift"] < 1e-6


def test_major_and_minor_axis_spins_stay_bounded():
  """최대축/최소축 정렬 회전은 작은 섭동에도 크게 발산하지 않아야 한다(안정)."""
  result = m.demo_major_and_minor_axis_spins_are_stable()
  assert result["ratio_minor"] < 10.0
  assert result["ratio_major"] < 10.0


def test_intermediate_axis_spin_diverges():
  """이 스크립트의 핵심 주장(중간축 정리): 중간축 정렬 회전은 같은 크기 섭동에
  크게 발산해야 한다(불안정)."""
  result = m.demo_intermediate_axis_spin_is_unstable()
  assert result["perturbation_growth_ratio"] > 20.0


def test_intermediate_axis_growth_dominates_comparison():
  """세 축을 나란히 비교하면 중간축의 섭동 성장 배율이 다른 두 축보다 압도적으로
  커야 한다."""
  rows = m.demo_perturbation_growth_comparison()
  by_label = {row["axis_label"]: row["growth_ratio"] for row in rows}
  intermediate_ratio = by_label["중간축(I2)"]
  assert intermediate_ratio > by_label["최소축(I1)"] * 5
  assert intermediate_ratio > by_label["최대축(I3)"] * 5


def test_rk4_step_matches_manual_derivative_for_small_step():
  """아주 작은 스텝에서는 RK4 한 스텝의 결과가 오일러 방정식 우변으로 예측한
  1차 근사와 거의 일치해야 한다."""
  omega = np.array([0.5, 0.8, 0.3])
  inertia_diag = (1.0, 2.0, 3.0)
  dt = 1e-6
  new_omega = m.rk4_step(omega, dt, inertia_diag)
  expected_approx = omega + dt * m.euler_equations_derivative(omega, inertia_diag)
  assert new_omega == pytest.approx(expected_approx, abs=1e-9)
