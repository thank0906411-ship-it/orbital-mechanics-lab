"""reaction_wheel_desaturation.py 검증: 휠 반작용 토크의 부호가 몸체가 원하는
토크의 음수인지(실제로 겪은 발산 버그의 회귀 테스트), 휠이 없을 때 21번의
오일러 방정식과 일치하는지, 포화 클램프가 같은 부호 축만 자르는지, 디새추레이션
비용 공식이 모멘트암/질량에 반비례하는지, 외부 토크 없을 때 총 각운동량
크기가 보존되는지, 지속 외란이 휠 모멘텀을 선형으로 쌓는지, 포화가 포인팅을
무너뜨리는지, 디새추레이션이 작지만 0이 아닌 비용으로 포인팅을 회복시키는지
확인."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("attitude/reaction_wheel_desaturation.py")

INERTIA = (1.0, 2.0, 3.0)


def test_wheel_reaction_torque_is_negative_of_desired_body_torque():
  """모터 명령은 몸체가 원하는 토크의 음수여야 한다 - 부호를 반대로 두면
  양의 피드백이 되어 발산한다(실제로 겪은 버그, 회귀 테스트)."""
  kp, kd = m.pid.compute_pd_gains(INERTIA, 5.0, 0.7)
  q_err = m.pid.axis_angle_to_quaternion([1, 0, 0], 10.0)
  omega = np.array([0.01, 0.0, 0.0])
  tau_body_desired = m.pid.pd_control_torque(q_err, omega, np.zeros(3), kp, kd)
  returned = m.wheel_reaction_control_torque(q_err, omega, kp, kd, np.zeros(3), 1e6)
  assert np.allclose(returned, tau_body_desired)
  state = np.concatenate([m.IDENTITY_QUATERNION, omega, np.zeros(3)])
  deriv = m.wheel_augmented_state_derivative(state, np.array(INERTIA), np.zeros(3), tau_body_desired)
  dh_wheel = deriv[7:10]
  assert np.allclose(dh_wheel, -tau_body_desired), "휠 모멘텀 변화율은 몸체 토크의 음수여야 함"


def test_wheel_augmented_derivative_matches_pid_module_when_wheel_absent():
  """h_wheel=0이고 torque_cmd를 몸체 토크로 직접 넘기면, 이 파일의 미분함수가
  21번의 torqued_euler_equations_derivative와 정확히 일치해야 한다."""
  inertia = np.array(INERTIA)
  omega = np.array([0.1, -0.05, 0.02])
  torque = np.array([0.01, -0.02, 0.005])
  state = np.concatenate([m.IDENTITY_QUATERNION, omega, np.zeros(3)])
  deriv = m.wheel_augmented_state_derivative(state, inertia, torque, np.zeros(3))
  expected_domega = m.pid.torqued_euler_equations_derivative(omega, INERTIA, torque)
  assert np.allclose(deriv[4:7], expected_domega)


def test_clamp_wheel_torque_zeroes_saturated_axis_regardless_of_direction():
  """포화된 축은 방향에 상관없이 토크가 0이 되고, 포화 안 된 축은 그대로
  통과해야 한다. 방향을 인식해 "줄이는 방향"만 통과시키는 접근은 자세 오차가
  180도 근처일 때 PD 법칙이 명령하는 큰 토크가 한 스텝 안에 반대 극성으로
  오버슈트시키는 문제를 실측으로 발견해 폐기했다(clamp_wheel_torque
  docstring 참고)."""
  h_wheel = np.array([0.05, -0.05, 0.0])
  tau_body = np.array([-1.0, 1.0, 1.0])
  result = m.clamp_wheel_torque(tau_body, h_wheel, 0.05)
  assert result[0] == 0.0, "포화된 축은 방향 무관하게 토크가 0이어야 함"
  assert result[1] == 0.0, "포화된 축은 방향 무관하게 토크가 0이어야 함"
  assert result[2] == 1.0, "포화 안 된 축의 토크는 항상 통과해야 함"


def test_desaturation_delta_v_cost_scales_inversely_with_moment_arm_and_mass():
  """모멘트암이나 질량을 2배로 하면 같은 모멘텀 덤핑 비용이 절반이 되어야
  한다 - 공식이 단순 반비례임을 확인."""
  base = m.desaturation_delta_v_cost(0.1, 0.3, 100.0)
  double_arm = m.desaturation_delta_v_cost(0.1, 0.6, 100.0)
  double_mass = m.desaturation_delta_v_cost(0.1, 0.3, 200.0)
  assert double_arm == pytest.approx(base / 2)
  assert double_mass == pytest.approx(base / 2)


def test_total_angular_momentum_magnitude_conserved_without_external_torque():
  """데모1: 외부 토크가 없으면 L_total 벡터 자체는 몸체좌표계에서 회전해도
  크기는 전체 적분 동안 1e-6 이내로 보존돼야 한다."""
  result = m.demo_total_angular_momentum_is_conserved_without_external_torque()
  assert result["max_drift"] < 1e-6


def test_sustained_disturbance_accumulates_wheel_momentum_at_disturbance_rate():
  """데모2: 포화 없이는 휠 모멘텀 증가율이 외란 크기와 1% 이내로 일치하고,
  포인팅 오차는 1도 미만으로 유지돼야 한다."""
  result = m.demo_sustained_disturbance_saturates_wheel_while_pointing_stays_tight()
  assert result["growth_rate"] == pytest.approx(m.DEFAULT_DISTURBANCE_TORQUE_NM, rel=0.01)
  assert result["max_error_deg"] < 1.0


def test_saturation_causes_large_pointing_error_without_desaturation():
  """데모3: 포화 시점이 capacity/disturbance 예측과 5초 이내로 일치하고,
  포화 후 오차가 30도를 넘어야 한다(하드 실패, 점진적 열화가 아님)."""
  result = m.demo_saturation_causes_pointing_loss_without_desaturation()
  assert abs(result["sat_time"] - result["predicted_sat_time"]) < 5.0
  assert result["max_error_after_sat"] > 30.0


def test_desaturation_burn_recovers_pointing_at_nonzero_but_small_delta_v_cost():
  """데모4(캡스톤): 디새추레이션 후 오차가 2도 미만으로 회복되고, 그 비용이
  0보다 크고 22번 기본 호만 전이 델타-V보다는 작아야 한다."""
  result = m.demo_desaturation_burn_recovers_control_at_a_real_delta_v_cost()
  assert result["final_err_deg"] < 2.0
  assert 0 < result["desat_dv_km_s"] < result["hohmann_dv_km_s"]
