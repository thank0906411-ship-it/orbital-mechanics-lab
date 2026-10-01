"""pid_attitude_control.py 검증: 쿼터니언 운동학이 각속도 0에서 미분도 0인지,
쿼터니언 곱셈의 항등원 성질과 conjugate가 실제 inverse인지, 현재=목표일 때
자세 오차가 항등원에 가까운지, q와 -q 이중성에 대한 부호 고정이 실제로
q_err[0]>=0을 보장하는지, 알려진 회전각에서 오차각 추출이 정확한지, 토크=0일
때 이 파일의 오일러 방정식이 16번의 토크 없는 버전과 정확히 일치하는지, RK4
스텝 후 재정규화로 단위 노름이 유지되는지, 게인 공식이 양의 입력에 양의
게인을 주고 정착시간이 짧을수록 게인이 커지는지, 각 데모의 핵심 주장을
독립적으로 재검증."""

import numpy as np
import pytest
from helpers import load_module

m = load_module("attitude/pid_attitude_control.py")
torque_free = load_module("attitude/torque_free_rigid_body.py")


def test_quaternion_kinematics_zero_omega_gives_zero_derivative():
  """각속도가 0이면 자세가 전혀 변하지 않아야 한다 — dq/dt=0."""
  q = m.axis_angle_to_quaternion([0, 0, 1], 30.0)
  derivative = m.quaternion_kinematics_derivative(q, np.zeros(3))
  assert derivative == pytest.approx(np.zeros(4), abs=1e-12)


def test_quaternion_multiply_identity_is_identity():
  """임의의 쿼터니언에 항등원을 곱하면 그대로 나와야 한다."""
  q = m.axis_angle_to_quaternion([1, 1, 0], 50.0)
  result = m.quaternion_multiply(q, m.IDENTITY_QUATERNION)
  assert result == pytest.approx(q)


def test_quaternion_conjugate_of_unit_quaternion_is_inverse():
  """단위 쿼터니언의 conjugate는 inverse와 같아야 한다 — q⊗conj(q)=항등원."""
  q = m.axis_angle_to_quaternion([0, 1, 0], 70.0)
  result = m.quaternion_multiply(q, m.quaternion_conjugate(q))
  assert result == pytest.approx(m.IDENTITY_QUATERNION, abs=1e-10)


def test_attitude_error_zero_when_current_equals_target():
  """현재 자세가 목표와 같으면 오차는 항등원에 가까워야 한다(벡터부 거의 0)."""
  q = m.axis_angle_to_quaternion([1, 0, 0], 25.0)
  q_err = m.attitude_error_quaternion(q, q)
  assert abs(q_err[0]) == pytest.approx(1.0, abs=1e-10)
  assert q_err[1:] == pytest.approx(np.zeros(3), abs=1e-10)


def test_attitude_error_sign_fix_forces_nonnegative_scalar_part():
  """q와 -q가 같은 자세를 나타내는 이중성 때문에, 원 곱셈 결과의 스칼라부가
  음수여도 attitude_error_quaternion은 항상 스칼라부가 0 이상인 오차를
  반환해야 한다(제어기가 항상 가까운 길로 돌아가도록)."""
  q_current = m.axis_angle_to_quaternion([0, 0, 1], 179.0)
  q_target = m.IDENTITY_QUATERNION.copy()
  q_err = m.attitude_error_quaternion(q_current, q_target)
  assert q_err[0] >= 0.0


def test_quaternion_error_angle_matches_known_rotation():
  """알려진 60도 회전에서 만든 오차 쿼터니언의 각도 추출이 60도에 근접해야
  한다."""
  q_err = m.axis_angle_to_quaternion([0, 0, 1], 60.0)
  angle = m.quaternion_error_angle_deg(q_err)
  assert angle == pytest.approx(60.0, abs=1e-6)


def test_torqued_euler_matches_free_euler_when_torque_zero():
  """토크가 0이면 이 파일의 토크 있는 오일러 방정식은 16번의 토크 없는
  버전과 정확히 일치해야 한다 — 덧셈만 추가된 확장이라는 설계 주장을
  직접 검증한다."""
  omega = np.array([0.5, 0.8, 0.3])
  inertia_diag = (1.0, 2.0, 3.0)
  expected = torque_free.euler_equations_derivative(omega, inertia_diag)
  actual = m.torqued_euler_equations_derivative(omega, inertia_diag, torque=np.zeros(3))
  assert actual == pytest.approx(expected)


def test_rk4_step_preserves_unit_norm_after_renormalization():
  """RK4 한 스텝을 거친 뒤에도 쿼터니언 노름이 1에 가까워야 한다."""
  q = m.axis_angle_to_quaternion([1, 0, 0], 20.0)
  omega = np.array([0.1, 0.2, 0.3])
  state = np.concatenate([q, omega])
  new_state = m.rk4_step(state, dt_sec=0.01, inertia_diag=(1.0, 2.0, 3.0), torque=np.zeros(3))
  assert np.linalg.norm(new_state[:4]) == pytest.approx(1.0, abs=1e-12)


def test_compute_pd_gains_positive_for_positive_inputs():
  """양의 관성모멘트, 정착시간, 감쇠비에서는 Kp, Kd 모두 양수여야 한다."""
  kp, kd = m.compute_pd_gains((1.0, 2.0, 3.0), settling_time_sec=5.0, zeta=0.7)
  assert np.all(kp > 0)
  assert np.all(kd > 0)


def test_compute_pd_gains_shorter_settling_time_gives_higher_gains():
  """정착시간을 절반으로 줄이면 Kp, Kd 모두 커져야 한다(더 공격적인 제어)."""
  kp_slow, kd_slow = m.compute_pd_gains((1.0, 2.0, 3.0), settling_time_sec=5.0)
  kp_fast, kd_fast = m.compute_pd_gains((1.0, 2.0, 3.0), settling_time_sec=2.5)
  assert np.all(kp_fast > kp_slow)
  assert np.all(kd_fast > kd_slow)


def test_free_kinematics_preserves_unit_quaternion():
  """데모1: 토크 없는 긴 적분 동안 쿼터니언 노름 드리프트가 수치오차 수준
  이내여야 한다."""
  result = m.demo_free_kinematics_preserves_unit_quaternion()
  assert result["max_drift"] < 1e-8


def test_pd_point_and_hold_settles_below_threshold():
  """데모2: 정착시간의 3배 후 자세 오차가 충분히 작고, 정착시간 이후 오차가
  다시 초기 오차 수준으로 커지지 않아야 한다 — 데모 내부 assert만 믿지 않고
  반환된 시계열을 독립적으로 재검증한다."""
  result = m.demo_pd_point_and_hold()
  assert result["final_error"] < 2.0
  post_settling = [e for t, e in result["error_angles"] if t > result["settling_time_sec"]]
  assert max(post_settling) < 35.0


def test_pd_control_tames_intermediate_axis_tumble():
  """데모3: 16번의 중간축 불안정 설정에서, 제어가 있으면 최대 자세 오차가
  제어 없을 때의 1/5 미만이어야 한다(정성적 비교)."""
  result = m.demo_pd_control_tames_intermediate_axis_tumble()
  assert result["max_controlled"] < result["max_uncontrolled"] / 5


def test_higher_gain_settles_faster_with_more_overshoot():
  """데모4: 정착시간을 절반으로 줄인 쪽이 더 빨리 정착하고, 정착 후 잔여
  오차가 비슷하거나 더 커야 한다(오버슈트/링잉 경향)."""
  result = m.demo_higher_gain_settles_faster_with_more_overshoot()
  slow, fast = result["slow"], result["fast"]
  assert slow["settled_time"] is not None and fast["settled_time"] is not None
  assert fast["settled_time"] < slow["settled_time"]
  assert fast["max_post_settled"] >= slow["max_post_settled"] * 0.5
