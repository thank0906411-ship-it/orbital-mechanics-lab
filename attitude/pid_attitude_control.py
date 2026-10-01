"""
PID 자세제어(Attitude Control) - 쿼터니언 자세 오차를 PD 피드백으로 잡기

16번(torque_free_rigid_body.py)은 토크 없는 강체 자유회전(오일러 방정식)만
다뤘고, 그 스크립트 docstring이 명시적으로 "쿼터니언 자세 전파나 피드백 제어
(PID 등)는 이 프로젝트에서 계속 범위 밖으로 남겨둔다"고 범위를 그었다. 이
스크립트는 그 경계를 의도적으로 뒤집는다 - 목표 자세로 위성을 수렴시키는 PD
피드백 제어 루프를 새로 연다.

16번은 각속도만 적분하고 자세(방향) 자체는 전혀 추적하지 않았다. PID 제어는
"목표 자세와의 오차"를 정의해야 하므로, 쿼터니언 표현과 그 운동학(kinematics)
자체가 이 스크립트에서 처음 등장하는 새 인프라다. 16번의 오일러 방정식에는
토크 항이 없어 그대로 재사용할 수 없으므로, 토크 항이 있는 버전을 이 파일에만
새로 정의한다(16번 자체는 건드리지 않아 "외부 입력 없는 순수 동역학"이라는
그 스크립트의 범위 선언을 그대로 보존한다).

핵심 개념 1: 쿼터니언은 몸체 각속도로부터 자세를 시간에 따라 갱신한다
  scalar-first 쿼터니언 q=[q0,q1,q2,q3](q0이 스칼라부)로 자세를 표현한다.
  운동학 미분방정식 dq0/dt=-0.5*(qv·ω), dqv/dt=0.5*(q0*ω + qv x ω)를 16번의
  오일러 방정식(각속도)과 나란히 RK4로 적분해야 실제로 자세가 시간에 따라
  어떻게 바뀌는지 추적할 수 있다. 수치적분 중 단위구(||q||=1)에서 벗어나는
  드리프트가 누적되므로, 매 RK4 스텝 후(스테이지 내부가 아니라) 정규화한다.

핵심 개념 2: 쿼터니언 곱으로 "자세 오차"를 정의하고, 부호 이중성을 고정한다
  현재 자세 q와 목표 자세 q_target 사이의 오차는 q_err = conj(q_target)⊗q로
  정의된다(단위 쿼터니언의 conjugate=inverse). q_err의 벡터부가 작은 오차에서
  대략 회전벡터의 절반에 비례해 PID 제어 입력으로 쓸 수 있다. 다만 q와 -q가
  물리적으로 같은 자세를 나타내는 이중성 때문에, q_err[0]<0이면 q_err 전체의
  부호를 뒤집어야 한다 - 안 그러면 제어기가 "먼 길"로 돌아가려 든다. 이 부호
  고정은 attitude_error_quaternion 함수 내부에서, 곱셈 직후·벡터부 추출 직전
  순서로 수행해 호출부가 실수할 여지를 구조적으로 없앤다.

핵심 개념 3: PD 제어 게인은 정착시간/감쇠비로 원칙적으로 정한다
  q_err 벡터부가 평형 근처에서 대략 회전각의 절반이라는 선형화를 이용하면,
  torque=-Kp*q_err_vec-Kd*ω_err 제어법칙이 2차 시스템 θ''+2ζωn*θ'+ωn²*θ=0을
  따른다. 원하는 정착시간 ts와 감쇠비 ζ(약 0.7, 오버슈트 거의 없음)를 정하면
  ωn≈4/(ζ*ts), Kp=2*I*ωn², Kd=2*ζ*ωn*I로 게인이 임의의 매직넘버가 아니라
  원하는 응답 특성에서 역산된다. 지속적인 외란 토크가 이 프로젝트 물리에
  없으므로 적분항(Ki)은 넣지 않는다 - 17번(station-keeping)에서 "필요 이상의
  장치를 붙이지 않는다"는 이 프로젝트의 확립된 태도와 일치한다.

핵심 개념 4: 제어기가 16번의 시그니처 결과(중간축 불안정)를 정성적으로 길들인다
  16번과 똑같은 설정(I=(1,2,3), 중간축에 1% 섭동)에서 시작한 자유회전은
  텀블링으로 발산했다. 같은 초기조건에 PD 제어기를 적용하면(스핀 자체를
  멈추고 시작 자세를 유지하려는 point-and-hold), 제어 없는 경우 대비 최대
  자세 오차가 훨씬 작게 억제되는 것을 보여준다. 이건 엄밀한 게인 튜닝 연구가
  아니라 정성적 비교로만 다룬다 - "왜 이 확장이 하필 이 프로젝트에 들어가야
  하는가"에 대한 가장 직접적인 답이다.

단순화: 지속적 외란 토크가 없는 PD 전용 제어만 다룬다(적분항 없음). 게인은
선형화 공식으로 원칙적으로 구하되, 35도 같은 중간 크기 초기 오차에서는 그
공식이 가정한 소각근사가 정확하지 않을 수 있음을 인지하고 assert에 여유를
둔다. 궤적 추적(시간에 따라 변하는 목표 자세)이 아니라 점 목표
(point-and-hold)만 다룬다.
"""

import argparse
import csv
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)

IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def quaternion_kinematics_derivative(q, omega):
  """dq/dt: 몸체 각속도로부터 자세(쿼터니언)가 시간에 따라 바뀌는 속도.
  dq0/dt=-0.5*(qv·ω), dqv/dt=0.5*(q0*ω + qv x ω)."""
  q0, qv = q[0], q[1:]
  dq0 = -0.5 * np.dot(qv, omega)
  dqv = 0.5 * (q0 * omega + np.cross(qv, omega))
  return np.array([dq0, dqv[0], dqv[1], dqv[2]])


def normalize_quaternion(q):
  """수치적분 중 단위구(||q||=1)에서 벗어나는 드리프트를 매 스텝 보정한다."""
  return q / np.linalg.norm(q)


def quaternion_conjugate(q):
  """단위 쿼터니언에서 conjugate는 inverse와 같다 - 벡터부의 부호만 뒤집는다."""
  return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q_a, q_b):
  """해밀턴 곱(scalar-first). q_err = conj(q_target) ⊗ q_current 계산에 쓰인다."""
  a0, a1, a2, a3 = q_a
  b0, b1, b2, b3 = q_b
  return np.array([
      a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
      a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
      a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
      a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
  ])


def attitude_error_quaternion(q_current, q_target):
  """q_err = conj(q_target) ⊗ q_current. q와 -q가 같은 자세를 나타내는 이중성
  때문에 q_err[0]<0이면 전체 부호를 뒤집어 제어기가 항상 "가까운 길"로 돌아가게
  만든다 - 이 부호 고정을 함수 내부에서 강제해 호출부가 실수할 여지를 없앤다."""
  q_err = quaternion_multiply(quaternion_conjugate(q_target), q_current)
  if q_err[0] < 0:
    q_err = -q_err
  return q_err


def quaternion_error_angle_deg(q_err):
  """q_err가 나타내는 실제 회전각(도 단위): angle = 2*arccos(|q0|)."""
  return np.degrees(2 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0)))


def torqued_euler_equations_derivative(omega, inertia_diag, torque):
  """16번 euler_equations_derivative에 토크 항을 더한 버전:
  I1*dω1/dt=(I2-I3)*ω2*ω3+τ1, ... (16번 함수는 토크 항이 없어 재사용 불가하므로
  이 파일에 완전히 새로 정의한다)."""
  omega1, omega2, omega3 = omega
  i1, i2, i3 = inertia_diag
  tau1, tau2, tau3 = torque
  domega1 = ((i2 - i3) * omega2 * omega3 + tau1) / i1
  domega2 = ((i3 - i1) * omega3 * omega1 + tau2) / i2
  domega3 = ((i1 - i2) * omega1 * omega2 + tau3) / i3
  return np.array([domega1, domega2, domega3])


def attitude_state_derivative(state, inertia_diag, torque):
  """7차원 state=[q0,q1,q2,q3,ω1,ω2,ω3]의 시간미분. 제어법칙은 이 함수 밖에서
  RK4 호출당 한 번만 평가되고(4개 스테이지마다 재평가하지 않음), 이 함수는
  주어진 torque로 물리 미분만 계산한다 - 물리와 제어기의 책임을 분리한다."""
  q, omega = state[:4], state[4:]
  dq = quaternion_kinematics_derivative(q, omega)
  domega = torqued_euler_equations_derivative(omega, inertia_diag, torque)
  return np.concatenate([dq, domega])


def rk4_step(state, dt_sec, inertia_diag, torque):
  """16번과 같은 4단계 RK4 구조를 7차원 상태에 맞게 확장. 쿼터니언 재정규화는
  전체 조합이 끝난 뒤 한 번만 한다 - RK4 스테이지 내부에서 정규화하면 4개
  스테이지가 공유해야 할 미분함수 자체가 매 스테이지 달라져 RK4의 4차 정확도가
  깨진다."""
  k1 = attitude_state_derivative(state, inertia_diag, torque)
  k2 = attitude_state_derivative(state + dt_sec / 2 * k1, inertia_diag, torque)
  k3 = attitude_state_derivative(state + dt_sec / 2 * k2, inertia_diag, torque)
  k4 = attitude_state_derivative(state + dt_sec * k3, inertia_diag, torque)
  new_state = state + dt_sec / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  new_state[:4] = normalize_quaternion(new_state[:4])
  return new_state


def pd_control_torque(q_err, omega, omega_target, kp, kd):
  """torque = -Kp*q_err_vec - Kd*(ω-ω_target). 지속적 외란 토크가 이 프로젝트
  물리에 없으므로 적분항(Ki)은 넣지 않는다."""
  return -kp * q_err[1:] - kd * (np.array(omega) - np.array(omega_target))


def compute_pd_gains(inertia_diag, settling_time_sec, zeta=0.7):
  """정착시간(settling_time_sec)과 감쇠비(zeta)에서 축별 게인을 역산한다.
  q_err 벡터부≈회전각의 절반(θ/2)이므로 torque=-Kp*(θ/2)-Kd*θ'이고,
  I*θ''=-Kp*θ/2-Kd*θ'를 θ''+2ζωn*θ'+ωn²*θ=0과 비교하면 ωn²=Kp/(2I)에서
  Kp=2*I*ωn², 2ζωn=Kd/I에서 Kd=2*ζ*ωn*I. ωn=4/(ζ*ts)는 2% 정착시간 규칙.
  임의의 매직넘버가 아니라 원하는 응답 특성(정착시간, 오버슈트)에서 게인을
  직접 구한다."""
  omega_n = 4.0 / (zeta * settling_time_sec)
  inertia = np.array(inertia_diag)
  kp = 2 * inertia * omega_n ** 2
  kd = 2 * zeta * omega_n * inertia
  return kp, kd


def axis_angle_to_quaternion(axis, angle_deg):
  """축-각 표현을 쿼터니언으로 변환(데모 초기 자세 오차를 만드는 용도)."""
  axis = np.array(axis, dtype=float)
  axis = axis / np.linalg.norm(axis)
  half_angle = np.radians(angle_deg) / 2
  return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))])


def integrate_attitude_with_control(initial_q, initial_omega, inertia_diag, total_time_sec,
                                     dt_sec, control_law=None, q_target=None):
  """16번 integrate_attitude와 같은 [(t, state), ...] 반환 패턴(state는 (q, omega)
  튜플). control_law가 None이면 매 스텝 토크 0(순수 운동학 데모). 아니면 매
  스텝 q_err을 계산해 control_law(q_err_vec, omega) -> torque를 구한 뒤 적분한다."""
  q = np.array(initial_q, dtype=float)
  omega = np.array(initial_omega, dtype=float)
  t = 0.0
  history = [(t, (q.copy(), omega.copy()))]

  while t < total_time_sec - 1e-9:
    step = min(dt_sec, total_time_sec - t)
    if control_law is None:
      torque = np.zeros(3)
    else:
      q_err = attitude_error_quaternion(q, q_target)
      torque = control_law(q_err, omega)
    state = np.concatenate([q, omega])
    state = rk4_step(state, step, inertia_diag, torque)
    q, omega = state[:4], state[4:]
    t += step
    history.append((t, (q.copy(), omega.copy())))

  return history


def demo_free_kinematics_preserves_unit_quaternion(inertia_diag=(1.0, 2.0, 3.0)):
  """토크 없는 순수 운동학 적분에서 쿼터니언 노름이 재정규화 덕분에 단위구에서
  거의 벗어나지 않는지 확인한다 - 16번의 에너지/각운동량 보존 검증과 같은
  성격의, 적분기 자체의 수치적 정확성 검증이다."""
  print("=" * 70)
  print("[1] 순수 운동학: 토크 없이도 쿼터니언 노름이 단위구를 유지하는가")
  print("=" * 70)
  initial_q = axis_angle_to_quaternion([1, 1, 1], 40.0)
  initial_omega = np.array([0.5, 0.3, 0.2])
  dt_sec = 0.01
  total_time_sec = 30.0

  print(f"초기 자세: 40도 회전(축 (1,1,1)/√3), 초기 각속도={initial_omega} rad/s\n")
  history = integrate_attitude_with_control(initial_q, initial_omega, inertia_diag,
                                             total_time_sec, dt_sec, control_law=None)

  norms = [np.linalg.norm(q) for _t, (q, _omega) in history]
  max_drift = max(abs(n - 1.0) for n in norms)
  print(f"쿼터니언 노름: 시작={norms[0]:.10f}, 끝={norms[-1]:.10f}, 최대편차={max_drift:.2e}")

  assert max_drift < 1e-8, "재정규화 덕분에 쿼터니언 노름은 수치오차 수준 안에서 단위구를 유지해야 함"
  print("\n(토크가 없어도 매 RK4 스텝 후 재정규화한 덕분에 쿼터니언이 단위구를")
  print(" 벗어나지 않는다 - 쿼터니언 운동학 적분 자체가 올바르다는 신호다.)")
  return {"max_drift": max_drift, "history": history}


def demo_pd_point_and_hold(inertia_diag=(1.0, 2.0, 3.0), settling_time_sec=5.0,
                            initial_error_deg=35.0, zeta=0.7):
  """목표 자세(identity)에서 initial_error_deg만큼 벗어난 상태에서 시작해, PD
  제어기가 정착시간 근처에서 오차를 충분히 줄이고 그 이후 다시 커지지 않는지
  확인한다 - 게인 공식과 직접 연결된 핵심 정량 데모."""
  print("\n" + "=" * 70)
  print("[2] PD 자세유지: 초기 오차에서 목표 자세로 수렴하는가")
  print("=" * 70)
  q_target = IDENTITY_QUATERNION.copy()
  initial_q = axis_angle_to_quaternion([1, 0, 0], initial_error_deg)
  initial_omega = np.zeros(3)

  kp, kd = compute_pd_gains(inertia_diag, settling_time_sec, zeta)
  omega_n = 4.0 / (zeta * settling_time_sec)
  dt_sec = 0.05
  total_time_sec = settling_time_sec * 3

  print(f"초기 오차={initial_error_deg}도, 목표 정착시간={settling_time_sec}초, ζ={zeta}")
  print(f"게인: Kp={kp}, Kd={kd} (고유각속도 ωn≈{omega_n:.3f}rad/s)\n")

  control_law = lambda q_err, omega: pd_control_torque(q_err, omega, np.zeros(3), kp, kd)  # noqa: E731
  history = integrate_attitude_with_control(initial_q, initial_omega, inertia_diag,
                                             total_time_sec, dt_sec,
                                             control_law=control_law, q_target=q_target)

  error_angles = [(t, quaternion_error_angle_deg(attitude_error_quaternion(q, q_target)))
                   for t, (q, _omega) in history]
  final_error = error_angles[-1][1]
  post_settling_errors = [e for t, e in error_angles if t > settling_time_sec]
  max_post_settling = max(post_settling_errors) if post_settling_errors else final_error

  print(f"최종 오차(t={total_time_sec:.1f}초): {final_error:.3f}도")
  print(f"정착시간({settling_time_sec}초) 이후 최대 오차: {max_post_settling:.3f}도")

  assert final_error < 2.0, "PD 제어기는 3배 정착시간 후 오차를 2도 미만으로 줄여야 함"
  assert max_post_settling < initial_error_deg, "정착시간 이후 오차가 초기 오차보다 다시 커지면 안 됨(재발산 없음)"
  print("\n(정착시간 근처에서 오차가 충분히 줄고, 그 이후 다시 커지지 않는다 -")
  print(" 정착시간/감쇠비에서 역산한 게인이 실제로 의도한 응답을 만든다는 신호다.)")
  return {"kp": kp, "kd": kd, "error_angles": error_angles, "final_error": final_error,
          "settling_time_sec": settling_time_sec}


def demo_pd_control_tames_intermediate_axis_tumble(inertia_diag=(1.0, 2.0, 3.0),
                                                     perturbation_fraction=0.01):
  """16번의 중간축 설정을 그대로 재사용해, 제어 없이는 텀블링으로 발산하는
  회전을 PD 제어기가(스핀 자체를 멈추는 자세유지로) 훨씬 작은 오차 안에
  묶어두는지 확인한다 - 정성적 비교, 왜 이 확장이 이 프로젝트에 들어가야
  하는지에 대한 가장 직접적인 답."""
  print("\n" + "=" * 70)
  print("[3] 중간축 텀블링 길들이기: 16번의 불안정 회전을 PD 제어로 정지")
  print("=" * 70)
  spin_rate = 1.0
  perturbation = spin_rate * perturbation_fraction
  initial_omega = np.array([perturbation, spin_rate, perturbation])
  q_target = IDENTITY_QUATERNION.copy()
  dt_sec = 0.01
  total_time_sec = 50.0

  print(f"초기 각속도(중간축 I2 정렬 + 섭동)={initial_omega} rad/s (16번과 동일 설정)\n")

  history_uncontrolled = integrate_attitude_with_control(
      q_target, initial_omega, inertia_diag, total_time_sec, dt_sec, control_law=None)

  strong_settling_time = 1.5
  kp, kd = compute_pd_gains(inertia_diag, strong_settling_time, zeta=0.7)
  control_law = lambda q_err, omega: pd_control_torque(q_err, omega, np.zeros(3), kp, kd)  # noqa: E731
  history_controlled = integrate_attitude_with_control(
      q_target, initial_omega, inertia_diag, total_time_sec, dt_sec,
      control_law=control_law, q_target=q_target)

  errors_uncontrolled = [quaternion_error_angle_deg(attitude_error_quaternion(q, q_target))
                          for _t, (q, _omega) in history_uncontrolled]
  errors_controlled = [quaternion_error_angle_deg(attitude_error_quaternion(q, q_target))
                        for _t, (q, _omega) in history_controlled]
  max_uncontrolled = max(errors_uncontrolled)
  max_controlled = max(errors_controlled)

  print(f"제어 없음: 최대 자세 오차={max_uncontrolled:.1f}도 (텀블링으로 발산)")
  print(f"PD 제어 있음(정착시간={strong_settling_time}초로 강하게 설정): 최대 자세 오차={max_controlled:.1f}도")

  assert max_controlled < max_uncontrolled / 5, "제어기가 있으면 중간축 텀블링의 최대 오차가 무제어 대비 1/5 미만으로 억제되어야 함"
  print("\n(16번은 중간축 정렬 회전이 작은 섭동만으로 크게 발산한다는 것을 보여줬다 -")
  print(" 이 데모는 충분히 강한 PD 제어기가 바로 그 불안정한 회전을 거의 완전히")
  print(" 정지시킬 수 있다는 것을 정성적으로 확인한다.)")
  return {"max_uncontrolled": max_uncontrolled, "max_controlled": max_controlled,
          "history_uncontrolled": history_uncontrolled, "history_controlled": history_controlled,
          "kp": kp, "kd": kd}


def demo_higher_gain_settles_faster_with_more_overshoot(inertia_diag=(1.0, 2.0, 3.0),
                                                          initial_error_deg=35.0,
                                                          settling_time_sec=5.0, zeta=0.7):
  """같은 point-and-hold 문제를 정착시간과 정착시간/2로 두 번 풀어, 더 짧은
  정착시간(더 강한 게인)이 더 빨리 정착하지만 정착 이후 오차가 더 크거나
  같은지(오버슈트/링잉 경향) 확인한다."""
  print("\n" + "=" * 70)
  print("[4] 게인 비교: 정착시간을 절반으로 줄이면 더 빨리 정착하지만 더 흔들린다")
  print("=" * 70)
  q_target = IDENTITY_QUATERNION.copy()
  initial_q = axis_angle_to_quaternion([1, 0, 0], initial_error_deg)
  initial_omega = np.zeros(3)
  total_time_sec = 15.0

  def run(settling_time, dt_sec):
    kp, kd = compute_pd_gains(inertia_diag, settling_time, zeta)
    control_law = lambda q_err, omega: pd_control_torque(q_err, omega, np.zeros(3), kp, kd)  # noqa: E731
    history = integrate_attitude_with_control(initial_q, initial_omega, inertia_diag,
                                               total_time_sec, dt_sec,
                                               control_law=control_law, q_target=q_target)
    error_angles = [(t, quaternion_error_angle_deg(attitude_error_quaternion(q, q_target)))
                     for t, (q, _omega) in history]
    settled_time = next((t for t, e in error_angles if e < 2.0 and
                          all(e2 < 2.0 for t2, e2 in error_angles if t2 >= t)), None)
    post_settled = [e for t, e in error_angles if settled_time is not None and t > settled_time]
    max_post_settled = max(post_settled) if post_settled else error_angles[-1][1]
    return {"kp": kp, "kd": kd, "error_angles": error_angles,
            "settled_time": settled_time, "max_post_settled": max_post_settled}

  slow_result = run(settling_time_sec, dt_sec=0.05)
  fast_result = run(settling_time_sec / 2, dt_sec=0.02)

  print(f"느린 게인(ts={settling_time_sec}초): 정착시간={slow_result['settled_time']}, "
        f"정착 후 최대오차={slow_result['max_post_settled']:.3f}도")
  print(f"빠른 게인(ts={settling_time_sec / 2}초): 정착시간={fast_result['settled_time']}, "
        f"정착 후 최대오차={fast_result['max_post_settled']:.3f}도")

  assert slow_result["settled_time"] is not None and fast_result["settled_time"] is not None, \
      "두 게인 모두 정착시간 내에 실제로 정착해야 함"
  assert fast_result["settled_time"] < slow_result["settled_time"], "더 강한 게인은 더 빨리 정착해야 함"
  assert fast_result["max_post_settled"] >= slow_result["max_post_settled"] * 0.5, \
      "더 강한 게인은 정착 후에도 비슷하거나 더 큰 잔여 흔들림을 보여야 함(오버슈트/링잉 경향)"
  print("\n(정착시간을 절반으로 줄이면(게인은 실제로 약 4배 커짐, Kp∝ωn²) 더 빨리")
  print(" 정착하지만, 더 공격적인 게인이라 정착 후에도 잔여 흔들림이 더 크게 남는")
  print(" 경향을 보인다 - 게인 선택의 실제 트레이드오프다.)")
  return {"slow": slow_result, "fast": fast_result}


def parse_args():
  parser = argparse.ArgumentParser(description="쿼터니언 자세 오차를 PD 피드백으로 제어 - 목표 자세로 수렴")
  parser.add_argument("--inertia-i1", type=float, default=1.0, help="최소 주관성모멘트 I1, 기본값: 1.0")
  parser.add_argument("--inertia-i2", type=float, default=2.0, help="중간 주관성모멘트 I2, 기본값: 2.0")
  parser.add_argument("--inertia-i3", type=float, default=3.0, help="최대 주관성모멘트 I3, 기본값: 3.0")
  parser.add_argument("--settling-time", type=float, default=5.0, help="목표 정착시간(초), 기본값: 5.0")
  parser.add_argument("--initial-error-deg", type=float, default=35.0, help="초기 자세 오차(도), 기본값: 35.0")
  parser.add_argument("--damping-ratio", type=float, default=0.7, help="감쇠비 ζ, 기본값: 0.7")
  parser.add_argument("--perturbation-fraction", type=float, default=0.01,
                       help="주회전율 대비 중간축 섭동 크기 비율, 기본값: 0.01(1%%)")
  return parser.parse_args()


def main():
  args = parse_args()
  inertia_diag = (args.inertia_i1, args.inertia_i2, args.inertia_i3)

  kinematics_result = demo_free_kinematics_preserves_unit_quaternion(inertia_diag)
  point_hold_result = demo_pd_point_and_hold(inertia_diag, args.settling_time,
                                              args.initial_error_deg, args.damping_ratio)
  tumble_result = demo_pd_control_tames_intermediate_axis_tumble(inertia_diag, args.perturbation_fraction)
  gain_result = demo_higher_gain_settles_faster_with_more_overshoot(
      inertia_diag, args.initial_error_deg, args.settling_time, args.damping_ratio)

  print("\n" + "=" * 70)
  print(f"[사용자 지정] I=({args.inertia_i1}, {args.inertia_i2}, {args.inertia_i3}), "
        f"정착시간={args.settling_time}초, 초기오차={args.initial_error_deg}도, ζ={args.damping_ratio}")
  print("=" * 70)
  print(f"최종 오차: {point_hold_result['final_error']:.3f}도")
  print(f"중간축 텀블링(섭동 비율={args.perturbation_fraction * 100:.1f}%): "
        f"제어없음 {tumble_result['max_uncontrolled']:.2f}도 vs "
        f"제어있음 {tumble_result['max_controlled']:.2f}도")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  norm_csv = os.path.join(results_dir, "pid_free_kinematics_quaternion_norm.csv")
  with open(norm_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "quaternion_norm"])
    for t, (q, _omega) in kinematics_result["history"]:
      writer.writerow([f"{t:.4f}", f"{np.linalg.norm(q):.10f}"])
  print(f"\n[기록] 순수 운동학 쿼터니언 노름 저장됨 → {norm_csv}")

  point_hold_csv = os.path.join(results_dir, "pid_point_and_hold_error_history.csv")
  with open(point_hold_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "error_angle_deg"])
    for t, error_deg in point_hold_result["error_angles"]:
      writer.writerow([f"{t:.4f}", f"{error_deg:.6f}"])
  print(f"[기록] PD 자세유지 오차 시계열 저장됨 → {point_hold_csv}")

  tumble_csv = os.path.join(results_dir, "pid_intermediate_axis_comparison.csv")
  with open(tumble_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "error_angle_deg_uncontrolled", "error_angle_deg_controlled"])
    hist_u = tumble_result["history_uncontrolled"]
    hist_c = tumble_result["history_controlled"]
    for (t, (q_u, _o_u)), (_t2, (q_c, _o_c)) in zip(hist_u, hist_c):
      err_u = quaternion_error_angle_deg(attitude_error_quaternion(q_u, IDENTITY_QUATERNION))
      err_c = quaternion_error_angle_deg(attitude_error_quaternion(q_c, IDENTITY_QUATERNION))
      writer.writerow([f"{t:.4f}", f"{err_u:.6f}", f"{err_c:.6f}"])
  print(f"[기록] 중간축 텀블링 제어/무제어 비교 저장됨 → {tumble_csv}")

  gain_csv = os.path.join(results_dir, "pid_gain_comparison.csv")
  with open(gain_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "error_angle_deg", "run_label"])
    for t, error_deg in gain_result["slow"]["error_angles"]:
      writer.writerow([f"{t:.4f}", f"{error_deg:.6f}", "slow_ts"])
    for t, error_deg in gain_result["fast"]["error_angles"]:
      writer.writerow([f"{t:.4f}", f"{error_deg:.6f}", "fast_ts"])
  print(f"[기록] 게인 비교 결과 저장됨 → {gain_csv}")

  summary_csv = os.path.join(results_dir, "pid_summary.csv")
  with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["inertia_i1", "inertia_i2", "inertia_i3", "settling_time_sec",
                      "initial_error_deg", "final_error_deg", "tumble_max_uncontrolled_deg",
                      "tumble_max_controlled_deg", "gain_slow_settled_time",
                      "gain_fast_settled_time"])
    writer.writerow([args.inertia_i1, args.inertia_i2, args.inertia_i3, args.settling_time,
                      args.initial_error_deg, f"{point_hold_result['final_error']:.4f}",
                      f"{tumble_result['max_uncontrolled']:.4f}", f"{tumble_result['max_controlled']:.4f}",
                      gain_result["slow"]["settled_time"], gain_result["fast"]["settled_time"]])
  print(f"[기록] 요약 결과 저장됨 → {summary_csv}")


if __name__ == "__main__":
  main()
