"""
반작용휠 모멘텀 저장과 디새추레이션 - 22번의 "자세 재정렬은 델타-V 무료"라는
주장에 현실적인 예외를 만든다

22번(orbit_raise_and_reorient.py)은 핵심 개념 3에서 "자세 재정렬은 이
스크립트의 PD 토크 제어 모델에서 델타-V를 전혀 소모하지 않는다(반작용휠
디새추레이션 같은 추가 델타-V는 모델링하지 않음)"이라고 명시적으로 인정한
갭이 있다. 이 스크립트는 그 갭을 실제로 채운다 - 반작용휠이 제어 토크를
흡수하며 모멘텀을 저장하다가 포화되면, 디새추레이션(모멘텀 덤핑) 기동이
필요해지고 이것이 실제로 델타-V를 소모한다는 것을 보여준다. 동시에
자세동역학 그룹(16번 자유회전, 21번 PID 제어)을 3개 스크립트로 완성한다.

핵심 개념 1: 휠은 토크를 "흡수"해 몸체에 반작용 토크를 돌려준다
  PD 제어 법칙은 여전히 21번과 똑같은 공식으로 "몸체가 원하는 토크"를
  계산한다(tau_body_desired = -Kp*q_err[1:] - Kd*ω). 하지만 이 토크는
  추력기가 아니라 휠 모터가 만든다 - 모터가 휠에 토크(torque_cmd)를 가해
  휠을 가속시키면, 뉴턴 제3법칙에 의해 몸체는 정확히 반대 부호의 반작용
  토크를 받는다. 즉 torque_cmd = -tau_body_desired이고, 몸체 방정식은
      I*dω/dt = torque_external + tau_body_desired - ω x (I*ω + h_wheel)
      dh_wheel/dt = torque_cmd = -tau_body_desired
  이다. 처음 구현에서 torque_cmd를 몸체 방정식에서 또 빼는 실수를 저질러
  (이미 반영된 반작용을 중복 계산해 domega에 양의 댐핑 항이 생김) 각속도가
  10초 안에 발산했다 - 이 부호 오류를 회귀 테스트로 명시적으로 가드한다.

핵심 개념 2: 외부 토크가 없으면 총 각운동량의 "크기"는 보존된다
  몸체와 휠을 합친 전체 시스템의 각운동량은 L_total = I*ω + h_wheel이다.
  외부 토크가 0이면 |L_total|은 보존되지만(관성좌표계에서 전체 각운동량
  벡터가 고정되므로), 몸체좌표계로 표현한 벡터 자체는 몸체가 회전하면서
  계속 방향이 바뀐다 - 16번이 각속도 벡터가 아니라 각운동량의 "크기"만
  보존을 검증하는 것과 같은 이유다. 이 "크기만 보존" 항등식이 공식 자체의
  부호/항 오류를 잡아내는 가장 강력한 검증이다.

핵심 개념 3: 지속적 외란이 휠 모멘텀을 조용히, 선형으로 쌓는다
  중력경도 토크 같은 작은 상수 외란이 계속 작용하면, 휠은 이를 상쇄하기
  위해 거의 일정한 역토크를 계속 가해야 하고, 그 결과 휠 모멘텀이 시간에
  거의 정확히 선형으로(h(t) ≈ 외란 크기 x t) 쌓인다 - 포인팅 오차는 거의
  0으로 유지되므로 겉보기엔 아무 문제가 없어 보이지만, 휠은 조용히 포화를
  향해 가고 있다.

핵심 개념 4: 포화는 점진적 열화가 아니라 급격한 실패다
  휠 모멘텀이 용량에 도달하면(이 스크립트는 축별 하드 컷오프를 쓴다 - 포화된
  축은 방향에 상관없이 토크를 완전히 0으로 자른다), 그 순간부터 외란을
  상쇄할 수 없어 포인팅 오차가 빠르게 커진다. 처음에는 "모멘텀을 줄이는
  방향이면 통과시킨다"는 방향 인식 클램프를 시도했는데, 자세 오차가 180도
  근처(쿼터니언 오차 벡터부가 거의 0인 특이점 근처)에서는 PD 법칙이 아주 큰
  토크를 명령할 수 있어, "줄이는 방향"이라는 판단이 맞더라도 그 토크가 한
  적분 스텝 안에 모멘텀을 반대 극성의 훨씬 큰 값으로 오버슈트시켜버리는
  문제를 실측으로 발견했다(0.05 -> -0.33으로 한 스텝에 점프) - 방향 무관
  전면 차단이 더 안전하고, "포화되면 휠 자체 모터로는 더 이상 손쓸 수 없고
  외부 디새추레이션이 반드시 필요하다"는 메시지도 오히려 더 분명해진다.
  디새추레이션(모멘텀 덤핑) 기동 없이는 자세가 회복되지 않는다 -
  "완만한 성능 저하"가 아니라 "하드 실패"다.

핵심 개념 5: 디새추레이션은 작지만 0이 아닌 델타-V를 쓴다
  모멘텀 암 d에 있는 추력기가 토크 tau=F*d를 내면, 모멘텀 Δh를 덤핑하는 데
  걸리는 연소시간은 Δh/tau이고, 그동안 전달되는 임펄스는 F*t_burn이라
  델타-V는 Δh/(d*m_sat)가 된다(로켓방정식 깊이는 다루지 않는 단순화 - 17번의
  평면변경 공식과 같은 수준의 닫힌 형태).

단순화: 축마다 독립된 휠 1개씩, 총 3개 휠(피라미드/경사 배치는 다루지 않음)만
가정한다. 포화는 축별 하드 컷오프로 단순화한다(연속적인 토크 감쇠가 아님).
디새추레이션 자체는 순간적인 것으로 가정한다(연소시간 동안의 자세 변화는
무시). 모든 상수(휠 용량, 외란 크기, 모멘텀 암, 위성 질량)는 특정 문헌값이
아니라 수 분 내에 포화가 일어나는 시연 가능한 시간축을 만들기 위한 예시값
이다.

22번과의 관계: 22번의 PD 제어 공식(pd_control_torque와 동일한 형태)을
그대로 재사용하되, 그 출력의 부호를 반전해 휠 모터 명령으로 쓴다. 21번
(pid_attitude_control.py)의 쿼터니언/게인 함수를 importlib로 재사용한다.
06번(hohmann_transfer.py)의 hohmann_transfer_delta_v를 재사용해 디새추레이션
비용을 22번 기본 전이의 호만 델타-V와 라이브로 비교한다.
"""

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)

_PID_PATH = os.path.join(_THIS_DIR, "pid_attitude_control.py")
_pid_spec = importlib.util.spec_from_file_location("pid_module", _PID_PATH)
pid = importlib.util.module_from_spec(_pid_spec)
_pid_spec.loader.exec_module(pid)

_HOHMANN_PATH = os.path.join(_ROOT_DIR, "missions", "hohmann_transfer.py")
_hohmann_spec = importlib.util.spec_from_file_location("hohmann_module", _HOHMANN_PATH)
hohmann = importlib.util.module_from_spec(_hohmann_spec)
_hohmann_spec.loader.exec_module(hohmann)

IDENTITY_QUATERNION = pid.IDENTITY_QUATERNION

DEFAULT_WHEEL_CAPACITY_NMS = 0.05  # 예시값(특정 휠 사양 아님) - 아래 외란과 조합해 ~100초 포화 시연
DEFAULT_DISTURBANCE_TORQUE_NM = 5e-4  # N*m, 상수 외란(예: 보정 안 된 중력경도 토크) 예시값
DEFAULT_THRUSTER_MOMENT_ARM_M = 0.3  # m, 예시 소형위성 추력기 모멘트암
DEFAULT_SPACECRAFT_MASS_KG = 100.0  # kg, 22번의 관성모멘트(50,70,90) 규모와 일관된 예시 질량


def wheel_augmented_state_derivative(state, inertia_diag, torque_external, torque_cmd):
  """10차원 state=[q0,q1,q2,q3,ω1,ω2,ω3,h1,h2,h3]의 시간미분. 몸체는
  torque_external과 PD 법칙이 원한 토크(torque_cmd의 반작용, 즉 -torque_cmd가
  아니라 +torque_cmd 그 자체가 이미 "몸체가 원하는 토크"의 부호로 전달된다 -
  호출부에서 torque_cmd를 "몸체 토크"로 넘기고, 휠은 그 반대 부호로 모멘텀이
  바뀐다는 계약을 쓴다). 구체적으로:
      I*dω/dt = torque_external + torque_cmd - ω x (I*ω + h_wheel)
      dh_wheel/dt = -torque_cmd
  (torque_cmd를 몸체 방정식에서 다시 빼면 이미 반영된 반작용을 중복 계산해
  양의 피드백으로 발산한다 - 이 부호가 가장 위험한 지점이다)."""
  q, omega, h_wheel = state[:4], state[4:7], state[7:10]
  dq = pid.quaternion_kinematics_derivative(q, omega)
  l_total = inertia_diag * omega + h_wheel
  domega = (torque_external + torque_cmd - np.cross(omega, l_total)) / inertia_diag
  dh_wheel = -torque_cmd
  return np.concatenate([dq, domega, dh_wheel])


def rk4_step_with_wheel(state, dt_sec, inertia_diag, torque_external, torque_cmd):
  """21번 rk4_step과 동일한 10차원 RK4 클론 - torque_external/torque_cmd는
  4단계 동안 고정, 쿼터니언 재정규화는 전체 조합 이후 한 번만."""
  k1 = wheel_augmented_state_derivative(state, inertia_diag, torque_external, torque_cmd)
  k2 = wheel_augmented_state_derivative(state + dt_sec / 2 * k1, inertia_diag, torque_external, torque_cmd)
  k3 = wheel_augmented_state_derivative(state + dt_sec / 2 * k2, inertia_diag, torque_external, torque_cmd)
  k4 = wheel_augmented_state_derivative(state + dt_sec * k3, inertia_diag, torque_external, torque_cmd)
  new_state = state + dt_sec / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  new_state[:4] = pid.normalize_quaternion(new_state[:4])
  return new_state


def clamp_wheel_torque(tau_body_desired, h_wheel, capacity_nms):
  """축별 하드 컷오프: |h_wheel[i]|가 용량 이상인 축은 방향에 상관없이 토크를
  완전히 0으로 자른다(그 축의 휠 모터가 포화 상태에서는 아예 작동을 멈춘다는
  단순화). 처음에는 "모멘텀을 줄이는 방향이면 통과시킨다"는 방향 인식
  클램프를 시도했는데, 포인팅 오차가 180도 근처(쿼터니언 오차의 벡터부가
  거의 0이 되는 특이점 근처)에서는 PD 법칙이 아주 큰 토크를 명령할 수 있어,
  "줄이는 방향"이라는 판단이 맞더라도 그 토크가 한 적분 스텝 안에 모멘텀을
  반대 극성의 훨씬 큰 값으로 오버슈트시켜 버리는 문제를 실측으로 발견했다
  (0.05 -> -0.33으로 한 스텝에 점프). 방향 무관 전면 차단이 더 안전하고,
  "포화되면 휠 자체 모터로는 더 이상 손쓸 수 없고 외부 디새추레이션이
  반드시 필요하다"는 메시지도 오히려 더 분명해진다."""
  out = np.array(tau_body_desired, dtype=float).copy()
  for i in range(3):
    if abs(h_wheel[i]) >= capacity_nms:
      out[i] = 0.0
  return out


def wheel_reaction_control_torque(q_err, omega, kp, kd, h_wheel, capacity_nms):
  """pd_control_torque와 동일한 공식으로 tau_body_desired를 구하고, 포화된
  축에서는 clamp_wheel_torque로 자른 뒤 몸체 토크로 반환한다. 휠 모터 명령은
  이 값의 음수(-결과)로 wheel_augmented_state_derivative의 torque_cmd 자리에
  들어간다."""
  tau_body_desired = pid.pd_control_torque(q_err, omega, np.zeros(3), kp, kd)
  return clamp_wheel_torque(tau_body_desired, h_wheel, capacity_nms)


def desaturation_delta_v_cost(delta_h_nms, moment_arm_m, spacecraft_mass_kg):
  """deltaV = delta_h_nms / (moment_arm_m * spacecraft_mass_kg), km/s.
  로켓방정식 깊이는 다루지 않는 단순화(17번 평면변경 공식과 같은 수준의
  닫힌 형태): 모멘텀 암 d의 추력기가 토크 tau=F*d를 내 모멘텀 Δh를 덤핑하는
  연소시간 Δh/tau 동안 전달되는 임펄스 F*t_burn을 질량으로 나눈 값."""
  delta_v_m_s = delta_h_nms / (moment_arm_m * spacecraft_mass_kg)
  return delta_v_m_s / 1000.0


def integrate_attitude_with_wheel(initial_q, initial_omega, initial_h_wheel, inertia_diag,
                                   total_time_sec, dt_sec, kp, kd, capacity_nms,
                                   torque_external_fn=None, q_target=None):
  """21번 integrate_attitude_with_control의 휠 버전. torque_external_fn(t)
  -> 3-vector, 기본값은 영벡터. 반환: [(t, (q, omega, h_wheel)), ...]."""
  if torque_external_fn is None:
    torque_external_fn = lambda t: np.zeros(3)  # noqa: E731
  if q_target is None:
    q_target = IDENTITY_QUATERNION.copy()

  state = np.concatenate([initial_q, initial_omega, initial_h_wheel]).astype(float)
  t = 0.0
  history = [(t, (state[:4].copy(), state[4:7].copy(), state[7:10].copy()))]

  while t < total_time_sec - 1e-9:
    step = min(dt_sec, total_time_sec - t)
    q, omega, h_wheel = state[:4], state[4:7], state[7:10]
    q_err = pid.attitude_error_quaternion(q, q_target)
    tau_body = wheel_reaction_control_torque(q_err, omega, kp, kd, h_wheel, capacity_nms)
    torque_external = torque_external_fn(t)
    torque_cmd = tau_body  # wheel_augmented_state_derivative가 이미 +torque_cmd를 몸체 토크로 씀
    state = rk4_step_with_wheel(state, step, inertia_diag, torque_external, torque_cmd)
    t += step
    history.append((t, (state[:4].copy(), state[4:7].copy(), state[7:10].copy())))

  return history


def apply_desaturation_burn(h_wheel, omega):
  """h_wheel과 몸체 각속도를 순간적으로 0으로 리셋한다. 실제 모멘텀 덤핑
  추력기 연소는 몸체에도 토크를 가하므로, 이를 "휠 모멘텀 제거와 동시에
  몸체를 정지시키는 순간적 안정화 기동"으로 단순화한다(연소시간 동안의
  자세 변화 자체는 무시). 휠 모멘텀만 지우고 이미 커져 있던 몸체 각속도를
  그대로 두면, PD 제어기가 그 각속도를 상쇄하는 데 다시 휠 모멘텀을 많이
  써야 해서 재포화·재발산이 반복되는 비현실적인 결과가 나온다 - 실측으로
  확인한 뒤 이 단순화를 선택했다. 반환: 덤핑 직전 |h_wheel|(비용 계산용
  delta_h)."""
  delta_h = np.linalg.norm(h_wheel)
  h_wheel[:] = 0.0
  omega[:] = 0.0
  return delta_h


def demo_total_angular_momentum_is_conserved_without_external_torque(inertia_diag=(1.0, 2.0, 3.0)):
  """외부 토크가 전혀 없을 때, 내부 휠 토크만으로 몸체와 휠을 합친 총
  각운동량의 "크기"가 보존되는지 확인한다 - 공식 자체의 부호/항 오류를 잡는
  가장 강력한 검증."""
  print("=" * 70)
  print("[1] 총 각운동량 보존: 외부 토크 없이 휠끼리만 모멘텀을 주고받아도")
  print("=" * 70)
  inertia = np.array(inertia_diag)
  q = IDENTITY_QUATERNION.copy()
  omega = np.array([0.05, -0.03, 0.02])
  h_wheel = np.zeros(3)
  dt_sec = 0.001
  total_time_sec = 20.0

  l0 = np.linalg.norm(inertia * omega + h_wheel)
  state = np.concatenate([q, omega, h_wheel])
  t = 0.0
  max_drift = 0.0
  while t < total_time_sec - 1e-9:
    step = min(dt_sec, total_time_sec - t)
    # 내부 휠 토크만(외부 토크 0), 시간에 따라 변하는 사인파로 흔들어본다
    torque_cmd = np.array([0.01 * np.sin(t), 0.01 * np.cos(2 * t), 0.005 * np.sin(3 * t)])
    state = rk4_step_with_wheel(state, step, inertia, np.zeros(3), torque_cmd)
    t += step
    omega_now, h_now = state[4:7], state[7:10]
    l_now = np.linalg.norm(inertia * omega_now + h_now)
    max_drift = max(max_drift, abs(l_now - l0))

  print(f"초기 |L_total|={l0:.6f}, 20초 적분 중 최대 드리프트={max_drift:.3e}")
  assert max_drift < 1e-6, "외부 토크가 없으면 몸체+휠 총 각운동량의 크기는 보존돼야 함"
  print("\n(몸체좌표계로 표현한 L_total 벡터 자체는 몸체가 회전하며 방향이")
  print(" 계속 바뀌지만, 크기는 전체 적분 동안 거의 정확히 보존된다 - 16번이")
  print(" 각속도 벡터가 아니라 각운동량 크기만 보존을 검증하는 것과 같은 이유다.)")
  return {"l0": l0, "max_drift": max_drift}


def demo_sustained_disturbance_saturates_wheel_while_pointing_stays_tight(
    inertia_diag=(1.0, 2.0, 3.0), settling_time_sec=5.0, zeta=0.7,
    disturbance_torque_nm=DEFAULT_DISTURBANCE_TORQUE_NM, dt_sec=0.05, total_time_sec=600.0):
  """목표 자세(identity)에서 시작해, 포화 없이(용량을 사실상 무한대로 두고)
  상수 외란 토크를 계속 받는다. 휠 모멘텀이 외란 크기와 거의 같은 속도로
  선형으로 쌓이는 동안, 포인팅 오차는 거의 0으로 유지된다는 것을 확인한다 -
  "조용히 쌓이는 문제"라는 핵심 메시지."""
  print("\n" + "=" * 70)
  print("[2] 지속 외란: 포인팅은 완벽해 보이지만 휠 모멘텀은 선형으로 쌓인다")
  print("=" * 70)
  inertia = np.array(inertia_diag)
  kp, kd = pid.compute_pd_gains(inertia_diag, settling_time_sec, zeta)
  disturbance = np.array([0.0, 0.0, disturbance_torque_nm])
  torque_external_fn = lambda t: disturbance  # noqa: E731
  huge_capacity = 1e6

  history = integrate_attitude_with_wheel(
      IDENTITY_QUATERNION.copy(), np.zeros(3), np.zeros(3), inertia,
      total_time_sec, dt_sec, kp, kd, huge_capacity, torque_external_fn)

  max_error_deg = 0.0
  for t, (q, _omega, _h) in history:
    q_err = pid.attitude_error_quaternion(q, IDENTITY_QUATERNION)
    max_error_deg = max(max_error_deg, pid.quaternion_error_angle_deg(q_err))

  final_h_mag = np.linalg.norm(history[-1][1][2])
  growth_rate = final_h_mag / total_time_sec

  print(f"외란 토크={disturbance_torque_nm:.1e}N*m, 적분시간={total_time_sec}초")
  print(f"최종 휠 모멘텀 크기={final_h_mag:.4f}N*m*s, 평균 증가율={growth_rate:.3e}N*m*s/s")
  print(f"적분 중 최대 포인팅 오차={max_error_deg:.4f}도")

  assert abs(growth_rate - disturbance_torque_nm) < 0.01 * disturbance_torque_nm, \
      "휠 모멘텀 증가율은 외란 토크 크기와 거의 같아야 함(휠이 상수 외란을 거의 완벽히 상쇄)"
  assert max_error_deg < 1.0, "포화 전에는 포인팅 오차가 거의 0으로 유지돼야 함 - 겉보기엔 문제가 안 보임"

  print("\n(휠 모멘텀이 외란 토크와 거의 같은 속도로 선형으로 쌓이는 동안")
  print(" 포인팅 오차는 1도 미만으로 유지된다 - 문제가 조용히 누적되고 있다는")
  print(" 것을 포인팅 오차만 봐서는 전혀 알 수 없다.)")
  return {"max_error_deg": max_error_deg, "growth_rate": growth_rate, "final_h_mag": final_h_mag}


def demo_saturation_causes_pointing_loss_without_desaturation(
    inertia_diag=(1.0, 2.0, 3.0), settling_time_sec=5.0, zeta=0.7,
    wheel_capacity_nms=DEFAULT_WHEEL_CAPACITY_NMS,
    disturbance_torque_nm=DEFAULT_DISTURBANCE_TORQUE_NM, dt_sec=0.05, total_time_sec=300.0):
  """데모2와 같은 설정에 실제 휠 용량을 적용한다. 포화 시점이
  capacity/disturbance 예측과 거의 정확히 일치하고, 포화 이후 포인팅 오차가
  빠르게 발산하는 것을 확인한다 - 포화는 완만한 성능 저하가 아니라 하드
  실패다."""
  print("\n" + "=" * 70)
  print("[3] 포화: 디새추레이션 없이는 포인팅이 하드 실패로 무너진다")
  print("=" * 70)
  inertia = np.array(inertia_diag)
  kp, kd = pid.compute_pd_gains(inertia_diag, settling_time_sec, zeta)
  disturbance = np.array([0.0, 0.0, disturbance_torque_nm])
  torque_external_fn = lambda t: disturbance  # noqa: E731

  history = integrate_attitude_with_wheel(
      IDENTITY_QUATERNION.copy(), np.zeros(3), np.zeros(3), inertia,
      total_time_sec, dt_sec, kp, kd, wheel_capacity_nms, torque_external_fn)

  sat_time = None
  max_error_before_sat = 0.0
  max_error_after_sat = 0.0
  for t, (q, _omega, h_wheel) in history:
    q_err = pid.attitude_error_quaternion(q, IDENTITY_QUATERNION)
    err_deg = pid.quaternion_error_angle_deg(q_err)
    if np.linalg.norm(h_wheel) >= wheel_capacity_nms:
      if sat_time is None:
        sat_time = t
      max_error_after_sat = max(max_error_after_sat, err_deg)
    else:
      max_error_before_sat = max(max_error_before_sat, err_deg)

  predicted_sat_time = wheel_capacity_nms / disturbance_torque_nm
  print(f"휠 용량={wheel_capacity_nms}N*m*s, 외란={disturbance_torque_nm:.1e}N*m")
  print(f"포화 시점={sat_time:.2f}초 (예측={predicted_sat_time:.2f}초)")
  print(f"포화 전 최대 오차={max_error_before_sat:.4f}도, 포화 후 최대 오차={max_error_after_sat:.2f}도")

  assert sat_time is not None, "이 설정에서는 반드시 포화가 일어나야 함"
  assert abs(sat_time - predicted_sat_time) < 5.0, "포화 시점은 capacity/disturbance 선형 예측과 거의 일치해야 함"
  assert max_error_after_sat > 30.0, "포화 후에는 포인팅 오차가 크게 발산해야 함(하드 실패)"

  print("\n(포화 시점이 단순한 선형 예측(용량/외란)과 거의 정확히 일치한다 -")
  print(" 포화가 일어나는 순간까지는 모멘텀이 꾸준히 쌓이기만 하기 때문이다.")
  print(" 포화 이후에는 외란을 더 이상 상쇄할 수 없어 포인팅이 급격히 무너진다.)")
  return {"sat_time": sat_time, "predicted_sat_time": predicted_sat_time,
          "max_error_before_sat": max_error_before_sat, "max_error_after_sat": max_error_after_sat,
          "history": history}


def demo_desaturation_burn_recovers_control_at_a_real_delta_v_cost(
    inertia_diag=(1.0, 2.0, 3.0), settling_time_sec=5.0, zeta=0.7,
    wheel_capacity_nms=DEFAULT_WHEEL_CAPACITY_NMS,
    disturbance_torque_nm=DEFAULT_DISTURBANCE_TORQUE_NM,
    moment_arm_m=DEFAULT_THRUSTER_MOMENT_ARM_M, spacecraft_mass_kg=DEFAULT_SPACECRAFT_MASS_KG,
    dt_sec=0.05):
  """데모3과 같은 시나리오를 포화 시점(capacity/disturbance)에 정확히
  디새추레이션 burn을 적용한 뒤, 다음 포화가 다시 일어나기 전
  (capacity/disturbance 창 안)까지 회복 구간을 본다 - 외란이 멈추지 않는
  한 한 번의 디새추레이션으로 영원히 해결되지는 않는다(17번 station-keeping의
  "주기적 보정"과 같은 구조). 디새추레이션을 포화 직후 신속하게 실행하는
  것이 중요하다는 것도 실측으로 확인했다 - 포화를 10초 넘게 방치하면(오차가
  이미 수 도 이상 커진 뒤라면) PD 법칙이 거의 반대쪽 자세를 향해 큰 토크를
  명령하게 되고, 단 한 번의 적분 스텝 안에 휠이 반대 극성으로 재포화되며
  회복에 실패하는 것을 발견했다. 디새추레이션 burn은 휠 모멘텀뿐 아니라
  몸체 각속도도 순간적으로 0으로 리셋한다고 단순화한다. 포인팅이 회복되고,
  그 비용이 0보다 크지만 22번 기본 호만 전이 델타-V보다는 훨씬 작다는 것을
  확인한다 - 22번의 "델타-V 0"이라는 주장에 현실적인 예외를 다는 캡스톤
  데모."""
  print("\n" + "=" * 70)
  print("[4] 디새추레이션: 포인팅을 회복시키지만 0이 아닌 델타-V가 든다")
  print("=" * 70)
  inertia = np.array(inertia_diag)
  kp, kd = pid.compute_pd_gains(inertia_diag, settling_time_sec, zeta)
  disturbance = np.array([0.0, 0.0, disturbance_torque_nm])
  torque_external_fn = lambda t: disturbance  # noqa: E731

  time_before_desat_sec = wheel_capacity_nms / disturbance_torque_nm  # 포화되는 바로 그 시점에 디새추레이션
  time_after_desat_sec = time_before_desat_sec  # 다음 포화가 다시 오기 전까지만 본다

  pre_history = integrate_attitude_with_wheel(
      IDENTITY_QUATERNION.copy(), np.zeros(3), np.zeros(3), inertia,
      time_before_desat_sec, dt_sec, kp, kd, wheel_capacity_nms, torque_external_fn)
  q_sat, omega_sat, h_sat = pre_history[-1][1]
  omega_sat = omega_sat.copy()
  h_sat = h_sat.copy()

  delta_h = apply_desaturation_burn(h_sat, omega_sat)
  desat_dv_km_s = desaturation_delta_v_cost(delta_h, moment_arm_m, spacecraft_mass_kg)

  post_history = integrate_attitude_with_wheel(
      q_sat, omega_sat, h_sat, inertia,
      time_after_desat_sec, dt_sec, kp, kd, wheel_capacity_nms, torque_external_fn)
  final_q, _final_omega, _final_h = post_history[-1][1]
  final_err = pid.quaternion_error_angle_deg(pid.attitude_error_quaternion(final_q, IDENTITY_QUATERNION))

  r1_km, r2_km = 6978.0, 7378.0
  _dv1, _dv2, hohmann_dv_km_s = hohmann.hohmann_transfer_delta_v(r1_km, r2_km)
  ratio_pct = desat_dv_km_s / hohmann_dv_km_s * 100

  print(f"디새추레이션 직전 휠 모멘텀 크기(=덤핑량)={delta_h:.4f}N*m*s")
  print(f"디새추레이션 델타-V={desat_dv_km_s * 1e6:.3f}mm/s")
  print(f"22번 기본 호만 전이(r1={r1_km}km, r2={r2_km}km) 델타-V={hohmann_dv_km_s * 1000:.2f}m/s")
  print(f"비율(디새추레이션/호만 전이)={ratio_pct:.4f}%")
  print(f"디새추레이션 {time_after_desat_sec}초 후 포인팅 오차={final_err:.4f}도")

  assert final_err < 2.0, "디새추레이션 후에는 PD 제어기가 다시 목표 자세로 수렴해야 함"
  assert 0 < desat_dv_km_s < hohmann_dv_km_s, \
      "디새추레이션 비용은 0보다 크지만 궤도 전이 델타-V보다는 작아야 함 - 22번의 '정확히 0'이라는 주장에 작지만 실제인 예외"

  print("\n(디새추레이션이 포인팅을 완전히 회복시키지만, 그 비용은 정확히 0이")
  print(" 아니다 - 22번이 '자세 재정렬은 델타-V를 전혀 소모하지 않는다'고 한")
  print(" 주장은 휠이 무한정 모멘텀을 저장할 수 있다는 암묵적 가정 위에서만")
  print(" 성립한다는 것을 보여준다. 외란이 멈추지 않는 한 휠은 같은 시간 간격")
  print(" (여기서는 용량/외란)마다 다시 포화되므로, 디새추레이션도 17번의")
  print(" 궤도 유지처럼 주기적으로 반복돼야 한다 - 한 번으로 영원히 끝나는")
  print(" 일회성 비용이 아니다.)")
  return {"delta_h": delta_h, "desat_dv_km_s": desat_dv_km_s, "hohmann_dv_km_s": hohmann_dv_km_s,
          "ratio_pct": ratio_pct, "final_err_deg": final_err,
          "pre_history": pre_history, "post_history": post_history}


def parse_args():
  parser = argparse.ArgumentParser(description="반작용휠 모멘텀 저장, 포화, 디새추레이션이 자세제어에 미치는 영향 계산")
  parser.add_argument("--inertia-i1", type=float, default=1.0, help="관성모멘트 I1, 기본값: 1.0")
  parser.add_argument("--inertia-i2", type=float, default=2.0, help="관성모멘트 I2, 기본값: 2.0")
  parser.add_argument("--inertia-i3", type=float, default=3.0, help="관성모멘트 I3, 기본값: 3.0")
  parser.add_argument("--settling-time", type=float, default=5.0, help="PD 제어 목표 정착시간(초), 기본값: 5.0")
  parser.add_argument("--damping-ratio", type=float, default=0.7, help="PD 제어 감쇠비, 기본값: 0.7")
  parser.add_argument("--wheel-capacity-nms", type=float, default=DEFAULT_WHEEL_CAPACITY_NMS,
                       help=f"휠 모멘텀 포화 용량(N*m*s), 기본값: {DEFAULT_WHEEL_CAPACITY_NMS}")
  parser.add_argument("--disturbance-torque-nm", type=float, default=DEFAULT_DISTURBANCE_TORQUE_NM,
                       help=f"상수 외란 토크(N*m), 기본값: {DEFAULT_DISTURBANCE_TORQUE_NM}")
  parser.add_argument("--thruster-moment-arm-m", type=float, default=DEFAULT_THRUSTER_MOMENT_ARM_M,
                       help=f"디새추레이션 추력기 모멘트암(m), 기본값: {DEFAULT_THRUSTER_MOMENT_ARM_M}")
  parser.add_argument("--spacecraft-mass-kg", type=float, default=DEFAULT_SPACECRAFT_MASS_KG,
                       help=f"위성 질량(kg), 기본값: {DEFAULT_SPACECRAFT_MASS_KG}")
  return parser.parse_args()


def main():
  args = parse_args()
  inertia_diag = (args.inertia_i1, args.inertia_i2, args.inertia_i3)

  demo_total_angular_momentum_is_conserved_without_external_torque(inertia_diag)
  demo_sustained_disturbance_saturates_wheel_while_pointing_stays_tight(
      inertia_diag, args.settling_time, args.damping_ratio, args.disturbance_torque_nm)
  saturation_result = demo_saturation_causes_pointing_loss_without_desaturation(
      inertia_diag, args.settling_time, args.damping_ratio,
      args.wheel_capacity_nms, args.disturbance_torque_nm)
  desat_result = demo_desaturation_burn_recovers_control_at_a_real_delta_v_cost(
      inertia_diag, args.settling_time, args.damping_ratio,
      args.wheel_capacity_nms, args.disturbance_torque_nm,
      args.thruster_moment_arm_m, args.spacecraft_mass_kg)

  print("\n" + "=" * 70)
  print(f"[사용자 지정] I={inertia_diag}, 정착시간={args.settling_time}초, "
        f"휠용량={args.wheel_capacity_nms}N*m*s, 외란={args.disturbance_torque_nm:.1e}N*m")
  print("=" * 70)
  print(f"포화 시점={saturation_result['sat_time']:.2f}초, "
        f"디새추레이션 델타-V={desat_result['desat_dv_km_s'] * 1e6:.3f}mm/s")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  conservation_csv = os.path.join(results_dir, "reaction_wheel_momentum_conservation_history.csv")
  inertia = np.array(inertia_diag)
  q = IDENTITY_QUATERNION.copy()
  omega = np.array([0.05, -0.03, 0.02])
  h_wheel = np.zeros(3)
  state = np.concatenate([q, omega, h_wheel])
  with open(conservation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "l_total_mag_nms", "h_wheel_mag_nms"])
    t = 0.0
    dt_sec = 0.001
    total_time_sec = 20.0
    while t < total_time_sec - 1e-9:
      step = min(dt_sec, total_time_sec - t)
      torque_cmd = np.array([0.01 * np.sin(t), 0.01 * np.cos(2 * t), 0.005 * np.sin(3 * t)])
      state = rk4_step_with_wheel(state, step, inertia, np.zeros(3), torque_cmd)
      t += step
      omega_now, h_now = state[4:7], state[7:10]
      l_mag = np.linalg.norm(inertia * omega_now + h_now)
      writer.writerow([f"{t:.4f}", f"{l_mag:.10f}", f"{np.linalg.norm(h_now):.10f}"])
  print(f"\n[기록] 각운동량 보존 시계열 저장됨 → {conservation_csv}")

  kp, kd = pid.compute_pd_gains(inertia_diag, args.settling_time, args.damping_ratio)
  disturbance = np.array([0.0, 0.0, args.disturbance_torque_nm])
  torque_external_fn = lambda t: disturbance  # noqa: E731

  accumulation_csv = os.path.join(results_dir, "reaction_wheel_disturbance_accumulation.csv")
  history = integrate_attitude_with_wheel(
      IDENTITY_QUATERNION.copy(), np.zeros(3), np.zeros(3), inertia,
      600.0, 0.05, kp, kd, 1e6, torque_external_fn)
  with open(accumulation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "h_wheel_mag_nms", "error_angle_deg"])
    for t, (qv, _omega, hv) in history:
      err_deg = pid.quaternion_error_angle_deg(pid.attitude_error_quaternion(qv, IDENTITY_QUATERNION))
      writer.writerow([f"{t:.2f}", f"{np.linalg.norm(hv):.6f}", f"{err_deg:.6f}"])
  print(f"[기록] 외란 축적 시계열 저장됨 → {accumulation_csv}")

  saturation_csv = os.path.join(results_dir, "reaction_wheel_saturation_pointing_loss.csv")
  with open(saturation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "h_wheel_mag_nms", "error_angle_deg", "saturated"])
    for t, (qv, _omega, hv) in saturation_result["history"]:
      err_deg = pid.quaternion_error_angle_deg(pid.attitude_error_quaternion(qv, IDENTITY_QUATERNION))
      saturated = np.linalg.norm(hv) >= args.wheel_capacity_nms
      writer.writerow([f"{t:.2f}", f"{np.linalg.norm(hv):.6f}", f"{err_deg:.6f}", saturated])
  print(f"[기록] 포화 궤적 저장됨 → {saturation_csv}")

  recovery_csv = os.path.join(results_dir, "reaction_wheel_desaturation_recovery.csv")
  with open(recovery_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "h_wheel_mag_nms", "error_angle_deg", "phase"])
    for t, (qv, _omega, hv) in desat_result["pre_history"]:
      err_deg = pid.quaternion_error_angle_deg(pid.attitude_error_quaternion(qv, IDENTITY_QUATERNION))
      writer.writerow([f"{t:.2f}", f"{np.linalg.norm(hv):.6f}", f"{err_deg:.6f}", "before_desat"])
    pre_end_t = desat_result["pre_history"][-1][0]
    for t, (qv, _omega, hv) in desat_result["post_history"]:
      err_deg = pid.quaternion_error_angle_deg(pid.attitude_error_quaternion(qv, IDENTITY_QUATERNION))
      writer.writerow([f"{pre_end_t + t:.2f}", f"{np.linalg.norm(hv):.6f}", f"{err_deg:.6f}", "after_desat"])
  print(f"[기록] 디새추레이션 회복 궤적 저장됨 → {recovery_csv}")

  summary_csv = os.path.join(results_dir, "reaction_wheel_summary.csv")
  with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["wheel_capacity_nms", "disturbance_torque_nm", "time_to_saturation_sec",
                      "delta_h_dumped_nms", "desaturation_delta_v_km_s", "orbit_raise_delta_v_km_s",
                      "final_error_after_recovery_deg"])
    writer.writerow([args.wheel_capacity_nms, args.disturbance_torque_nm,
                      f"{saturation_result['sat_time']:.4f}", f"{desat_result['delta_h']:.6f}",
                      f"{desat_result['desat_dv_km_s']:.10f}", f"{desat_result['hohmann_dv_km_s']:.8f}",
                      f"{desat_result['final_err_deg']:.6f}"])
  print(f"[기록] 요약 결과 저장됨 → {summary_csv}")


if __name__ == "__main__":
  main()
