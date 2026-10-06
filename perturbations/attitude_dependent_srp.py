"""
자세 의존 태양복사압(SRP) - 캐논볼 모델을 넘어, 자세가 유효 단면적을 바꾼다

24번(solar_radiation_pressure.py)은 명시적으로 캐논볼 모델(구형 위성, 자세
무관 단면적, 등방 반사)만 다뤘다 - 그 docstring이 "실제 위성의 비구형
형상에 따른 자세 의존 단면적 변화는 다루지 않는다"고 범위를 그었다. 이
스크립트는 그 갭을 연다 - 21번(pid_attitude_control.py)의 쿼터니언 자세
상태를 24번의 SRP 가속도 계산에 결합해, 평판 태양전지판처럼 자세에 따라
태양을 향한 유효 단면적이 달라지는 경우를 다룬다.

핵심 개념 1: 평판의 유효 단면적은 태양 입사각의 코사인에 비례한다
  캐논볼 모델은 자세와 무관하게 항상 같은 단면적 A를 노출한다. 실제 평판
  태양전지판은 태양을 정면으로 볼 때(입사각 0도) 전체 면적 A_max를 노출하지만,
  각도 θ만큼 기울면 노출 면적이 A_max*cos(θ)로 줄고, 태양이 패널 뒤쪽에
  있으면(θ>90도) 전혀 햇빛을 받지 않는다(max(0,cosθ)로 음수 방지).

핵심 개념 2: 쿼터니언으로 몸체좌표계 벡터를 관성좌표계로 회전시켜야 한다
  21번에는 쿼터니언x쿼터니언 연산(quaternion_multiply, quaternion_conjugate)
  만 있고, 쿼터니언으로 3차원 벡터 하나를 회전시키는 함수는 이 프로젝트에
  없었다 - 이 스크립트에서 처음 추가하는 인프라다. 쿼터니언 샌드위치
  v_inertial = q*[0,v_body]*conj(q)(결과의 벡터부만 취함)로 구현한다. 패널
  법선은 몸체좌표계에 고정돼 있고, 위성이 회전하면 관성좌표계에서 본 법선
  방향도 함께 회전한다.

핵심 개념 3: 24번의 기존 함수는 수정하지 않는다 - "유효 A/m"만 새로 계산한다
  24번의 srp_acceleration/srp_acceleration_with_shadow는 area_to_mass_ratio_m2_kg를
  단순 스칼라 곱으로만 쓴다(accel = P_sr*Cr*(A/m)). 이 스크립트는 자세로부터
  "유효 A/m"(코사인 법칙 적용값)을 계산해 그 값을 24번의 기존 함수에 그대로
  넘기기만 한다 - 24번 자체는 1바이트도 수정하지 않는다.

핵심 개념 4: 텀블링 위성은 SRP가 진동하고, 태양지향 위성은 거의 일정하다
  토크 없는 자유회전(21번의 integrate_attitude_with_control에
  control_law=None)으로 계속 회전하는 위성은 패널이 태양을 향했다 등졌다
  반복하므로 유효 SRP 가속도가 시간에 따라 진동한다 - 캐논볼 모델의 "항상
  일정"이라는 가정이 깨지는 것을 직접 보여준다. 반대로 21번의 PD 제어로
  패널을 계속 태양 쪽으로 향하게 유지하면, 유효 SRP 가속도가 캐논볼 모델의
  최댓값 근처에서 거의 일정하게 유지된다.

핵심 개념 5: 자세 의존 SRP를 궤도 적분에 결합하면 캐논볼 모델과 경로가 갈라진다
  24번의 rk4_step_with_srp와 같은 시간의존 RK4 클론 구조를 쓰되, 매 스텝
  쿼터니언도 함께 적분하고(21번의 쿼터니언 운동학 + 16번 계열 토크 없는
  오일러 방정식) SRP 항에 그 순간의 유효 A/m을 쓴다. 여러 궤도 주기 적분하면
  텀블링 위성의 경로가 캐논볼 모델(항상 최대 A/m) 경로에서 서서히 벌어진다 -
  단, SRP 자체의 절대 크기(~1e-10km/s²)가 원래 작으므로 이 편차도 매우
  작다는 것을 함께 명시한다(과장하지 않음).

단순화: 패널은 몸체좌표계에 고정된 평판 1개(몸체 +z축 법선)만 가정한다
(여러 패널이나 자동 짐벌링 추적 패널은 다루지 않음). 패널 뒷면(cosθ<0)은
햇빛을 받지 않는다고만 가정하고 위성 본체에 의한 자가 차폐는 별도로
다루지 않는다. 24번의 원통형 그림자 근사(지구 그림자)는 그대로 재사용하고,
지구 그림자와 패널-뒷면 음영은 서로 다른 메커니즘이므로 "그림자 안 OR
패널이 태양을 등짐" 둘 다 0이 되는 조건으로 합성한다. 비구형 위성의
공기역학적 항력 단면적 변화는 다루지 않는다 - 이 확장은 SRP만 다룬다.

24번(solar_radiation_pressure.py)과의 관계: srp_acceleration,
srp_acceleration_with_shadow, is_in_shadow를 그대로 재사용한다(수정 없음).
21번(pid_attitude_control.py)과의 관계: 쿼터니언 운동학/PD 제어/RK4 적분
전부를 그대로 재사용하고, 쿼터니언으로 벡터를 회전시키는 함수만 새로
추가한다. 23번(third_body_perturbation.py)과의 관계: sun_position_eci_km을
24번을 거쳐 간접 재사용한다(24번이 이미 23번에서 가져온 것과 동일한 함수).
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

_SRP_PATH = os.path.join(_THIS_DIR, "solar_radiation_pressure.py")
_srp_spec = importlib.util.spec_from_file_location("srp_module", _SRP_PATH)
srp = importlib.util.module_from_spec(_srp_spec)
_srp_spec.loader.exec_module(srp)

_PID_PATH = os.path.join(_ROOT_DIR, "attitude", "pid_attitude_control.py")
_pid_spec = importlib.util.spec_from_file_location("pid_module", _PID_PATH)
pid = importlib.util.module_from_spec(_pid_spec)
_pid_spec.loader.exec_module(pid)

EARTH_MU_KM3_S2 = srp.EARTH_MU_KM3_S2
EARTH_RADIUS_KM = srp.EARTH_RADIUS_KM
sun_position_eci_km = srp.sun_position_eci_km
two_body_acceleration = srp.two_body_acceleration

DEFAULT_SEMI_MAJOR_AXIS_KM = 7000.0
DEFAULT_REFLECTIVITY_COEFFICIENT = srp.DEFAULT_REFLECTIVITY_COEFFICIENT
DEFAULT_MAX_AREA_TO_MASS_RATIO_M2_KG = srp.DEFAULT_AREA_TO_MASS_RATIO_M2_KG
DEFAULT_TUMBLE_RATE_RAD_S = 0.2
DEFAULT_PANEL_NORMAL_BODY = np.array([0.0, 0.0, 1.0])


def rotate_vector_by_quaternion(v_body, q):
  """쿼터니언 샌드위치 v_inertial = q*[0,v]*conj(q)로 몸체좌표계 벡터를
  관성좌표계로 회전시킨다. 21번에는 쿼터니언x쿼터니언 연산만 있고
  쿼터니언x벡터 연산이 없었다 - 이 프로젝트에 처음 등장하는 벡터 회전
  함수다."""
  q_v = np.concatenate([[0.0], np.array(v_body, dtype=float)])
  q_conj = pid.quaternion_conjugate(q)
  rotated = pid.quaternion_multiply(pid.quaternion_multiply(q, q_v), q_conj)
  return rotated[1:]


def effective_area_to_mass_ratio(q, panel_normal_body, sun_direction_inertial,
                                  max_area_to_mass_ratio_m2_kg):
  """패널 법선을 관성좌표계로 회전시켜 태양 방향과의 코사인각을 구하고,
  max(0,cosθ) 배율을 최대 A/m에 곱한다. 패널이 태양을 등지면(cosθ<0) 0."""
  normal_inertial = rotate_vector_by_quaternion(panel_normal_body, q)
  normal_inertial = normal_inertial / np.linalg.norm(normal_inertial)
  sun_direction_inertial = sun_direction_inertial / np.linalg.norm(sun_direction_inertial)
  cos_theta = np.dot(normal_inertial, sun_direction_inertial)
  return max_area_to_mass_ratio_m2_kg * max(0.0, cos_theta)


def sun_direction_from_satellite(position_km, sun_position_km):
  """위성 위치에서 본 태양 방향(단위벡터). 위성->태양 벡터를 정규화한다 -
  SRP 가속도 방향(태양->위성)과는 반대 부호임에 유의."""
  direction = np.array(sun_position_km, dtype=float) - np.array(position_km, dtype=float)
  return direction / np.linalg.norm(direction)


def attitude_dependent_srp_acceleration(position_km, sun_position_km, q, panel_normal_body,
                                         reflectivity_coefficient, max_area_to_mass_ratio_m2_kg):
  """유효 A/m을 계산해 24번의 srp_acceleration_with_shadow에 그대로 넘긴다 -
  24번 자체는 수정하지 않는다."""
  sun_direction_inertial = sun_direction_from_satellite(position_km, sun_position_km)
  eff_ratio = effective_area_to_mass_ratio(q, panel_normal_body, sun_direction_inertial,
                                            max_area_to_mass_ratio_m2_kg)
  return srp.srp_acceleration_with_shadow(position_km, sun_position_km,
                                           reflectivity_coefficient, eff_ratio), eff_ratio


def attitude_and_orbit_state_derivative(orbit_state, attitude_state, t, inertia_diag,
                                         panel_normal_body, reflectivity_coefficient,
                                         max_area_to_mass_ratio_m2_kg, torque, mu):
  """궤도 상태(위치/속도)와 자세 상태([q, omega])를 함께 미분한다. 토크는
  이 함수 밖에서 한 번만 평가된 값을 받는다(21번 attitude_state_derivative와
  같은 책임 분리)."""
  position_km, velocity_km_s = orbit_state
  sun_pos = sun_position_eci_km(t)
  srp_accel, _ = attitude_dependent_srp_acceleration(
      position_km, sun_pos, attitude_state[:4], panel_normal_body,
      reflectivity_coefficient, max_area_to_mass_ratio_m2_kg)
  accel = two_body_acceleration(position_km, mu) + srp_accel
  d_orbit = (velocity_km_s, accel)
  d_attitude = pid.attitude_state_derivative(attitude_state, inertia_diag, torque)
  return d_orbit, d_attitude


def rk4_step_with_attitude_and_srp(position_km, velocity_km_s, attitude_state, dt_sec, time_sec,
                                    inertia_diag, panel_normal_body, reflectivity_coefficient,
                                    max_area_to_mass_ratio_m2_kg, mu=EARTH_MU_KM3_S2):
  """24번 rk4_step_with_srp와 같은 시간의존 RK4 클론 구조지만, 궤도 상태와
  자세 상태([q,omega] 7차원)를 함께 적분한다. 토크 없는 자유회전(외부 토크
  0)만 다룬다 - 텀블링 자체는 궤도와 독립적으로 진행된다. 쿼터니언 재정규화는
  21번과 같은 원칙으로 전체 조합 이후 한 번만 수행한다."""
  torque = np.zeros(3)

  def derivative(pos, vel, att, t):
    d_orbit, d_att = attitude_and_orbit_state_derivative(
        (pos, vel), att, t, inertia_diag, panel_normal_body,
        reflectivity_coefficient, max_area_to_mass_ratio_m2_kg, torque, mu)
    return d_orbit[0], d_orbit[1], d_att

  k1_r, k1_v, k1_a = derivative(position_km, velocity_km_s, attitude_state, time_sec)
  k2_r, k2_v, k2_a = derivative(position_km + dt_sec / 2 * k1_r, velocity_km_s + dt_sec / 2 * k1_v,
                                 attitude_state + dt_sec / 2 * k1_a, time_sec + dt_sec / 2)
  k3_r, k3_v, k3_a = derivative(position_km + dt_sec / 2 * k2_r, velocity_km_s + dt_sec / 2 * k2_v,
                                 attitude_state + dt_sec / 2 * k2_a, time_sec + dt_sec / 2)
  k4_r, k4_v, k4_a = derivative(position_km + dt_sec * k3_r, velocity_km_s + dt_sec * k3_v,
                                 attitude_state + dt_sec * k3_a, time_sec + dt_sec)

  new_position = position_km + dt_sec / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
  new_velocity = velocity_km_s + dt_sec / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
  new_attitude = attitude_state + dt_sec / 6 * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
  new_attitude[:4] = pid.normalize_quaternion(new_attitude[:4])
  return new_position, new_velocity, new_attitude


def demo_effective_area_matches_cosine_law_at_known_angles(
    max_area_to_mass_ratio_m2_kg=DEFAULT_MAX_AREA_TO_MASS_RATIO_M2_KG):
  """패널 법선과 태양 방향 사이 각을 0/60/90/180도로 고정해 유효 A/m이
  코사인 법칙과 일치하는지 확인한다 - 공식 자체의 정합성 검증(거의 항등식)."""
  print("=" * 70)
  print("[1] 코사인 법칙: 유효 단면적이 입사각에 따라 올바르게 변하는가")
  print("=" * 70)
  panel_normal_body = DEFAULT_PANEL_NORMAL_BODY
  sun_direction_inertial = np.array([0.0, 0.0, 1.0])

  angles_deg = [0.0, 60.0, 90.0, 180.0]
  rows = []
  print(f"  {'입사각(도)':>12}{'유효 A/m':>16}{'예상 비율':>12}")
  for angle_deg in angles_deg:
    axis = np.array([1.0, 0.0, 0.0])
    q = pid.axis_angle_to_quaternion(axis, angle_deg)
    eff_ratio = effective_area_to_mass_ratio(q, panel_normal_body, sun_direction_inertial,
                                              max_area_to_mass_ratio_m2_kg)
    expected_ratio = max(0.0, np.cos(np.radians(angle_deg)))
    rows.append({"angle_deg": angle_deg, "effective_area_to_mass_ratio_m2_kg": eff_ratio,
                 "expected_ratio": expected_ratio})
    print(f"  {angle_deg:>12.1f}{eff_ratio:>16.6f}{expected_ratio:>12.4f}")

  assert abs(rows[0]["effective_area_to_mass_ratio_m2_kg"] - max_area_to_mass_ratio_m2_kg) < 1e-9
  assert abs(rows[1]["effective_area_to_mass_ratio_m2_kg"] - max_area_to_mass_ratio_m2_kg * 0.5) < 1e-9
  assert abs(rows[2]["effective_area_to_mass_ratio_m2_kg"]) < 1e-9
  assert rows[3]["effective_area_to_mass_ratio_m2_kg"] == 0.0

  print("\n(0도=정면(최대), 60도=절반, 90도=0, 180도=뒷면(0으로 클램프) -")
  print(" 코사인 법칙이 경계 케이스 전부에서 올바르게 동작한다.)")
  return {"rows": rows}


def demo_tumbling_satellite_srp_oscillates_while_cannonball_stays_constant(
    inertia_diag=(1.0, 2.0, 3.0), tumble_rate_rad_s=DEFAULT_TUMBLE_RATE_RAD_S,
    reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT,
    max_area_to_mass_ratio_m2_kg=DEFAULT_MAX_AREA_TO_MASS_RATIO_M2_KG):
  """토크 없는 자유회전으로 텀블링하는 위성의 유효 SRP 가속도가 캐논볼
  모델(항상 상수)과 달리 시간에 따라 진동하는지 확인한다."""
  print("\n" + "=" * 70)
  print("[2] 텀블링 위성: 캐논볼은 일정, 자세 의존 모델은 진동하는가")
  print("=" * 70)
  position_km = np.array([DEFAULT_SEMI_MAJOR_AXIS_KM, 0.0, 0.0])
  sun_position_km = sun_position_eci_km(0.0)
  panel_normal_body = DEFAULT_PANEL_NORMAL_BODY

  initial_q = pid.IDENTITY_QUATERNION.copy()
  initial_omega = np.array([tumble_rate_rad_s, tumble_rate_rad_s * 0.5, 0.0])
  dt_sec = 0.5
  rotation_period_sec = 2 * np.pi / tumble_rate_rad_s
  total_time_sec = rotation_period_sec * 3

  history = pid.integrate_attitude_with_control(initial_q, initial_omega, inertia_diag,
                                                 total_time_sec, dt_sec, control_law=None)

  cannonball_accel = np.linalg.norm(srp.srp_acceleration(
      position_km, sun_position_km, reflectivity_coefficient, max_area_to_mass_ratio_m2_kg))

  effective_mags = []
  rows = []
  for t, (q, _omega) in history:
    srp_accel, _eff_ratio = attitude_dependent_srp_acceleration(
        position_km, sun_position_km, q, panel_normal_body,
        reflectivity_coefficient, max_area_to_mass_ratio_m2_kg)
    mag = np.linalg.norm(srp_accel)
    effective_mags.append(mag)
    rows.append({"t_sec": t, "effective_srp_accel_km_s2": mag,
                 "cannonball_srp_accel_km_s2": cannonball_accel})

  effective_mags = np.array(effective_mags)
  std = float(np.std(effective_mags))
  max_val = float(np.max(effective_mags))

  print(f"텀블링 각속도≈{tumble_rate_rad_s}rad/s(회전주기≈{rotation_period_sec:.1f}초), "
        f"{total_time_sec:.1f}초 적분")
  print(f"캐논볼 SRP 가속도(상수)={cannonball_accel:.4e}km/s^2")
  print(f"자세 의존 SRP 가속도: 평균={np.mean(effective_mags):.4e}, "
        f"표준편차={std:.4e}, 최댓값={max_val:.4e}km/s^2")

  assert std > cannonball_accel * 0.05, "텀블링하면 유효 SRP가 캐논볼 대비 뚜렷이 진동해야 함"
  assert abs(max_val - cannonball_accel) / cannonball_accel < 0.05, \
      "패널이 정면을 향하는 순간에는 캐논볼 값과 거의 일치해야 함"

  print("\n(캐논볼 모델은 자세와 무관하게 항상 일정하지만, 패널이 달린 실제")
  print(" 위성은 텀블링하며 패널이 태양을 향했다 등졌다 반복해 SRP가 진동한다 -")
  print(" 최댓값은 패널이 정면을 향하는 순간 캐논볼 값에 거의 도달한다.)")
  return {"rows": rows, "std": std, "max_val": max_val, "cannonball_accel": cannonball_accel}


def demo_sun_pointing_control_keeps_srp_near_cannonball_maximum(
    inertia_diag=(1.0, 2.0, 3.0), settling_time_sec=5.0,
    reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT,
    max_area_to_mass_ratio_m2_kg=DEFAULT_MAX_AREA_TO_MASS_RATIO_M2_KG):
  """21번의 PD 제어로 패널이 태양을 계속 정면으로 향하도록 자세를 유지하면,
  유효 SRP 가속도가 캐논볼 최댓값 근처에서 거의 일정하게 유지되는지 확인한다."""
  print("\n" + "=" * 70)
  print("[3] 태양지향 제어: PD 제어가 유효 SRP를 캐논볼 최댓값 근처로 유지하는가")
  print("=" * 70)
  position_km = np.array([DEFAULT_SEMI_MAJOR_AXIS_KM, 0.0, 0.0])
  sun_position_km = sun_position_eci_km(0.0)
  panel_normal_body = DEFAULT_PANEL_NORMAL_BODY
  sun_direction_inertial = sun_direction_from_satellite(position_km, sun_position_km)

  # 패널 법선(몸체 +z)이 태양 방향과 일치하는 목표 자세: 두 벡터 사이 회전을 축-각으로 구함
  rotation_axis = np.cross(panel_normal_body, sun_direction_inertial)
  axis_norm = np.linalg.norm(rotation_axis)
  if axis_norm < 1e-9:
    q_target = pid.IDENTITY_QUATERNION.copy()
  else:
    rotation_axis = rotation_axis / axis_norm
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(panel_normal_body, sun_direction_inertial), -1.0, 1.0)))
    q_target = pid.axis_angle_to_quaternion(rotation_axis, angle_deg)

  initial_q = pid.axis_angle_to_quaternion([1, 0, 0], 60.0)
  initial_omega = np.zeros(3)
  kp, kd = pid.compute_pd_gains(inertia_diag, settling_time_sec)
  dt_sec = 0.05
  total_time_sec = settling_time_sec * 3

  control_law = lambda q_err, omega: pid.pd_control_torque(q_err, omega, np.zeros(3), kp, kd)  # noqa: E731
  history = pid.integrate_attitude_with_control(initial_q, initial_omega, inertia_diag,
                                                 total_time_sec, dt_sec,
                                                 control_law=control_law, q_target=q_target)

  cannonball_accel = np.linalg.norm(srp.srp_acceleration(
      position_km, sun_position_km, reflectivity_coefficient, max_area_to_mass_ratio_m2_kg))

  rows = []
  for t, (q, _omega) in history:
    srp_accel, _eff_ratio = attitude_dependent_srp_acceleration(
        position_km, sun_position_km, q, panel_normal_body,
        reflectivity_coefficient, max_area_to_mass_ratio_m2_kg)
    pointing_error_deg = pid.quaternion_error_angle_deg(pid.attitude_error_quaternion(q, q_target))
    rows.append({"t_sec": t, "effective_srp_accel_km_s2": np.linalg.norm(srp_accel),
                 "cannonball_srp_accel_km_s2": cannonball_accel,
                 "pointing_error_deg": pointing_error_deg})

  settled_rows = [r for r in rows if r["t_sec"] >= settling_time_sec]
  min_ratio = min(r["effective_srp_accel_km_s2"] for r in settled_rows) / cannonball_accel

  print(f"초기 오차=60도, 정착시간={settling_time_sec}초, 캐논볼 SRP={cannonball_accel:.4e}km/s^2")
  print(f"정착 후({settling_time_sec}초 이후) 최소 유효/캐논볼 비율={min_ratio * 100:.2f}%")

  assert min_ratio > 0.95, "정착 후에는 태양지향 제어가 유효 SRP를 캐논볼 최댓값의 95% 이상으로 유지해야 함"

  print("\n(텀블링 데모와 달리, 태양을 계속 바라보도록 제어하면 유효 SRP가")
  print(" 캐논볼 모델의 최댓값 근처에서 거의 일정하게 유지된다 - 태양지향")
  print(" 자세제어가 전력 생산뿐 아니라 SRP 예측 가능성에도 영향을 준다.)")
  return {"rows": rows, "min_ratio": min_ratio, "cannonball_accel": cannonball_accel}


def demo_attitude_dependent_srp_diverges_orbit_path_from_cannonball_over_time(
    inertia_diag=(1.0, 2.0, 3.0), tumble_rate_rad_s=DEFAULT_TUMBLE_RATE_RAD_S,
    semi_major_axis_km=DEFAULT_SEMI_MAJOR_AXIS_KM,
    reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT,
    max_area_to_mass_ratio_m2_kg=DEFAULT_MAX_AREA_TO_MASS_RATIO_M2_KG,
    num_orbits=5, dt_sec=1.0):
  """텀블링 위성의 궤도(자세 의존 SRP)와 캐논볼 모델 궤도(같은 초기조건)를
  나란히 여러 궤도 주기 적분해, 두 경로가 서서히 벌어지지만 그 편차가 반장축
  대비 작게 유지되는지 확인한다."""
  print("\n" + "=" * 70)
  print("[4] 궤도 경로 편차: 자세 의존 SRP가 캐논볼 모델과 다른 경로를 만드는가")
  print("=" * 70)
  speed_km_s = np.sqrt(EARTH_MU_KM3_S2 / semi_major_axis_km)
  orbital_period_sec = 2 * np.pi * np.sqrt(semi_major_axis_km ** 3 / EARTH_MU_KM3_S2)
  total_time_sec = orbital_period_sec * num_orbits
  panel_normal_body = DEFAULT_PANEL_NORMAL_BODY

  pos_tumble = np.array([semi_major_axis_km, 0.0, 0.0])
  vel_tumble = np.array([0.0, speed_km_s, 0.0])
  attitude_state = np.concatenate([pid.IDENTITY_QUATERNION, [tumble_rate_rad_s, tumble_rate_rad_s * 0.5, 0.0]])

  pos_cannonball = pos_tumble.copy()
  vel_cannonball = vel_tumble.copy()

  t = 0.0
  rows = []
  num_steps = round(total_time_sec / dt_sec)
  sample_every = max(1, num_steps // 500)
  for step in range(num_steps):
    pos_tumble, vel_tumble, attitude_state = rk4_step_with_attitude_and_srp(
        pos_tumble, vel_tumble, attitude_state, dt_sec, t, inertia_diag, panel_normal_body,
        reflectivity_coefficient, max_area_to_mass_ratio_m2_kg)
    pos_cannonball, vel_cannonball = srp.rk4_step_with_srp(
        pos_cannonball, vel_cannonball, dt_sec, t,
        reflectivity_coefficient, max_area_to_mass_ratio_m2_kg)
    t += dt_sec
    if step % sample_every == 0:
      diff_km = np.linalg.norm(pos_tumble - pos_cannonball)
      rows.append({"t_sec": t, "position_diff_km": diff_km})

  final_diff_km = np.linalg.norm(pos_tumble - pos_cannonball)
  rows.append({"t_sec": t, "position_diff_km": final_diff_km})

  print(f"{num_orbits}개 궤도 주기(약 {total_time_sec:.0f}초) 적분, dt={dt_sec}초")
  print(f"텀블링(자세 의존 SRP) vs 캐논볼 모델 최종 위치 차이={final_diff_km:.6f}km "
        f"(반장축 {semi_major_axis_km:.0f}km 대비 {final_diff_km / semi_major_axis_km * 100:.6f}%)")

  assert final_diff_km > 0.0, "두 모델의 경로가 완전히 같을 수는 없음"
  assert final_diff_km < semi_major_axis_km * 0.01, \
      "SRP 자체가 약한 섭동이므로 편차는 반장축 대비 작게 유지돼야 함"

  print("\n(두 경로는 분명히 갈라지지만, SRP 자체의 절대 크기가 매우 작으므로")
  print(" 그 편차도 반장축에 비하면 미미한 수준에 머문다 - 과장하지 않는다.)")
  return {"rows": rows, "final_diff_km": final_diff_km, "num_orbits": num_orbits}


def write_csv(filepath, rows, fieldnames):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  with open(filepath, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def parse_args():
  parser = argparse.ArgumentParser(description="자세 의존 태양복사압(SRP) 데모")
  parser.add_argument("--semi-major-axis-km", type=float, default=DEFAULT_SEMI_MAJOR_AXIS_KM,
                       help="궤도 반장축(km), 기본값: 7000km(LEO)")
  parser.add_argument("--reflectivity-coefficient", type=float, default=DEFAULT_REFLECTIVITY_COEFFICIENT,
                       help="복사압 반사계수 Cr, 기본값: 1.5(gray body 가정)")
  parser.add_argument("--max-area-to-mass-ratio-m2-kg", type=float,
                       default=DEFAULT_MAX_AREA_TO_MASS_RATIO_M2_KG,
                       help="패널 정면 기준 최대 단면적대질량비(m^2/kg), 기본값: 0.02")
  parser.add_argument("--tumble-rate-rad-s", type=float, default=DEFAULT_TUMBLE_RATE_RAD_S,
                       help="텀블링 각속도 크기(rad/s), 기본값: 0.2")
  parser.add_argument("--inertia-i1", type=float, default=1.0, help="주관성모멘트 I1, 기본값: 1.0")
  parser.add_argument("--inertia-i2", type=float, default=2.0, help="주관성모멘트 I2, 기본값: 2.0")
  parser.add_argument("--inertia-i3", type=float, default=3.0, help="주관성모멘트 I3, 기본값: 3.0")
  return parser.parse_args()


def main():
  args = parse_args()
  inertia_diag = (args.inertia_i1, args.inertia_i2, args.inertia_i3)

  cosine_result = demo_effective_area_matches_cosine_law_at_known_angles(
      args.max_area_to_mass_ratio_m2_kg)
  tumble_result = demo_tumbling_satellite_srp_oscillates_while_cannonball_stays_constant(
      inertia_diag, args.tumble_rate_rad_s, args.reflectivity_coefficient,
      args.max_area_to_mass_ratio_m2_kg)
  sun_pointing_result = demo_sun_pointing_control_keeps_srp_near_cannonball_maximum(
      inertia_diag, 5.0, args.reflectivity_coefficient, args.max_area_to_mass_ratio_m2_kg)
  divergence_result = demo_attitude_dependent_srp_diverges_orbit_path_from_cannonball_over_time(
      inertia_diag, args.tumble_rate_rad_s, args.semi_major_axis_km,
      args.reflectivity_coefficient, args.max_area_to_mass_ratio_m2_kg)

  print("\n" + "=" * 70)
  print(f"[사용자 지정] 반장축={args.semi_major_axis_km}km, 텀블링 각속도={args.tumble_rate_rad_s}rad/s, "
        f"최대 A/m={args.max_area_to_mass_ratio_m2_kg}m^2/kg")
  print("=" * 70)
  print(f"텀블링 SRP 표준편차={tumble_result['std']:.4e}km/s^2, "
        f"태양지향 최소비율={sun_pointing_result['min_ratio'] * 100:.2f}%, "
        f"궤도 편차={divergence_result['final_diff_km']:.6f}km")

  results_dir = os.path.join(_ROOT_DIR, "results")

  write_csv(os.path.join(results_dir, "attitude_dependent_srp_cosine_law_validation.csv"),
            cosine_result["rows"], ["angle_deg", "effective_area_to_mass_ratio_m2_kg", "expected_ratio"])
  write_csv(os.path.join(results_dir, "attitude_dependent_srp_tumbling_vs_cannonball.csv"),
            tumble_result["rows"], ["t_sec", "effective_srp_accel_km_s2", "cannonball_srp_accel_km_s2"])
  write_csv(os.path.join(results_dir, "attitude_dependent_srp_sun_pointing_comparison.csv"),
            sun_pointing_result["rows"],
            ["t_sec", "effective_srp_accel_km_s2", "cannonball_srp_accel_km_s2", "pointing_error_deg"])
  write_csv(os.path.join(results_dir, "attitude_dependent_srp_orbit_divergence.csv"),
            divergence_result["rows"], ["t_sec", "position_diff_km"])

  summary_row = {
      "tumbling_srp_std_km_s2": tumble_result["std"],
      "cannonball_srp_km_s2": tumble_result["cannonball_accel"],
      "sun_pointing_min_ratio_pct": sun_pointing_result["min_ratio"] * 100,
      "final_position_diff_km": divergence_result["final_diff_km"],
  }
  write_csv(os.path.join(results_dir, "attitude_dependent_srp_summary.csv"),
            [summary_row], list(summary_row.keys()))

  print(f"\n[기록] 코사인 법칙 검증 저장됨 → {os.path.join(results_dir, 'attitude_dependent_srp_cosine_law_validation.csv')}")
  print(f"[기록] 텀블링 vs 캐논볼 저장됨 → {os.path.join(results_dir, 'attitude_dependent_srp_tumbling_vs_cannonball.csv')}")
  print(f"[기록] 태양지향 비교 저장됨 → {os.path.join(results_dir, 'attitude_dependent_srp_sun_pointing_comparison.csv')}")
  print(f"[기록] 궤도 편차 저장됨 → {os.path.join(results_dir, 'attitude_dependent_srp_orbit_divergence.csv')}")
  print(f"[기록] 요약 결과 저장됨 → {os.path.join(results_dir, 'attitude_dependent_srp_summary.csv')}")


if __name__ == "__main__":
  main()
