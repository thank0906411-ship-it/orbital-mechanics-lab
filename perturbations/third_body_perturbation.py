"""
3체(태양/달) 섭동 - 17번이 "모델링하지 않는 요인"이라 부른 것을 실제로 계산한다

17번(station_keeping.py)의 docstring은 "세차율 계산 자체의 불확실성(더 높은
차수의 섭동, 대기항력, 태양/달의 중력 등 이 스크립트가 모델링하지 않는
요인들)"이라며 태양/달의 중력을 명시적으로 범위 밖으로 남겨뒀다. 이 스크립트는
그 요인을 실제로 계산 가능한 값으로 만든다 - 태양과 달이 위성에 가하는 중력이
지구 2체 문제에 더하는 섭동 가속도를 직접 구하고, 07번(J2 섭동)과 같은 단위
(가속도 크기, km/s²)로 비교해 고도에 따라 어느 섭동이 지배적인지 보여준다.

핵심 개념 1: 두 점질량 간 중력에 세 번째 천체가 더하는 상대 섭동
  지구 중심 좌표계에서 위성이 받는 태양/달의 중력은 단순히
  mu_body*r_sat_to_body/|...|^3이 아니다. 지구 자신도 같은 천체에 끌리므로,
  "위성이 느끼는 섭동"은 그 공통 가속도를 뺀 상대(차분)량이어야 한다:
      a_3rd = mu_body * [(r_body-r_sat)/|r_body-r_sat|^3 - r_body/|r_body|^3]
  두 번째 항을 빼지 않으면 지구 자체가 받는 가속도까지 중복 포함돼 크기와
  경향이 모두 틀어진다. 위성이 지구 중심(r_sat=0)에 있으면 두 항이 정확히
  상쇄돼 가속도가 0이 된다 - 이 항등식으로 공식 자체를 검증할 수 있다.

핵심 개념 2: 질량은 압도적으로 크지만 달의 섭동이 태양보다 크다
  태양의 GM(1.327e11 km^3/s^2)은 달의 GM(4902.8 km^3/s^2)보다 2700만 배
  크지만, 거리의 세제곱에 반비례하는 상대 섭동 공식 때문에 지구에 훨씬
  가까운 달의 섭동이 실제로는 태양의 약 2.2배 크다 - 교과서적으로 잘 알려진
  결과를 직접 수치로 재현한다.

핵심 개념 3: 고도가 높아질수록 J2 대비 3체 섭동의 비중이 커진다
  J2 가속도는 1/r^4로 매우 빠르게 감소하지만, 3체 섭동은 고도에 따라
  완만하게만 커진다. 그 결과 LEO에서는 J2가 압도적이지만, GEO 부근에서는
  3체 섭동이 J2와 같은 자릿수이거나 넘어선다 - 실제로 GEO 위성의 station-
  keeping에서 태양/달 섭동이 J2보다 더 중요한 보정 대상인 이유를 수치로
  보여준다.

핵심 개념 4: 시간에 따라 변하는 섭동
  궤도 적분에 더하면 태양/달이 움직이며 섭동의 크기/방향이 계속 바뀐다.
  03번/15번/18번과 같은 "RK4 루프 복제" 패턴을 쓰되, 이번엔 섭동 자체가
  시간에 의존하므로 derivative 클로저가 (pos, vel, t)를 받고 RK4 k2/k3/k4
  단계마다 t+dt/2, t+dt/2, t+dt로 시간을 진행시켜 호출해야 한다 - 기존
  저추력/항력 클론과 다른 새로운 요소다.

단순화: 태양과 달 모두 이심률 0인 원궤도, 지구 적도면과 같은 평면으로
근사한다(황도 기울기 23.4도, 달의 궤도경사 약 5.1도 모두 무시). 이는 방향
정확도를 희생하지만 이 스크립트가 보여주려는 핵심(크기와 시간에 따른 변화)은
그대로 보존한다 - 12번(patched_conic_interplanetary.py)이 지구 공전을 1AU
원궤도로 근사한 것과 같은 수준의 "충분히 좋은" 근사다. 천체력(ephemeris)
기반 정밀 위치는 범위 밖이다. 달력 기준시각(epoch) 정렬도 시도하지 않는다.
J2 가속도 크기는 07번의 세차율 함수를 재사용하지 않는다 - 그 함수들은 RAAN/
근점편각의 각속도를 주지, 3체 섭동과 비교 가능한 가속도 벡터 크기를 주지
않으므로, 적도면 단순화 공식(a_J2 = 1.5*J2*mu*R_E^2/r^4)을 이 스크립트에서
새로 구현한다.

17번(station_keeping.py)과의 관계: 이 스크립트가 계산하는 가속도는 J2 대비
LEO에서는 무시할 수 있지만 GEO에서는 무시하기 어려운 수준이 된다 - 17번이
"모델링하지 않는 요인"으로만 언급했던 잔여 오차의 구체적 크기 중 하나가
이것이다. 다만 이 스크립트는 가속도(km/s^2)를, 17번은 RAAN 세차율(rad/s)을
다루므로 두 수치를 직접 비교하거나 변환하지는 않는다 - 그러려면 가우스
섭동방정식 기반 평균화가 추가로 필요한 별도 범위다.
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

_KEPLER_PATH = os.path.join(_ROOT_DIR, "propagation", "kepler_orbit_propagation.py")
_kepler_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_kepler_spec)
_kepler_spec.loader.exec_module(kepler)

_RK4_PATH = os.path.join(_ROOT_DIR, "propagation", "two_body_numerical_integration.py")
_rk4_spec = importlib.util.spec_from_file_location("rk4_module", _RK4_PATH)
rk4_module = importlib.util.module_from_spec(_rk4_spec)
_rk4_spec.loader.exec_module(rk4_module)
two_body_acceleration = rk4_module.two_body_acceleration
rk4_step = rk4_module.rk4_step

_ORBIT_MATH_PATH = os.path.join(_ROOT_DIR, "orbit_math.py")
_orbit_math_spec = importlib.util.spec_from_file_location("orbit_math", _ORBIT_MATH_PATH)
orbit_math = importlib.util.module_from_spec(_orbit_math_spec)
_orbit_math_spec.loader.exec_module(orbit_math)
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2

MOON_MU_KM3_S2 = 4902.800  # 달 GM 문헌값 (km^3/s^2)
SUN_MU_KM3_S2 = 1.32712440018e11  # 태양 GM 문헌값 (km^3/s^2), 12번과 동일값을 로컬 재정의
AU_KM = 1.495978707e8
EARTH_ORBIT_RADIUS_KM = 1.0 * AU_KM
MOON_ORBIT_RADIUS_KM = 384400.0  # 달의 평균 궤도반지름. 19번의 EARTH_MOON_DISTANCE_KM과
                                   # 같은 값이지만 그 상수는 CR3BP 무차원화 전용이라 재사용하지 않고 새로 정의
SIDEREAL_MONTH_DAYS = 27.321661
SUN_ANGULAR_RATE_RAD_S = 2 * np.pi / (365.25 * 86400)
MOON_ANGULAR_RATE_RAD_S = 2 * np.pi / (SIDEREAL_MONTH_DAYS * 86400)
J2 = 1.08263e-3

ALTITUDE_SWEEP_KM = [300.0, 550.0, 800.0, 1500.0, 3000.0, 6000.0, 10000.0, 20200.0, 35786.0]


def third_body_acceleration(position_km, body_position_km, mu_body):
  """상대(차분) 3체 섭동 가속도:
  a_3rd = mu_body*[(r_body-r_sat)/|r_body-r_sat|^3 - r_body/|r_body|^3].
  위성이 원점(지구 중심)에 있으면 두 항이 정확히 상쇄돼 0이 된다."""
  body_position_km = np.array(body_position_km, dtype=float)
  position_km = np.array(position_km, dtype=float)
  r_sat_to_body = body_position_km - position_km
  d_sat_body = np.linalg.norm(r_sat_to_body)
  d_body = np.linalg.norm(body_position_km)
  return mu_body * (r_sat_to_body / d_sat_body ** 3 - body_position_km / d_body ** 3)


def sun_position_eci_km(time_sec, orbit_radius_km=EARTH_ORBIT_RADIUS_KM,
                         angular_rate_rad_s=SUN_ANGULAR_RATE_RAD_S):
  """태양을 지구 중심 원궤도(이심률 0, 황도 기울기 무시, 적도면과 동일 가정)로
  근사. t=0에서 +x축 방향."""
  angle = angular_rate_rad_s * time_sec
  return orbit_radius_km * np.array([np.cos(angle), np.sin(angle), 0.0])


def moon_position_eci_km(time_sec, orbit_radius_km=MOON_ORBIT_RADIUS_KM,
                          angular_rate_rad_s=MOON_ANGULAR_RATE_RAD_S):
  """달을 지구 중심 원궤도(이심률 0, 궤도면을 적도면과 동일 가정)로 근사.
  각속도는 항성월(sidereal month) 기준. t=0에서 +x축 방향."""
  angle = angular_rate_rad_s * time_sec
  return orbit_radius_km * np.array([np.cos(angle), np.sin(angle), 0.0])


def lunisolar_acceleration(position_km, time_sec):
  """태양+달 third_body_acceleration의 합. 반환: (total_accel, sun_accel,
  moon_accel) - 데모가 태양/달 기여를 분리해 비교할 수 있도록 성분별로도
  반환한다."""
  sun_accel = third_body_acceleration(position_km, sun_position_eci_km(time_sec), SUN_MU_KM3_S2)
  moon_accel = third_body_acceleration(position_km, moon_position_eci_km(time_sec), MOON_MU_KM3_S2)
  return sun_accel + moon_accel, sun_accel, moon_accel


def j2_acceleration_magnitude(r_km, mu=EARTH_MU_KM3_S2, j2=J2, re=EARTH_RADIUS_KM):
  """적도면 단순화 J2 가속도 크기: a_J2 = 1.5*J2*mu*R_E^2/r^4. 07번의 세차율
  함수와 달리 가속도 "크기"를 직접 주므로 3체 섭동과 같은 단위(km/s^2)로
  비교할 수 있다."""
  return 1.5 * j2 * mu * re ** 2 / r_km ** 4


def altitude_regime_comparison(altitude_km, time_sec=0.0):
  """주어진 고도의 적도 원궤도 위성 위치 하나에서 J2 가속도 크기와 태양/달
  섭동 가속도 크기를 비교한다."""
  r_km = EARTH_RADIUS_KM + altitude_km
  position_km = np.array([r_km, 0.0, 0.0])
  total_accel, sun_accel, moon_accel = lunisolar_acceleration(position_km, time_sec)
  j2_accel = j2_acceleration_magnitude(r_km)
  lunisolar_total = np.linalg.norm(total_accel)
  return {
      "altitude_km": altitude_km,
      "j2_accel_km_s2": j2_accel,
      "solar_accel_km_s2": np.linalg.norm(sun_accel),
      "lunar_accel_km_s2": np.linalg.norm(moon_accel),
      "lunisolar_total_accel_km_s2": lunisolar_total,
      "ratio_lunisolar_to_j2": lunisolar_total / j2_accel,
  }


def rk4_step_with_third_body(position_km, velocity_km_s, dt_sec, time_sec, mu=EARTH_MU_KM3_S2):
  """15번/18번과 동일한 'RK4 루프 복제' 패턴이되, 섭동이 시간에 의존하므로
  derivative 클로저가 (pos, vel, t)를 받고 k2/k3/k4 단계마다 t+dt/2, t+dt/2,
  t+dt로 시간을 진행시켜 호출한다. 03번 rk4_step의 derivative는 중심력만
  계산하도록 하드코딩되어 있어 그대로 재사용할 수 없으므로 이 함수가 그
  구조를 복제해 확장한다."""
  def derivative(pos, vel, t):
    total_accel, _sun, _moon = lunisolar_acceleration(pos, t)
    accel = two_body_acceleration(pos, mu) + total_accel
    return vel, accel

  k1_r, k1_v = derivative(position_km, velocity_km_s, time_sec)
  k2_r, k2_v = derivative(position_km + dt_sec / 2 * k1_r, velocity_km_s + dt_sec / 2 * k1_v, time_sec + dt_sec / 2)
  k3_r, k3_v = derivative(position_km + dt_sec / 2 * k2_r, velocity_km_s + dt_sec / 2 * k2_v, time_sec + dt_sec / 2)
  k4_r, k4_v = derivative(position_km + dt_sec * k3_r, velocity_km_s + dt_sec * k3_v, time_sec + dt_sec)

  new_position = position_km + dt_sec / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
  new_velocity = velocity_km_s + dt_sec / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
  return new_position, new_velocity


def demo_lunar_perturbation_exceeds_solar_despite_smaller_mass(altitude_km=550.0):
  """LEO 한 지점에서 태양/달 섭동 가속도 크기를 각각 계산하고 비교한다 - 질량은
  태양이 압도적으로 크지만(2700만 배), 거리의 세제곱에 반비례하는 상대 섭동
  공식 때문에 달의 섭동이 실제로는 더 크다는 교과서적 결과를 재현한다."""
  print("=" * 70)
  print("[1] 태양 vs 달: 질량은 압도적으로 작은 달의 섭동이 더 크다")
  print("=" * 70)
  r_km = EARTH_RADIUS_KM + altitude_km
  position_km = np.array([r_km, 0.0, 0.0])
  _total, sun_accel, moon_accel = lunisolar_acceleration(position_km, time_sec=0.0)
  sun_mag = np.linalg.norm(sun_accel)
  moon_mag = np.linalg.norm(moon_accel)
  ratio = moon_mag / sun_mag

  print(f"고도={altitude_km}km (지구 반지름 포함 r={r_km:.1f}km)")
  print(f"태양 GM={SUN_MU_KM3_S2:.3e}km^3/s^2, 태양 섭동 가속도={sun_mag:.3e}km/s^2")
  print(f"달 GM={MOON_MU_KM3_S2:.3e}km^3/s^2, 달 섭동 가속도={moon_mag:.3e}km/s^2")
  print(f"비율(달/태양)={ratio:.3f}")

  assert moon_mag > sun_mag, "달의 섭동 가속도가 태양의 섭동 가속도보다 커야 함(거리의 세제곱 효과가 질량비를 압도)"
  assert 1.5 < ratio < 3.0, "달/태양 섭동 비율은 교과서값(약 2.2) 근처여야 함"
  print("\n(태양이 달보다 2700만 배 무겁지만, 상대 섭동 공식은 거리의 세제곱에")
  print(" 반비례한다 - 지구에 훨씬 가까운 달의 섭동이 실제로는 더 크다.)")
  return {"sun_accel_km_s2": sun_mag, "moon_accel_km_s2": moon_mag, "ratio": ratio}


def demo_j2_vs_third_body_crossover_across_altitude():
  """LEO부터 GEO까지 고도를 스윕해 J2 가속도와 3체 섭동 가속도를 비교한다. J2는
  1/r^4로 빠르게 감소하지만 3체 섭동은 완만하게만 변해, 고도가 높아질수록
  3체 섭동의 상대적 비중이 커진다는 것을 확인한다."""
  print("\n" + "=" * 70)
  print("[2] 고도별 교차점: LEO는 J2가 압도적, GEO 근처는 3체 섭동이 J2를 능가")
  print("=" * 70)
  rows = [altitude_regime_comparison(alt) for alt in ALTITUDE_SWEEP_KM]

  print(f"  {'고도(km)':>10}{'J2 가속도':>16}{'3체 섭동':>16}{'비율(3체/J2)':>16}")
  for row in rows:
    print(f"  {row['altitude_km']:>10.0f}{row['j2_accel_km_s2']:>16.3e}"
          f"{row['lunisolar_total_accel_km_s2']:>16.3e}{row['ratio_lunisolar_to_j2']:>16.6f}")

  ratios = [row["ratio_lunisolar_to_j2"] for row in rows]
  assert ratios[0] < 0.01, "LEO(300km)에서는 J2가 3체 섭동을 압도해야 함(비율 1% 미만)"
  assert ratios[-1] > ratios[0] * 100, "GEO 근처에서는 LEO 대비 비율이 수백 배 이상 커져야 함"
  assert all(ratios[i + 1] >= ratios[i] for i in range(len(ratios) - 1)), \
      "고도가 높아질수록 비율이 단조증가해야 함(J2는 1/r^4로 급감, 3체 섭동은 완만하게 변함)"
  print(f"\n(고도가 {ALTITUDE_SWEEP_KM[0]:.0f}km에서 {ALTITUDE_SWEEP_KM[-1]:.0f}km로 올라가는 동안")
  print(f" 비율이 {ratios[0]:.6f}에서 {ratios[-1]:.3f}로 커진다 - GEO 부근에서는 3체 섭동이")
  print(" J2와 같은 자릿수이거나 넘어서, 실제 GEO 위성 운용에서 중요한 보정 대상이 된다.)")
  return rows


def demo_third_body_perturbs_real_trajectory(semi_major_axis_km=7000.0, duration_days=15.0,
                                              dt_sec=60.0, record_every_n_steps=50):
  """원궤도에서 시작해 3체 섭동을 더한 궤적과 순수 2체 궤적을 나란히 적분해,
  최종 위치 편차와 섭동 가속도 크기 시계열을 기록한다. 편차가 실제로 자라고,
  섭동 가속도가 태양/달의 운동에 따라 진동한다는 것(고정된 스냅샷이 아님)을
  확인한다."""
  print("\n" + "=" * 70)
  print("[3] 실제 궤적 적분: 3체 섭동이 순수 2체 궤적에서 벗어나게 한다")
  print("=" * 70)
  v_circular = np.sqrt(EARTH_MU_KM3_S2 / semi_major_axis_km)
  pos_perturbed = np.array([semi_major_axis_km, 0.0, 0.0])
  vel_perturbed = np.array([0.0, v_circular, 0.0])
  pos_two_body = pos_perturbed.copy()
  vel_two_body = vel_perturbed.copy()

  total_steps = int(duration_days * 86400 / dt_sec)
  t = 0.0
  history = []
  for step in range(total_steps):
    pos_perturbed, vel_perturbed = rk4_step_with_third_body(pos_perturbed, vel_perturbed, dt_sec, t)
    pos_two_body, vel_two_body = rk4_step(pos_two_body, vel_two_body, dt_sec)
    t += dt_sec
    if step % record_every_n_steps == 0:
      deviation_km = np.linalg.norm(pos_perturbed - pos_two_body)
      accel_mag, _sun, _moon = lunisolar_acceleration(pos_perturbed, t)
      history.append((t, deviation_km, np.linalg.norm(accel_mag)))

  final_deviation_km = np.linalg.norm(pos_perturbed - pos_two_body)
  accel_mags = np.array([row[2] for row in history])
  cv = accel_mags.std() / accel_mags.mean()

  print(f"반장축={semi_major_axis_km}km, 적분기간={duration_days}일, dt={dt_sec}초")
  print(f"3체 섭동 적용 궤적 vs 순수 2체 궤적 최종 편차: {final_deviation_km:.3f}km")
  print(f"섭동 가속도 변동계수(std/mean): {cv:.3f}")

  assert final_deviation_km > 1.0, "충분한 적분 기간이면 3체 섭동으로 인한 편차가 1km를 넘어야 함"
  assert cv > 0.01, "태양/달이 움직이며 섭동 가속도가 시간에 따라 진동해야 함(고정된 스냅샷이 아님)"
  print("\n(짧은 기간에는 편차가 작지만, 보정 없이 누적되면 3체 섭동만으로도")
  print(" 궤적이 순수 2체 예측에서 측정 가능한 수준으로 벗어난다. 섭동 가속도")
  print(" 자체도 태양/달의 운동에 따라 계속 진동한다.)")
  return {"history": history, "final_deviation_km": final_deviation_km, "accel_cv": cv}


def parse_args():
  parser = argparse.ArgumentParser(description="태양/달 중력(3체 섭동)이 위성 궤도에 주는 영향 계산")
  parser.add_argument("--altitude-km", type=float, default=550.0, help="비교용 고도(km), 기본값: 550km(LEO)")
  parser.add_argument("--integration-days", type=float, default=15.0, help="궤도 적분 기간(일), 기본값: 15일")
  parser.add_argument("--semi-major-axis-km", type=float, default=7000.0, help="적분용 반장축(km), 기본값: 7000km")
  return parser.parse_args()


def main():
  args = parse_args()

  demo_lunar_perturbation_exceeds_solar_despite_smaller_mass(args.altitude_km)
  crossover_rows = demo_j2_vs_third_body_crossover_across_altitude()
  trajectory_result = demo_third_body_perturbs_real_trajectory(
      args.semi_major_axis_km, args.integration_days)

  custom_comparison = altitude_regime_comparison(args.altitude_km)
  print("\n" + "=" * 70)
  print(f"[사용자 지정] 고도={args.altitude_km}km, 반장축={args.semi_major_axis_km}km, "
        f"적분기간={args.integration_days}일")
  print("=" * 70)
  print(f"지정 고도에서 비율(3체섭동/J2)={custom_comparison['ratio_lunisolar_to_j2']:.6f}")
  print(f"적분 결과 최종 편차={trajectory_result['final_deviation_km']:.3f}km")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  magnitude_csv = os.path.join(results_dir, "third_body_sun_vs_moon_magnitude.csv")
  r_km = EARTH_RADIUS_KM + args.altitude_km
  position_km = np.array([r_km, 0.0, 0.0])
  _total, sun_accel, moon_accel = lunisolar_acceleration(position_km, time_sec=0.0)
  with open(magnitude_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["body", "acceleration_km_s2"])
    writer.writerow(["sun", f"{np.linalg.norm(sun_accel):.8e}"])
    writer.writerow(["moon", f"{np.linalg.norm(moon_accel):.8e}"])
  print(f"\n[기록] 태양 vs 달 가속도 비교 저장됨 → {magnitude_csv}")

  altitude_csv = os.path.join(results_dir, "third_body_j2_vs_altitude.csv")
  with open(altitude_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["altitude_km", "j2_accel_km_s2", "solar_accel_km_s2", "lunar_accel_km_s2",
                      "lunisolar_total_accel_km_s2", "ratio_lunisolar_to_j2"])
    for row in crossover_rows:
      writer.writerow([row["altitude_km"], f"{row['j2_accel_km_s2']:.8e}",
                        f"{row['solar_accel_km_s2']:.8e}", f"{row['lunar_accel_km_s2']:.8e}",
                        f"{row['lunisolar_total_accel_km_s2']:.8e}", f"{row['ratio_lunisolar_to_j2']:.8e}"])
  print(f"[기록] 고도별 J2 vs 3체 섭동 비교 저장됨 → {altitude_csv}")

  trajectory_csv = os.path.join(results_dir, "third_body_trajectory_deviation.csv")
  with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "deviation_km", "lunisolar_accel_mag_km_s2"])
    for t, deviation_km, accel_mag in trajectory_result["history"]:
      writer.writerow([f"{t:.2f}", f"{deviation_km:.6f}", f"{accel_mag:.8e}"])
  print(f"[기록] 궤적 편차 시계열 저장됨 → {trajectory_csv}")


if __name__ == "__main__":
  main()
