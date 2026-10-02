"""
태양복사압(SRP) 섭동 - 9번(대기항력)과 거울상, 고도가 높아지면 대기는 사라져도
태양빛은 남는다

9번(missions/orbital_decay.py)은 저궤도에서 지배적인 대기항력 감쇠를 다뤘다.
이 스크립트는 그 거울상이다 — 대기가 사실상 없는 고궤도에서는 태양복사압(SRP)이
남는 유일한 비중력 섭동이 된다. "저고도는 항력, 고고도는 광압"이라는 대비를
완성해, perturbations/ 폴더에 J2(07번)/3체섭동(23번)/SRP 세 가지가 모두
갖춰진다.

핵심 개념 1: 복사압은 태양 반대방향으로 미는 힘이다
  태양빛이 위성 표면에 흡수/반사되며 운동량을 전달한다. 단순화된 구형(cannonball)
  모델에서 가속도는
      a_srp = P_sr * Cr * (A/m) * unit(r_sat - r_sun)
  로 주어진다(P_sr=1AU에서의 복사압, Cr=반사계수, A/m=단면적대질량비). 방향은
  태양→위성 방향(r_sat - r_sun을 정규화)이다 - 위성을 태양으로부터 밀어내는
  방향이지 끌어당기는 방향이 아니다. 부호가 뒤집히면 물리적으로 틀린 결과(태양
  쪽으로 끌림)가 나오므로, 이 부호 자체를 항등식 테스트로 검증한다.

핵심 개념 2: 복사압 자체는 위성의 지구 중심 거리와 거의 무관하다
  P_sr은 태양-위성 거리(~1AU, 지구 공전궤도 반경)에 반비례하는데, 위성이 지구를
  도는 궤도반경(수백~수만 km)은 1AU(약 1.5억km)에 비해 무시할 수준이다. 그 결과
  SRP 가속도 크기는 LEO든 GEO든 거의 동일하다(약 1.36e-10 km/s²로 고정) -
  J2(1/r⁴)나 3체 섭동(완만한 증가)과는 본질적으로 다른 "고도 무관" 섭동이라는
  것이 23번과의 차이점이다. 이 때문에 이 스크립트는 SRP를 J2나 3체 섭동과
  비교하지 않는다 - 실제로 계산해보면 GEO에서도 J2 대비 비율이 1% 미만이라
  23번 같은 흥미로운 교차점이 나오지 않는다.

핵심 개념 3: 저고도는 항력, 고고도는 SRP가 압도한다 (9번과의 완결된 대비)
  9번의 지수함수 대기밀도 모델을 그대로 재사용해 각 고도의 항력 가속도와 SRP
  가속도를 비교하면, 200km에서는 항력이 SRP보다 수백 배 크지만, 3000km에서는
  반대로 SRP가 항력보다 수억~수조 배 커진다 - 약 550~800km 사이에서 교차가
  일어난다. 실제로 운용 위성의 궤도수명 연구가 약 600~800km 이하에서만 항력을
  신경 쓰는 이유와 정확히 일치하는 결과다.

핵심 개념 4: 지구 그림자(원통형 근사)에 들어가면 SRP가 정확히 0이 된다
  태양-지구 축을 기준으로, 위성이 (1) 태양 반대쪽에 있고 AND (2) 그 축에서
  수직거리가 지구 반지름 이내에 있으면 완전히 그림자 안에 있다고 근사한다
  (원통형 그림자, umbra/penumbra 구분이나 태양의 유한 각지름은 무시). 두 조건
  모두 만족해야 그림자로 판정해야 한다 - 하나만 확인하면 궤도 반대편에 있지만
  지구에서 먼 위치도 그림자로 잘못 판정하는 버그가 생긴다.

단순화: 캐논볼 모델(구형 위성, 자세 무관 단면적, 등방 반사)만 다룬다 - 실제
위성의 비구형 형상에 따른 자세 의존 단면적 변화는 다루지 않는다. 그림자는
원통형 근사만 쓴다(원뿔형 penumbra, 태양의 유한 각지름은 무시). 태양 위치는
23번(third_body_perturbation.py)의 원궤도 근사를 그대로 재사용한다. 반사계수
(Cr)와 단면적대질량비(A/m)를 CLI로 노출하되, 9번의 "탄도계수(BC)" 용어는 쓰지
않는다 - BC는 항력계수(Cd)까지 포함하는 다른 물리량이라 SRP에 재사용하면
오해를 부른다(다만 숫자 자체는 9번의 기본 BC=50과 비교 가능하도록
A/m=1/50=0.02로 맞춘다). 궤적 적분 데모에서는 SRP만 단독으로 다룬다(항력/J2/
3체섭동과 결합하지 않음 - 9번의 항력 단독 적분 데모와 같은 원칙). 식(eclipse)
데모는 궤도면을 태양 방향과 일치시킨(베타각 0도) 특수 기하를 의도적으로
선택한 것이며, 일반적인 궤도의 식 비율 주장이 아니다.

9번(orbital_decay.py)과의 관계: atmospheric_density, circular_orbit_speed를
그대로 재사용해 항력-SRP 비교가 진짜 같은 모델에 기반하도록 한다. 23번
(third_body_perturbation.py)과의 관계: sun_position_eci_km과 태양 관련 상수를
그대로 재사용한다 - 둘 다 같은 원궤도 근사 태양 모델을 공유해야 일관성이
생긴다.
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

_ORBIT_MATH_PATH = os.path.join(_ROOT_DIR, "orbit_math.py")
_orbit_math_spec = importlib.util.spec_from_file_location("orbit_math", _ORBIT_MATH_PATH)
orbit_math = importlib.util.module_from_spec(_orbit_math_spec)
_orbit_math_spec.loader.exec_module(orbit_math)
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM

_THIRD_BODY_PATH = os.path.join(_THIS_DIR, "third_body_perturbation.py")
_third_body_spec = importlib.util.spec_from_file_location("third_body_module", _THIRD_BODY_PATH)
third_body_module = importlib.util.module_from_spec(_third_body_spec)
_third_body_spec.loader.exec_module(third_body_module)
sun_position_eci_km = third_body_module.sun_position_eci_km

_ORBITAL_DECAY_PATH = os.path.join(_ROOT_DIR, "missions", "orbital_decay.py")
_orbital_decay_spec = importlib.util.spec_from_file_location("orbital_decay_module", _ORBITAL_DECAY_PATH)
orbital_decay_module = importlib.util.module_from_spec(_orbital_decay_spec)
_orbital_decay_spec.loader.exec_module(orbital_decay_module)
atmospheric_density = orbital_decay_module.atmospheric_density
drag_deceleration = orbital_decay_module.drag_deceleration
circular_orbit_speed = orbital_decay_module.circular_orbit_speed

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2

SOLAR_CONSTANT_W_M2 = 1361.0
SPEED_OF_LIGHT_KM_S = 299792.458
DEFAULT_REFLECTIVITY_COEFFICIENT = 1.5  # Cr, gray body 가정
DEFAULT_AREA_TO_MASS_RATIO_M2_KG = 1.0 / 50.0  # 9번의 BC=50과 비교 가능하도록

ALTITUDE_SWEEP_KM = [200.0, 300.0, 400.0, 550.0, 800.0, 1500.0, 3000.0]


def srp_acceleration(position_km, sun_position_km, reflectivity_coefficient, area_to_mass_ratio_m2_kg):
  """a_srp = P_sr*Cr*(A/m)*unit(r_sat-r_sun). 태양->위성 방향(태양 반대쪽)을
  향한다 - 부호가 뒤집히면 태양 쪽으로 끌리는 물리적 오류가 됨."""
  position_km = np.array(position_km, dtype=float)
  sun_position_km = np.array(sun_position_km, dtype=float)
  r_sun_to_sat = position_km - sun_position_km
  direction = r_sun_to_sat / np.linalg.norm(r_sun_to_sat)
  p_sr = SOLAR_CONSTANT_W_M2 / (SPEED_OF_LIGHT_KM_S * 1000.0)  # N/m^2(Pa), km/s -> m/s 환산
  accel_m_s2 = p_sr * reflectivity_coefficient * area_to_mass_ratio_m2_kg
  return (accel_m_s2 / 1000.0) * direction  # km/s^2


def is_in_shadow(position_km, sun_position_km, earth_radius_km=EARTH_RADIUS_KM):
  """원통형 그림자 근사: (1) 태양 반대쪽(night side)에 있고 AND (2) 태양-지구
  축에서 수직거리가 지구 반지름 이내인 경우에만 그림자로 판정한다. 두 조건
  모두 필요 - 하나만 확인하면 궤도 반대편이지만 지구에서 먼 위치도 그림자로
  잘못 판정한다."""
  position_km = np.array(position_km, dtype=float)
  sun_dir = np.array(sun_position_km, dtype=float) / np.linalg.norm(sun_position_km)
  along_sun_axis = np.dot(position_km, sun_dir)
  behind_earth = along_sun_axis < 0
  perp_vector = position_km - along_sun_axis * sun_dir
  perp_distance = np.linalg.norm(perp_vector)
  return bool(behind_earth and perp_distance < earth_radius_km)


def srp_acceleration_with_shadow(position_km, sun_position_km, reflectivity_coefficient, area_to_mass_ratio_m2_kg):
  """그림자 안이면 정확히 0 벡터, 아니면 srp_acceleration 그대로."""
  if is_in_shadow(position_km, sun_position_km):
    return np.zeros(3)
  return srp_acceleration(position_km, sun_position_km, reflectivity_coefficient, area_to_mass_ratio_m2_kg)


def rk4_step_with_srp(position_km, velocity_km_s, dt_sec, time_sec,
                       reflectivity_coefficient, area_to_mass_ratio_m2_kg, mu=EARTH_MU_KM3_S2):
  """23번 rk4_step_with_third_body와 동일한 시간의존 RK4 클론 구조 - derivative
  클로저가 (pos, vel, t)를 받고 매 k단계마다 태양 위치를 그 시각에 다시
  계산한다."""
  def derivative(pos, vel, t):
    sun_pos = sun_position_eci_km(t)
    accel = two_body_acceleration(pos, mu) + srp_acceleration_with_shadow(
        pos, sun_pos, reflectivity_coefficient, area_to_mass_ratio_m2_kg)
    return vel, accel

  k1_r, k1_v = derivative(position_km, velocity_km_s, time_sec)
  k2_r, k2_v = derivative(position_km + dt_sec / 2 * k1_r, velocity_km_s + dt_sec / 2 * k1_v, time_sec + dt_sec / 2)
  k3_r, k3_v = derivative(position_km + dt_sec / 2 * k2_r, velocity_km_s + dt_sec / 2 * k2_v, time_sec + dt_sec / 2)
  k4_r, k4_v = derivative(position_km + dt_sec * k3_r, velocity_km_s + dt_sec * k3_v, time_sec + dt_sec)
  new_position = position_km + dt_sec / 6 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
  new_velocity = velocity_km_s + dt_sec / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
  return new_position, new_velocity


def drag_srp_altitude_comparison(altitude_km, reflectivity_coefficient, area_to_mass_ratio_m2_kg, mu=EARTH_MU_KM3_S2):
  """주어진 고도에서 항력 감속(9번 atmospheric_density/circular_orbit_speed
  재사용) vs SRP 가속도 크기를 비교한다. BC=1/(A/m)로 환산해 9번의
  drag_deceleration을 그대로 호출 - 이 변환이 BC 용어가 재등장하는 유일한
  지점(CLI에는 노출하지 않음)."""
  r_km = EARTH_RADIUS_KM + altitude_km
  speed_km_s = circular_orbit_speed(r_km, mu)
  velocity_km_s = np.array([0.0, speed_km_s, 0.0])
  ballistic_coefficient_kg_m2 = 1.0 / area_to_mass_ratio_m2_kg
  drag_accel = np.linalg.norm(drag_deceleration(velocity_km_s, altitude_km, ballistic_coefficient_kg_m2))

  position_km = np.array([r_km, 0.0, 0.0])
  sun_position_km = sun_position_eci_km(0.0)
  srp_accel = np.linalg.norm(srp_acceleration(
      position_km, sun_position_km, reflectivity_coefficient, area_to_mass_ratio_m2_kg))

  return {
      "altitude_km": altitude_km,
      "drag_accel_km_s2": drag_accel,
      "srp_accel_km_s2": srp_accel,
      "ratio_srp_to_drag": srp_accel / drag_accel,
  }


def demo_srp_magnitude_is_physically_sane(reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT,
                                           area_to_mass_ratio_m2_kg=DEFAULT_AREA_TO_MASS_RATIO_M2_KG):
  """임의 위치에서 SRP 가속도 크기와 방향을 계산해 물리적으로 타당한 범위
  안에 있는지, 그리고 태양 반대방향을 향하는지 확인한다."""
  print("=" * 70)
  print("[1] SRP 가속도 크기/방향: 물리적으로 타당한가")
  print("=" * 70)
  sun_position_km = sun_position_eci_km(0.0)
  position_km = np.array([7000.0, 0.0, 0.0])
  accel = srp_acceleration(position_km, sun_position_km, reflectivity_coefficient, area_to_mass_ratio_m2_kg)
  accel_mag = np.linalg.norm(accel)
  direction_sun_to_sat = (position_km - sun_position_km)
  direction_sun_to_sat /= np.linalg.norm(direction_sun_to_sat)

  print(f"반사계수 Cr={reflectivity_coefficient}, 단면적대질량비 A/m={area_to_mass_ratio_m2_kg}m^2/kg")
  print(f"SRP 가속도 크기: {accel_mag:.3e}km/s^2")
  print(f"태양->위성 방향 성분: {np.dot(accel, direction_sun_to_sat):.3e} (양수여야 태양 반대방향)")

  assert 1e-12 < accel_mag < 1e-8, "SRP 가속도 크기는 전형적인 위성에서 물리적으로 타당한 범위(1e-12~1e-8 km/s^2) 안에 있어야 함"
  assert np.dot(accel, direction_sun_to_sat) > 0, "SRP 가속도는 태양->위성 방향(태양 반대쪽)을 향해야 함 - 부호가 뒤집히면 태양 쪽으로 끌리는 오류"

  print("\n(복사압은 아주 작지만(1e-10 km/s^2 안팎) 0이 아니다 - 장기간 누적되면")
  print(" GPS/GEO 위성의 궤도에 측정 가능한 영향을 준다는 것이 알려져 있다.)")
  return {"accel_mag_km_s2": accel_mag}


def demo_drag_dominates_low_srp_dominates_high_altitude(reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT,
                                                          area_to_mass_ratio_m2_kg=DEFAULT_AREA_TO_MASS_RATIO_M2_KG):
  """9번의 항력 모델을 재사용해 고도별로 항력과 SRP를 비교한다 - 저고도에서는
  항력이 압도적이지만 고고도에서는 SRP가 유일하게 남는 섭동이 된다는, 9번과의
  거울상 관계를 완성하는 핵심 데모."""
  print("\n" + "=" * 70)
  print("[2] 고도별 교차점: 저고도는 항력, 고고도는 SRP가 지배")
  print("=" * 70)
  rows = [drag_srp_altitude_comparison(alt, reflectivity_coefficient, area_to_mass_ratio_m2_kg)
          for alt in ALTITUDE_SWEEP_KM]

  print(f"  {'고도(km)':>10}{'항력 가속도':>16}{'SRP 가속도':>16}{'비율(SRP/항력)':>18}")
  for row in rows:
    print(f"  {row['altitude_km']:>10.0f}{row['drag_accel_km_s2']:>16.3e}"
          f"{row['srp_accel_km_s2']:>16.3e}{row['ratio_srp_to_drag']:>18.3e}")

  ratios = [row["ratio_srp_to_drag"] for row in rows]
  assert ratios[0] < 0.01, "200km에서는 항력이 SRP를 압도해야 함(비율 1% 미만)"
  assert ratios[-1] > 1e6, "3000km에서는 SRP가 항력을 압도해야 함(비율 100만배 이상)"
  assert any(0.1 < r < 10 for r in ratios), "두 섭동이 비슷한 크기가 되는 교차점이 스윕 범위 안에 존재해야 함"
  assert all(ratios[i + 1] >= ratios[i] for i in range(len(ratios) - 1)), \
      "고도가 높아질수록 비율이 단조증가해야 함(항력은 지수함수로 급감, SRP는 거의 일정)"

  print("\n(9번이 저궤도에서 지배적인 항력을 다뤘다면, 이 스크립트는 고궤도에서")
  print(" 항력이 사실상 사라진 뒤에도 남는 유일한 비중력 섭동을 보여준다 -")
  print(" 약 550~800km 사이에서 두 섭동의 크기가 역전된다.)")
  return rows


def demo_eclipse_zeroes_out_srp_during_shadow_passes(semi_major_axis_km=7000.0,
                                                       reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT,
                                                       area_to_mass_ratio_m2_kg=DEFAULT_AREA_TO_MASS_RATIO_M2_KG,
                                                       dt_sec=10.0):
  """궤도면을 태양 방향과 일치시킨(베타각 0도) 원형 적도 궤도를 한 주기
  적분해, 지구 그림자를 지나는 동안 SRP 가속도가 정확히 0이 되는지 확인한다.
  이 궤도 기하는 식 효과를 뚜렷하게 보여주기 위해 의도적으로 선택한 특수
  케이스이며, 일반적인 궤도의 식 비율 주장이 아니다."""
  print("\n" + "=" * 70)
  print("[3] 식(eclipse) 효과: 지구 그림자를 지나는 동안 SRP가 정확히 0이 되는가")
  print("=" * 70)
  speed_km_s = circular_orbit_speed(semi_major_axis_km)
  position_km = np.array([semi_major_axis_km, 0.0, 0.0])
  velocity_km_s = np.array([0.0, speed_km_s, 0.0])
  period_sec = 2 * np.pi * np.sqrt(semi_major_axis_km ** 3 / EARTH_MU_KM3_S2)

  t = 0.0
  accel_mags = []
  full_value = np.linalg.norm(srp_acceleration(
      position_km, sun_position_eci_km(0.0), reflectivity_coefficient, area_to_mass_ratio_m2_kg))
  while t < period_sec:
    sun_pos = sun_position_eci_km(t)
    accel = srp_acceleration_with_shadow(position_km, sun_pos, reflectivity_coefficient, area_to_mass_ratio_m2_kg)
    accel_mags.append(np.linalg.norm(accel))
    position_km, velocity_km_s = rk4_step_with_srp(
        position_km, velocity_km_s, dt_sec, t, reflectivity_coefficient, area_to_mass_ratio_m2_kg)
    t += dt_sec

  accel_mags = np.array(accel_mags)
  in_shadow_count = np.sum(accel_mags < 1e-20)
  eclipse_fraction = in_shadow_count / len(accel_mags)

  print(f"반장축={semi_major_axis_km}km, 궤도주기={period_sec / 60:.1f}분")
  print(f"전체 조명 시 SRP 가속도: {full_value:.3e}km/s^2")
  print(f"식 비율(그림자에 있는 시간 비중): {eclipse_fraction * 100:.1f}%")

  assert np.any(accel_mags < 1e-20), "궤도 일부 구간에서는 SRP 가속도가 정확히 0이어야 함(그림자)"
  assert np.any(np.isclose(accel_mags, full_value, rtol=1e-6)), "궤도 일부 구간에서는 SRP 가속도가 전체 조명값과 같아야 함(일광)"
  assert 0.30 < eclipse_fraction < 0.45, "베타각 0도 원형 적도 궤도의 식 비율은 실제 LEO 위성의 전형적 범위(약 30~40%) 안에 있어야 함"

  print("\n(궤도면을 태양 방향과 일치시킨 특수 기하에서, 궤도의 상당 부분(약")
  print(" 1/3)이 지구 그림자 안에 들어가 SRP가 정확히 0이 된다 - 일반적인")
  print(" 궤도의 식 비율은 베타각에 따라 달라지므로 이 수치는 이 특수 케이스에만")
  print(" 해당한다.)")
  return {"eclipse_fraction": eclipse_fraction, "full_value_km_s2": full_value}


def demo_larger_area_to_mass_ratio_increases_srp_effect(reflectivity_coefficient=DEFAULT_REFLECTIVITY_COEFFICIENT):
  """단면적대질량비(A/m)를 스윕해 SRP 가속도가 A/m에 정확히 선형 비례하는지
  확인한다 - 공식 자체가 A/m에 선형이므로 거의 항등식에 가까운 검증이다."""
  print("\n" + "=" * 70)
  print("[4] 면적대질량비 민감도: SRP는 A/m에 정확히 선형 비례하는가")
  print("=" * 70)
  sun_position_km = sun_position_eci_km(0.0)
  position_km = np.array([7000.0, 0.0, 0.0])
  area_to_mass_ratios = [0.01, 0.02, 0.05, 0.1]
  rows = []
  for ratio in area_to_mass_ratios:
    accel_mag = np.linalg.norm(srp_acceleration(position_km, sun_position_km, reflectivity_coefficient, ratio))
    rows.append({"area_to_mass_ratio_m2_kg": ratio, "srp_accel_km_s2": accel_mag})
    print(f"  A/m={ratio:.3f}m^2/kg  SRP 가속도={accel_mag:.3e}km/s^2")

  a1 = rows[0]["srp_accel_km_s2"]
  ratio1 = rows[0]["area_to_mass_ratio_m2_kg"]
  for row in rows[1:]:
    expected_scale = row["area_to_mass_ratio_m2_kg"] / ratio1
    actual_scale = row["srp_accel_km_s2"] / a1
    assert abs(actual_scale - expected_scale) < 1e-6 * expected_scale, \
        "SRP 가속도는 면적대질량비에 정확히 선형 비례해야 함(공식 자체가 선형)"

  print("\n(무겁고 작은 위성(A/m 작음)보다 가볍고 넓은 위성(A/m 큼, 태양돛에")
  print(" 가까운 극단)일수록 SRP의 영향을 더 크게 받는다 - 공식이 선형이므로")
  print(" 당연한 결과지만, 실제로 수치가 정확히 비례하는지 직접 확인했다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="태양복사압(SRP) 섭동이 위성 궤도에 주는 영향 계산")
  parser.add_argument("--reflectivity-coefficient", type=float, default=DEFAULT_REFLECTIVITY_COEFFICIENT,
                       help="복사압 반사계수 Cr, 기본값: 1.5(gray body 가정)")
  parser.add_argument("--area-to-mass-ratio-m2-kg", type=float, default=DEFAULT_AREA_TO_MASS_RATIO_M2_KG,
                       help="단면적대질량비(m^2/kg), 기본값: 0.02(9번 BC=50과 동등)")
  parser.add_argument("--semi-major-axis-km", type=float, default=7000.0,
                       help="식(eclipse) 데모용 반장축(km), 기본값: 7000km(LEO)")
  return parser.parse_args()


def main():
  args = parse_args()

  demo_srp_magnitude_is_physically_sane(args.reflectivity_coefficient, args.area_to_mass_ratio_m2_kg)
  crossover_rows = demo_drag_dominates_low_srp_dominates_high_altitude(
      args.reflectivity_coefficient, args.area_to_mass_ratio_m2_kg)
  eclipse_result = demo_eclipse_zeroes_out_srp_during_shadow_passes(
      args.semi_major_axis_km, args.reflectivity_coefficient, args.area_to_mass_ratio_m2_kg)
  sensitivity_rows = demo_larger_area_to_mass_ratio_increases_srp_effect(args.reflectivity_coefficient)

  print("\n" + "=" * 70)
  print(f"[사용자 지정] Cr={args.reflectivity_coefficient}, A/m={args.area_to_mass_ratio_m2_kg}m^2/kg, "
        f"반장축={args.semi_major_axis_km}km")
  print("=" * 70)
  print(f"식 비율: {eclipse_result['eclipse_fraction'] * 100:.1f}%")

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  altitude_csv = os.path.join(results_dir, "solar_radiation_pressure_altitude_vs_drag.csv")
  with open(altitude_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["altitude_km", "drag_accel_km_s2", "srp_accel_km_s2", "ratio_srp_to_drag"])
    for row in crossover_rows:
      writer.writerow([row["altitude_km"], f"{row['drag_accel_km_s2']:.8e}",
                        f"{row['srp_accel_km_s2']:.8e}", f"{row['ratio_srp_to_drag']:.8e}"])
  print(f"\n[기록] 고도별 항력 vs SRP 비교 저장됨 → {altitude_csv}")

  eclipse_csv = os.path.join(results_dir, "solar_radiation_pressure_eclipse_trajectory.csv")
  speed_km_s = circular_orbit_speed(args.semi_major_axis_km)
  position_km = np.array([args.semi_major_axis_km, 0.0, 0.0])
  velocity_km_s = np.array([0.0, speed_km_s, 0.0])
  period_sec = 2 * np.pi * np.sqrt(args.semi_major_axis_km ** 3 / EARTH_MU_KM3_S2)
  dt_sec = 10.0
  t = 0.0
  with open(eclipse_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t_sec", "x_km", "y_km", "z_km", "in_shadow", "srp_accel_mag_km_s2"])
    while t < period_sec:
      sun_pos = sun_position_eci_km(t)
      in_shadow = is_in_shadow(position_km, sun_pos)
      accel = srp_acceleration_with_shadow(
          position_km, sun_pos, args.reflectivity_coefficient, args.area_to_mass_ratio_m2_kg)
      writer.writerow([f"{t:.2f}", f"{position_km[0]:.4f}", f"{position_km[1]:.4f}", f"{position_km[2]:.4f}",
                        in_shadow, f"{np.linalg.norm(accel):.8e}"])
      position_km, velocity_km_s = rk4_step_with_srp(
          position_km, velocity_km_s, dt_sec, t, args.reflectivity_coefficient, args.area_to_mass_ratio_m2_kg)
      t += dt_sec
  print(f"[기록] 식 궤적 시계열 저장됨 → {eclipse_csv}")

  sensitivity_csv = os.path.join(results_dir, "solar_radiation_pressure_area_to_mass_sensitivity.csv")
  with open(sensitivity_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["area_to_mass_ratio_m2_kg", "srp_accel_km_s2"])
    for row in sensitivity_rows:
      writer.writerow([row["area_to_mass_ratio_m2_kg"], f"{row['srp_accel_km_s2']:.8e}"])
  print(f"[기록] 면적대질량비 민감도 저장됨 → {sensitivity_csv}")


if __name__ == "__main__":
  main()
