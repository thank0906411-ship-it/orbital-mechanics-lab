"""
궤도역학 기초 심화 - 궤도요소 <-> 상태벡터 변환과 보존량(에너지, 각운동량)

01번은 궤도면(perifocal frame) 안에서의 2차원 위치(x_p, y_p)까지만 계산했다. 실제
위성은 3차원 공간에 있고, 그 궤도면이 지구 중심 관성좌표계(ECI)에 대해 임의의 각도로
기울어져 있다 — 이 3차원 방향을 정하는 것이 나머지 3개 궤도요소(경사각 i, 승교점
적경 RAAN, 근점 편각 argp)다. 이 스크립트는 6개 궤도요소 전체를 3차원 위치/속도
벡터(상태벡터)로 변환하고, 그 반대(상태벡터 -> 궤도요소)도 구현한 뒤, 2체 문제에서
반드시 보존돼야 하는 물리량(비에너지, 비각운동량)이 궤도를 도는 내내 실제로 일정한지
검증한다.

핵심 개념 1: 궤도면에서 ECI로 가는 3번의 회전
  궤도면 좌표 (x_p, y_p, 0)에 다음 순서로 회전을 적용하면 ECI 좌표가 나온다:
    1) 근점 편각(argp)만큼 z축으로 회전 (궤도면 내에서 근점의 방향을 맞춤)
    2) 경사각(i)만큼 x축으로 회전 (궤도면을 실제 기울기로 세움)
    3) 승교점 적경(RAAN)만큼 z축으로 회전 (궤도면이 적도면과 만나는 선의 방향을 맞춤)
  이 세 회전을 합성한 3x3 회전행렬 R = Rz(RAAN) * Rx(i) * Rz(argp)를 궤도면 좌표에
  곱하면 ECI 위치가 나온다. 속도 벡터도 같은 회전행렬을 궤도면 속도에 곱해서 구한다.

핵심 개념 2: 궤도면에서의 속도는 각운동량과 반사통경(semi-latus rectum)으로 구해진다
  궤도면 좌표계에서 속도 성분은
      vx_p = -mu/h * sin(nu),  vy_p = mu/h * (e + cos(nu))
  로 주어진다 (h는 비각운동량, mu는 중심천체 인력상수). 이 식은 각운동량 보존
  (h = r^2 * d(nu)/dt = 일정)으로부터 유도된다 — 위치와 속도를 동시에 다루려면
  이 궤도면 속도식이 필요하다.

핵심 개념 3: vis-viva 방정식과 두 가지 보존량
  vis-viva 방정식 v^2 = mu*(2/r - 1/a)는 궤도 위 어느 지점에서든 속력을 반장축과
  거리만으로 구할 수 있게 해준다. 이로부터 비에너지(specific energy)
      epsilon = v^2/2 - mu/r = -mu/(2a)
  가 궤도 전체에서 오직 반장축(a)에만 의존하는 상수임을 알 수 있다. 마찬가지로
  비각운동량(specific angular momentum) h = |r x v| = sqrt(mu*a*(1-e^2))도 상수다.
  이 스크립트는 01번이 만든 궤도 위 여러 지점에서 실제로 v를 계산해, epsilon과 h가
  이론값과 일치하고 지점에 따라 변하지 않는지 수치로 확인한다.

핵심 개념 4: 상태벡터 -> 궤도요소 역변환은 몇 가지 특이 케이스가 있다
  비각운동량 벡터 h = r x v, 이심률 벡터 e_vec = (v x h)/mu - r/|r| 로부터 e=|e_vec|,
  a = -mu/(2*epsilon)를 구하고, 경사각은 cos(i) = h_z/|h|, 승교점 방향 벡터
  n = z_hat x h 로부터 RAAN = atan2(n_y, n_x)를 구한다. 다만 원궤도(e=0)에서는
  근점 자체가 정의되지 않아 근점 편각(argp)이 무의미해지고, 적도궤도(i=0 또는 180도)
  에서는 승교점 자체가 없어 RAAN이 무의미해진다 — 이 스크립트는 이런 특이 케이스를
  감추지 않고, 01번의 원궤도 데모(e=0)로 역변환을 시도했을 때 실제로 근점 편각이
  불안정해지는 것을 직접 보여준다.

01번(케플러 궤도 전파)과의 관계: 이 스크립트는 01번의 solve_kepler_equation,
eccentric_to_true_anomaly, perifocal_position, EARTH_MU_KM3_S2를 그대로 재사용한다.
01번이 만든 궤도면 내 위치에 이 스크립트의 3차원 회전과 속도식을 더해 실제 위성의
전체 상태벡터를 완성한다.
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
_KEPLER_PATH = os.path.join(_THIS_DIR, "kepler_orbit_propagation.py")
_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kepler)

_orbit_math_spec = importlib.util.spec_from_file_location("orbit_math", os.path.join(_ROOT_DIR, "orbit_math.py"))
orbit_math = importlib.util.module_from_spec(_orbit_math_spec)
_orbit_math_spec.loader.exec_module(orbit_math)
rotation_matrix_x = orbit_math.rotation_matrix_x
rotation_matrix_z = orbit_math.rotation_matrix_z

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2

# 원궤도(e~0)/적도궤도(i~0,180도)에서 근점편각/RAAN이 물리적으로 정의되지 않는 경계를
# 판정하는 공통 허용오차. 두 판정 모두 이 값 하나로 통제한다 — 실제 SGP4 출력처럼
# 잡음이 섞인 입력에 맞춰 느슨하게 조정할 때 값을 한 곳에서만 바꾸면 되게 하기 위함이다.
SINGULARITY_TOLERANCE = 1e-9


def perifocal_velocity(mu, semi_latus_rectum_km, eccentricity, true_anomaly_rad):
  """궤도면 좌표계에서의 속도: vx_p = -mu/h*sin(nu), vy_p = mu/h*(e+cos(nu)).
  h = sqrt(mu*p) (p: 반사통경, semi-latus rectum)."""
  h = np.sqrt(mu * semi_latus_rectum_km)
  vx_p = -mu / h * np.sin(true_anomaly_rad)
  vy_p = mu / h * (eccentricity + np.cos(true_anomaly_rad))
  return vx_p, vy_p


def orbital_elements_to_state_vector(semi_major_axis_km, eccentricity, inclination_rad,
                                      raan_rad, argp_rad, true_anomaly_rad, ecc_anomaly_rad,
                                      mu=EARTH_MU_KM3_S2):
  """6개 궤도요소 + (진근점 이각, 이심 이각)으로부터 ECI 위치/속도 벡터를 만든다.
  반환: (position_km[3], velocity_km_s[3])."""
  semi_latus_rectum = semi_major_axis_km * (1 - eccentricity ** 2)
  _r, x_p, y_p = kepler.perifocal_position(semi_major_axis_km, eccentricity, true_anomaly_rad, ecc_anomaly_rad)
  vx_p, vy_p = perifocal_velocity(mu, semi_latus_rectum, eccentricity, true_anomaly_rad)

  position_perifocal = np.array([x_p, y_p, 0.0])
  velocity_perifocal = np.array([vx_p, vy_p, 0.0])

  rotation = rotation_matrix_z(raan_rad) @ rotation_matrix_x(inclination_rad) @ rotation_matrix_z(argp_rad)
  position_eci = rotation @ position_perifocal
  velocity_eci = rotation @ velocity_perifocal
  return position_eci, velocity_eci


def state_vector_to_orbital_elements(position_km, velocity_km_s, mu=EARTH_MU_KM3_S2):
  """상태벡터 -> 궤도요소 역변환. 원궤도(e~0)나 적도궤도(i~0/180도)에서는 argp/RAAN이
  불안정해질 수 있다는 것을 그대로 반환값에 남긴다(감추지 않음)."""
  r_vec = np.array(position_km, dtype=float)
  v_vec = np.array(velocity_km_s, dtype=float)
  r = np.linalg.norm(r_vec)
  v = np.linalg.norm(v_vec)

  h_vec = np.cross(r_vec, v_vec)
  h = np.linalg.norm(h_vec)

  specific_energy = v ** 2 / 2 - mu / r
  semi_major_axis = -mu / (2 * specific_energy)

  e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
  eccentricity = np.linalg.norm(e_vec)

  inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

  z_hat = np.array([0.0, 0.0, 1.0])
  node_vec = np.cross(z_hat, h_vec)
  node_norm = np.linalg.norm(node_vec)

  if node_norm < SINGULARITY_TOLERANCE:
    raan = 0.0  # 적도궤도: 승교점 자체가 정의되지 않음
  else:
    raan = np.arctan2(node_vec[1], node_vec[0])
    if raan < 0:
      raan += 2 * np.pi

  if eccentricity < SINGULARITY_TOLERANCE or node_norm < SINGULARITY_TOLERANCE:
    argp = 0.0  # 원궤도: 근점 자체가 정의되지 않음
  else:
    cos_argp = np.dot(node_vec, e_vec) / (node_norm * eccentricity)
    argp = np.arccos(np.clip(cos_argp, -1.0, 1.0))
    if e_vec[2] < 0:
      argp = 2 * np.pi - argp

  return {
      "semi_major_axis_km": semi_major_axis, "eccentricity": eccentricity,
      "inclination_rad": inclination, "raan_rad": raan, "argp_rad": argp,
      "specific_energy": specific_energy, "angular_momentum": h,
  }


def demo_perifocal_to_eci_roundtrip():
  """궤도요소로 ECI 상태벡터를 만들고, 그걸 다시 궤도요소로 역변환했을 때 원래
  값(a, e, i, RAAN)이 복원되는지 확인한다."""
  print("=" * 70)
  print("[1] 궤도요소 -> ECI 상태벡터 -> 궤도요소 왕복 변환 검증")
  print("=" * 70)
  test_cases = [
      {"a": 7000.0, "e": 0.1, "i": np.radians(45.0), "raan": np.radians(30.0), "argp": np.radians(60.0), "nu": np.radians(20.0)},
      {"a": 10000.0, "e": 0.4, "i": np.radians(97.0), "raan": np.radians(120.0), "argp": np.radians(200.0), "nu": np.radians(150.0)},
  ]
  print(f"  {'반장축(km)':>12}{'이심률':>10}{'경사각(도)':>12}{'RAAN(도)':>12}{'역변환 a':>14}{'역변환 e':>12}")

  rows = []
  for case in test_cases:
    ecc_anomaly = 2 * np.arctan2(
        np.sqrt(1 - case["e"]) * np.sin(case["nu"] / 2), np.sqrt(1 + case["e"]) * np.cos(case["nu"] / 2))
    pos, vel = orbital_elements_to_state_vector(
        case["a"], case["e"], case["i"], case["raan"], case["argp"], case["nu"], ecc_anomaly)
    recovered = state_vector_to_orbital_elements(pos, vel)
    print(f"  {case['a']:>12.1f}{case['e']:>10.3f}{np.degrees(case['i']):>12.2f}"
          f"{np.degrees(case['raan']):>12.2f}{recovered['semi_major_axis_km']:>14.2f}{recovered['eccentricity']:>12.4f}")
    rows.append({
        "input_a": case["a"], "input_e": case["e"], "input_i_deg": np.degrees(case["i"]),
        "input_raan_deg": np.degrees(case["raan"]),
        "recovered_a": recovered["semi_major_axis_km"], "recovered_e": recovered["eccentricity"],
        "recovered_i_deg": np.degrees(recovered["inclination_rad"]),
        "recovered_raan_deg": np.degrees(recovered["raan_rad"]),
    })
    assert abs(recovered["semi_major_axis_km"] - case["a"]) < 1e-6, "반장축이 왕복 변환 후 복원돼야 함"
    assert abs(recovered["eccentricity"] - case["e"]) < 1e-6, "이심률이 왕복 변환 후 복원돼야 함"
    assert abs(np.degrees(recovered["inclination_rad"]) - np.degrees(case["i"])) < 1e-6

  print("\n(6개 궤도요소가 3차원 상태벡터로, 다시 6개 궤도요소로 정확히 왕복 복원된다 —")
  print(" 회전행렬 합성과 역변환 공식이 서로 정확히 대응한다는 뜻이다.)")
  return rows


def demo_conserved_quantities_along_orbit():
  """01번의 궤도 위 여러 지점에서 비에너지(epsilon)와 비각운동량(h)이 실제로
  일정한지 확인한다 — 2체 문제의 핵심 물리 법칙."""
  print("\n" + "=" * 70)
  print("[2] 궤도를 도는 내내 비에너지와 비각운동량이 보존되는가")
  print("=" * 70)
  a, e, i, raan, argp = 8000.0, 0.5, np.radians(30.0), np.radians(10.0), np.radians(50.0)
  expected_energy = -EARTH_MU_KM3_S2 / (2 * a)
  expected_h = np.sqrt(EARTH_MU_KM3_S2 * a * (1 - e ** 2))
  print(f"이론값: 비에너지 epsilon = {expected_energy:.6f} km^2/s^2, "
        f"비각운동량 h = {expected_h:.4f} km^2/s\n")

  period = 2 * np.pi / kepler.mean_motion(a)
  sample_fractions = np.linspace(0, 0.9, 6)
  print(f"  {'궤도 진행률':>12}{'거리(km)':>12}{'속력(km/s)':>14}{'비에너지':>16}{'비각운동량':>16}")

  rows = []
  for frac in sample_fractions:
    t = frac * period
    state = kepler.propagate_orbit(a, e, mean_anomaly0_rad=0.0, time_sec=t)
    pos, vel = orbital_elements_to_state_vector(a, e, i, raan, argp, state["true_anomaly"], state["ecc_anomaly"])
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    energy = v ** 2 / 2 - EARTH_MU_KM3_S2 / r
    h = np.linalg.norm(np.cross(pos, vel))
    print(f"  {frac:>12.2f}{r:>12.1f}{v:>14.4f}{energy:>16.6f}{h:>16.4f}")
    rows.append({"time_fraction": frac, "r_km": r, "v_km_s": v, "specific_energy": energy, "angular_momentum": h})

  max_energy_error = max(abs(row["specific_energy"] - expected_energy) for row in rows)
  max_h_error = max(abs(row["angular_momentum"] - expected_h) for row in rows)
  print(f"\n최대 비에너지 오차: {max_energy_error:.2e}, 최대 비각운동량 오차: {max_h_error:.2e}")

  assert max_energy_error < 1e-6, "비에너지는 궤도 전체에서 상수여야 함(오차가 수치정밀도 수준이어야 함)"
  assert max_h_error < 1e-6, "비각운동량은 궤도 전체에서 상수여야 함"
  return rows


def demo_vis_viva_speed_prediction():
  """vis-viva 방정식으로 예측한 속력이, 실제 상태벡터에서 계산한 속력과 일치하는지
  근지점과 원지점에서 각각 확인한다."""
  print("\n" + "=" * 70)
  print("[3] vis-viva 방정식: v^2 = mu*(2/r - 1/a)")
  print("=" * 70)
  a, e = 7500.0, 0.3
  r_perigee = a * (1 - e)
  r_apogee = a * (1 + e)

  rows = []
  for label, r, nu, ecc_anomaly in [("근지점", r_perigee, 0.0, 0.0), ("원지점", r_apogee, np.pi, np.pi)]:
    v_predicted = np.sqrt(EARTH_MU_KM3_S2 * (2 / r - 1 / a))
    _pos, vel = orbital_elements_to_state_vector(a, e, 0.0, 0.0, 0.0, nu, ecc_anomaly)
    v_actual = np.linalg.norm(vel)
    print(f"[{label}] r={r:.1f}km, vis-viva 예측 속력={v_predicted:.4f}km/s, 실제 계산 속력={v_actual:.4f}km/s")
    rows.append({"location": label, "r_km": r, "v_predicted": v_predicted, "v_actual": v_actual})
    assert abs(v_predicted - v_actual) < 1e-6, f"{label}에서 vis-viva 예측이 실제 속력과 일치해야 함"

  print(f"\n(근지점 속력({rows[0]['v_actual']:.2f}km/s)이 원지점 속력({rows[1]['v_actual']:.2f}km/s)보다")
  print(" 빠르다 — 01번에서 확인한 케플러 제2법칙과 같은 현상을, 이번엔 실제 속력 수치로 재확인한다.)")
  assert rows[0]["v_actual"] > rows[1]["v_actual"], "근지점 속력이 원지점 속력보다 빨라야 함"
  return rows


def demo_singular_case_circular_orbit_argp_undefined():
  """원궤도(e=0)에서 역변환을 시도하면 근점 편각(argp)이 물리적으로 무의미해진다는
  것을 감추지 않고 보여준다 — 이심률 벡터 자체가 거의 영벡터가 되기 때문이다."""
  print("\n" + "=" * 70)
  print("[4] 특이 케이스: 원궤도에서는 근점 편각이 정의되지 않는다")
  print("=" * 70)
  a, e, i, raan, argp_input = 7000.0, 0.0, np.radians(45.0), np.radians(30.0), np.radians(90.0)
  pos, vel = orbital_elements_to_state_vector(a, e, i, raan, argp_input, true_anomaly_rad=0.0, ecc_anomaly_rad=0.0)
  recovered = state_vector_to_orbital_elements(pos, vel)
  print(f"입력 근점 편각: {np.degrees(argp_input):.1f}도 (원궤도이므로 이 값 자체가 임의적)")
  print(f"역변환된 이심률: {recovered['eccentricity']:.2e} (거의 0)")
  print(f"역변환된 근점 편각: {np.degrees(recovered['argp_rad']):.1f}도 (0으로 강제됨 — 원궤도에서는 정의 불가)")
  print("\n(이심률 벡터 e_vec의 크기가 0에 가까우면 그 방향(근점 방향) 자체가 수치적으로")
  print(" 불안정해진다 — 이 스크립트는 이 경우 argp를 0으로 강제해 '정의되지 않음'을")
  print(" 명시적으로 표현하지, 임의의(노이즈에 좌우되는) 값을 그대로 반환하지 않는다.)")
  assert recovered["eccentricity"] < 1e-6, "원궤도 입력의 역변환 이심률은 0에 가까워야 함"
  return {"input_argp_deg": np.degrees(argp_input), "recovered_eccentricity": recovered["eccentricity"],
          "recovered_argp_deg": np.degrees(recovered["argp_rad"])}


def parse_args():
  parser = argparse.ArgumentParser(description="궤도요소<->상태벡터 변환과 보존량(비에너지/비각운동량) 검증")
  return parser.parse_args()


def main():
  parse_args()

  roundtrip_rows = demo_perifocal_to_eci_roundtrip()
  conservation_rows = demo_conserved_quantities_along_orbit()
  vis_viva_rows = demo_vis_viva_speed_prediction()
  singular_case = demo_singular_case_circular_orbit_argp_undefined()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  roundtrip_csv = os.path.join(results_dir, "orbital_elements_roundtrip.csv")
  with open(roundtrip_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["input_a", "input_e", "input_i_deg", "input_raan_deg", "recovered_a", "recovered_e",
                      "recovered_i_deg", "recovered_raan_deg"])
    for row in roundtrip_rows:
      writer.writerow([f"{row['input_a']:.2f}", f"{row['input_e']:.4f}", f"{row['input_i_deg']:.2f}",
                        f"{row['input_raan_deg']:.2f}", f"{row['recovered_a']:.4f}", f"{row['recovered_e']:.6f}",
                        f"{row['recovered_i_deg']:.4f}", f"{row['recovered_raan_deg']:.4f}"])
  print(f"\n[기록] 궤도요소 왕복 변환 결과 저장됨 → {roundtrip_csv}")

  conservation_csv = os.path.join(results_dir, "conserved_quantities.csv")
  with open(conservation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["time_fraction", "r_km", "v_km_s", "specific_energy", "angular_momentum"])
    for row in conservation_rows:
      writer.writerow([f"{row['time_fraction']:.3f}", f"{row['r_km']:.2f}", f"{row['v_km_s']:.6f}",
                        f"{row['specific_energy']:.8f}", f"{row['angular_momentum']:.6f}"])
  print(f"[기록] 보존량(비에너지/비각운동량) 결과 저장됨 → {conservation_csv}")

  vis_viva_csv = os.path.join(results_dir, "vis_viva_validation.csv")
  with open(vis_viva_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["location", "r_km", "v_predicted", "v_actual"])
    for row in vis_viva_rows:
      writer.writerow([row["location"], f"{row['r_km']:.2f}", f"{row['v_predicted']:.6f}", f"{row['v_actual']:.6f}"])
  print(f"[기록] vis-viva 검증 결과 저장됨 → {vis_viva_csv}")

  singular_csv = os.path.join(results_dir, "circular_orbit_singular_case.csv")
  with open(singular_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["input_argp_deg", "recovered_eccentricity", "recovered_argp_deg"])
    writer.writerow([f"{singular_case['input_argp_deg']:.2f}", f"{singular_case['recovered_eccentricity']:.2e}",
                      f"{singular_case['recovered_argp_deg']:.2f}"])
  print(f"[기록] 원궤도 특이 케이스 결과 저장됨 → {singular_csv}")


if __name__ == "__main__":
  main()
