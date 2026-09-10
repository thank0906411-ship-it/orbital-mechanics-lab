"""
J2 섭동 - 지구 편평도가 궤도면을 서서히 회전시킨다

01~06번은 지구를 완전한 구로 가정하는 순수 2체 문제만 다뤘다. 실제 지구는 적도가
약간 부풀어 있는 회전타원체(oblate spheroid)이고, 이 비대칭성(J2 항)이 위성 궤도에
장기적인 섭동을 일으킨다. 가장 두드러진 효과는 궤도면 자체가 서서히 회전하는
것(RAAN 세차, 근점편각 세차)이다. 이 스크립트는 이 세차 속도를 계산하고, 01번의
순간 궤도 위에 이 세차를 누적한 장기 시계열을 만들어 08번의 시각화가 "궤도면이
회전하는 모습"을 그릴 수 있게 한다.

핵심 개념 1: J2는 지구가 완전한 구가 아니라는 것을 나타내는 무차원 계수다
  지구의 중력장을 구면조화함수로 전개했을 때, 가장 큰 비구형 항이 J2(약 1.08263e-3)
  다. 이 항은 지구가 적도 방향으로 부풀어 있다는 것(적도 반지름이 극반지름보다 약
  21km 더 큼)을 반영한다. J2가 0이면(완전한 구) 01~06번에서 다룬 순수 케플러 궤도가
  그대로 유지되지만, J2가 있으면 궤도면 자체가 시간에 따라 회전한다.

핵심 개념 2: RAAN 세차 - 궤도면이 지구 자전축 주위로 돈다
  승교점 적경(RAAN, Ω)의 시간 변화율은
      dΩ/dt = -(3/2) * n * J2 * (R_E/p)^2 * cos(i)
  로 주어진다(n=평균운동, R_E=지구 적도반지름, p=반사통경=a(1-e^2), i=경사각). 이 식의
  핵심 특징은 부호다: 순행궤도(i<90도)에서는 cos(i)>0이라 dΩ/dt<0(서쪽으로 세차),
  역행궤도(i>90도)에서는 dΩ/dt>0(동쪽으로 세차)다. 태양동기궤도(SSO)는 이 세차 속도를
  1년에 정확히 360도(태양 주위 공전과 같은 속도)가 되도록 경사각을 골라 만든
  궤도다 — 이 스크립트는 SSO 조건에 가까운 경사각에서 실제로 세차 속도가 그 값에
  가까운지 확인한다.

핵심 개념 3: 근점편각 세차 - 타원의 장축 방향도 돈다
  근점편각(ω)의 시간 변화율은
      dω/dt = (3/4) * n * J2 * (R_E/p)^2 * (5*cos^2(i) - 1)
  이다. 이 식은 경사각에 따라 부호가 바뀌는 특이점을 갖는다: 5*cos^2(i)-1=0, 즉
  i ≈ 63.4도(또는 116.6도)에서 dω/dt=0이 된다 — 이것이 "임계경사각(critical
  inclination)"으로, 이 경사각에서는 근점편각이 세차하지 않아 근지점 위치가
  안정적으로 유지된다(몰니야 궤도가 이 경사각을 쓰는 이유). 이 스크립트는 이
  임계경사각 근처에서 실제로 세차 속도가 0에 가까워지는지 확인한다.

핵심 개념 4: 이 프로젝트의 섭동 모델은 "장기 평균 세차"만 다룬다(감추지 않음)
  실제 J2 섭동은 궤도 내에서도 단주기 진동(short-period variation)을 일으키지만,
  이 스크립트는 RAAN/근점편각이 시간에 따라 선형으로(위 공식의 순간 변화율로) 증가한다고
  가정하는 "평균 요소(mean element)" 근사만 구현한다. 01번의 순간 위치에 이 세차를
  얹어 장기 시계열을 만드는 것도, 각 시점마다 그 시점의 (세차된) RAAN을 새 궤도면
  회전에 적용하는 근사다 — 진짜 수치적분 기반 정밀 섭동 모델(SGP4 등)과는 다르다는
  것을 명시한다.

01번(케플러 전파)과의 관계: 이 스크립트는 01번의 propagate_orbit, mean_motion을
재사용해 매 시점의 궤도면 내 위치를 구하고, 거기에 이 스크립트가 계산한 시간에 따라
누적된 RAAN/근점편각 세차를 적용한다.
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
_spec = importlib.util.spec_from_file_location("kepler_module", _KEPLER_PATH)
kepler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kepler)

_orbit_math_spec = importlib.util.spec_from_file_location("orbit_math", os.path.join(_ROOT_DIR, "orbit_math.py"))
orbit_math = importlib.util.module_from_spec(_orbit_math_spec)
_orbit_math_spec.loader.exec_module(orbit_math)
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM
rotation_matrix_x = orbit_math.rotation_matrix_x
rotation_matrix_z = orbit_math.rotation_matrix_z

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2
J2 = 1.08263e-3  # 지구 중력장의 2차 대역조화계수 (무차원)


def semi_latus_rectum(semi_major_axis_km, eccentricity):
  """p = a*(1 - e^2)"""
  return semi_major_axis_km * (1 - eccentricity ** 2)


def raan_precession_rate(semi_major_axis_km, eccentricity, inclination_rad, mu=EARTH_MU_KM3_S2):
  """dOmega/dt = -(3/2)*n*J2*(R_E/p)^2*cos(i) (rad/s)."""
  n = kepler.mean_motion(semi_major_axis_km, mu)
  p = semi_latus_rectum(semi_major_axis_km, eccentricity)
  return -1.5 * n * J2 * (EARTH_RADIUS_KM / p) ** 2 * np.cos(inclination_rad)


def argp_precession_rate(semi_major_axis_km, eccentricity, inclination_rad, mu=EARTH_MU_KM3_S2):
  """domega/dt = (3/4)*n*J2*(R_E/p)^2*(5*cos^2(i) - 1) (rad/s)."""
  n = kepler.mean_motion(semi_major_axis_km, mu)
  p = semi_latus_rectum(semi_major_axis_km, eccentricity)
  return 0.75 * n * J2 * (EARTH_RADIUS_KM / p) ** 2 * (5 * np.cos(inclination_rad) ** 2 - 1)


def propagate_with_j2_precession(semi_major_axis_km, eccentricity, inclination_rad,
                                  raan0_rad, argp0_rad, mean_anomaly0_rad, time_sec, mu=EARTH_MU_KM3_S2):
  """01번의 순간 궤도면 위치 + 시간에 선형 누적된 RAAN/근점편각 세차를 적용한 ECI 위치.
  반환: dict(state(01번 propagate_orbit 결과), raan_rad, argp_rad, position_eci)."""
  state = kepler.propagate_orbit(semi_major_axis_km, eccentricity, mean_anomaly0_rad, time_sec, mu)
  raan_rate = raan_precession_rate(semi_major_axis_km, eccentricity, inclination_rad, mu)
  argp_rate = argp_precession_rate(semi_major_axis_km, eccentricity, inclination_rad, mu)
  raan = raan0_rad + raan_rate * time_sec
  argp = argp0_rad + argp_rate * time_sec

  x_p, y_p = state["x_p"], state["y_p"]
  position_perifocal = np.array([x_p, y_p, 0.0])

  rotation = rotation_matrix_z(raan) @ rotation_matrix_x(inclination_rad) @ rotation_matrix_z(argp)
  position_eci = rotation @ position_perifocal
  return {"state": state, "raan_rad": raan, "argp_rad": argp, "position_eci": position_eci}


def demo_raan_precession_direction_depends_on_inclination():
  """순행궤도(i<90도)는 RAAN이 서쪽(음의 방향)으로, 역행궤도(i>90도)는 동쪽(양의
  방향)으로 세차한다는 것을 확인한다."""
  print("=" * 70)
  print("[1] RAAN 세차 방향: 순행궤도 vs 역행궤도")
  print("=" * 70)
  a, e = 7000.0, 0.01
  print(f"반장축={a}km, 이심률={e}\n")
  print(f"  {'경사각(도)':>12}{'RAAN 세차율(도/일)':>22}")

  rows = []
  for i_deg in [0.0, 30.0, 51.6, 90.0, 98.0, 120.0, 180.0]:
    rate_rad_s = raan_precession_rate(a, e, np.radians(i_deg))
    rate_deg_day = np.degrees(rate_rad_s) * 86400
    print(f"  {i_deg:>12.1f}{rate_deg_day:>22.4f}")
    rows.append({"inclination_deg": i_deg, "raan_rate_deg_per_day": rate_deg_day})

  prograde_rate = next(r["raan_rate_deg_per_day"] for r in rows if r["inclination_deg"] == 51.6)
  retrograde_rate = next(r["raan_rate_deg_per_day"] for r in rows if r["inclination_deg"] == 98.0)
  polar_rate = next(r["raan_rate_deg_per_day"] for r in rows if r["inclination_deg"] == 90.0)

  print(f"\n(순행궤도(51.6도)는 서쪽으로({prograde_rate:.2f}도/일), 태양동기 근처(98도)는")
  print(f" 동쪽으로({retrograde_rate:.2f}도/일) 세차한다. 극궤도(90도)는 세차율이 정확히")
  print(f" 0({polar_rate:.6f}도/일)이다 — cos(90도)=0이기 때문이다.)")

  assert prograde_rate < 0, "순행궤도(i<90도)는 RAAN이 서쪽(음의 방향)으로 세차해야 함"
  assert retrograde_rate > 0, "역행궤도(i>90도)는 RAAN이 동쪽(양의 방향)으로 세차해야 함"
  assert abs(polar_rate) < 1e-9, "극궤도(i=90도)는 RAAN 세차율이 0이어야 함(cos(90도)=0)"
  return rows


def demo_critical_inclination_zero_argp_precession():
  """임계경사각(약 63.4도)에서 근점편각 세차율이 0에 가까워지는지 확인한다 —
  몰니야 궤도가 이 경사각을 쓰는 이유다."""
  print("\n" + "=" * 70)
  print("[2] 임계경사각(약 63.4도): 근점편각이 세차하지 않는 지점")
  print("=" * 70)
  a, e = 26600.0, 0.7  # 몰니야류 고이심률 궤도 규모
  critical_inclination_deg = np.degrees(np.arccos(1 / np.sqrt(5)))
  print(f"이론적 임계경사각: {critical_inclination_deg:.4f}도 (5*cos^2(i)-1=0의 해)\n")
  print(f"  {'경사각(도)':>12}{'근점편각 세차율(도/일)':>24}")

  rows = []
  for i_deg in [0.0, 30.0, 50.0, critical_inclination_deg, 80.0, 90.0]:
    rate_rad_s = argp_precession_rate(a, e, np.radians(i_deg))
    rate_deg_day = np.degrees(rate_rad_s) * 86400
    print(f"  {i_deg:>12.4f}{rate_deg_day:>24.6f}")
    rows.append({"inclination_deg": i_deg, "argp_rate_deg_per_day": rate_deg_day})

  critical_rate = rows[3]["argp_rate_deg_per_day"]
  print(f"\n임계경사각에서의 세차율: {critical_rate:.8f}도/일 (이론적으로 정확히 0이어야 함)")
  assert abs(critical_rate) < 1e-6, "임계경사각(약 63.4도)에서 근점편각 세차율은 0에 가까워야 함"
  print("\n(경사각이 임계경사각을 지나면서 세차율의 부호가 바뀐다 — 몰니야 궤도가 이 경사각을")
  print(" 쓰는 이유는, 근점편각이 세차하지 않아 원지점(가장 느리게 지나가는 지점, 러시아")
  print(" 상공)의 위치가 장기적으로 안정되게 유지되기 때문이다.)")
  return rows


def demo_long_term_raan_drift_time_series():
  """01번의 순간 위치에 RAAN 세차를 누적한 장기 시계열을 생성한다 — 08번이
  '궤도면이 서서히 회전하는' 모습을 시각화할 수 있게 하는 핵심 데이터."""
  print("\n" + "=" * 70)
  print("[3] 장기 RAAN 드리프트 시계열 생성 (08번 시각화용)")
  print("=" * 70)
  a, e, i = 7000.0, 0.01, np.radians(97.4)  # 태양동기궤도에 가까운 경사각
  raan_rate = raan_precession_rate(a, e, i)
  raan_rate_deg_day = np.degrees(raan_rate) * 86400
  days = 30
  print(f"반장축={a}km, 경사각={np.degrees(i):.1f}도, RAAN 세차율={raan_rate_deg_day:.4f}도/일")
  print(f"관찰 기간: {days}일\n")

  num_orbits_per_day = 86400 / (2 * np.pi / kepler.mean_motion(a))
  print(f"하루 궤도 수: 약 {num_orbits_per_day:.1f}바퀴")

  sample_days = np.linspace(0, days, 7)
  rows = []
  print(f"  {'경과일':>8}{'RAAN(도)':>14}")
  for day in sample_days:
    t = day * 86400
    result = propagate_with_j2_precession(a, e, i, raan0_rad=0.0, argp0_rad=0.0, mean_anomaly0_rad=0.0, time_sec=t)
    raan_deg = np.degrees(result["raan_rad"]) % 360
    print(f"  {day:>8.1f}{raan_deg:>14.2f}")
    rows.append({"day": day, "t_sec": t, "raan_deg": raan_deg,
                 "position_eci_x_km": result["position_eci"][0], "position_eci_y_km": result["position_eci"][1],
                 "position_eci_z_km": result["position_eci"][2]})

  print(f"\n{days}일 동안 누적된 RAAN 드리프트: 약 {abs(raan_rate_deg_day * days):.2f}도")
  assert abs(raan_rate_deg_day) > 0.01, "이 경사각(97.4도, SSO 근처)에서는 RAAN 세차율이 뚜렷해야 함"
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="J2 섭동에 의한 RAAN/근점편각 세차 계산")
  parser.add_argument("--semi-major-axis-km", type=float, default=7000.0, help="반장축(km), 기본값: 7000km")
  parser.add_argument("--inclination-deg", type=float, default=97.4, help="경사각(도), 기본값: 97.4(SSO 근처)")
  return parser.parse_args()


def main():
  args = parse_args()

  raan_direction_rows = demo_raan_precession_direction_depends_on_inclination()
  critical_inclination_rows = demo_critical_inclination_zero_argp_precession()
  drift_rows = demo_long_term_raan_drift_time_series()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  raan_csv = os.path.join(results_dir, "j2_raan_precession_by_inclination.csv")
  with open(raan_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["inclination_deg", "raan_rate_deg_per_day"])
    for row in raan_direction_rows:
      writer.writerow([row["inclination_deg"], f"{row['raan_rate_deg_per_day']:.6f}"])
  print(f"\n[기록] 경사각별 RAAN 세차율 저장됨 → {raan_csv}")

  critical_csv = os.path.join(results_dir, "j2_critical_inclination_argp.csv")
  with open(critical_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["inclination_deg", "argp_rate_deg_per_day"])
    for row in critical_inclination_rows:
      writer.writerow([f"{row['inclination_deg']:.4f}", f"{row['argp_rate_deg_per_day']:.8f}"])
  print(f"[기록] 임계경사각 근점편각 세차율 저장됨 → {critical_csv}")

  drift_csv = os.path.join(results_dir, "j2_long_term_raan_drift.csv")
  with open(drift_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["day", "t_sec", "raan_deg", "position_eci_x_km", "position_eci_y_km", "position_eci_z_km"])
    for row in drift_rows:
      writer.writerow([f"{row['day']:.2f}", f"{row['t_sec']:.1f}", f"{row['raan_deg']:.4f}",
                        f"{row['position_eci_x_km']:.4f}", f"{row['position_eci_y_km']:.4f}", f"{row['position_eci_z_km']:.4f}"])
  print(f"[기록] 장기 RAAN 드리프트 시계열 저장됨 → {drift_csv}")

  print(f"\n(참고: --semi-major-axis-km={args.semi_major_axis_km}, --inclination-deg={args.inclination_deg}는 "
        f"향후 CLI 파라미터화에 대비한 자리다.)")


if __name__ == "__main__":
  main()
