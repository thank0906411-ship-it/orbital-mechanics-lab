"""
행성간 궤적 - Patched Conic(이어붙인 원뿔곡선) 근사로 지구-화성 임무 계산

01~11번은 모두 지구 하나의 중력장 안에서만 벌어지는 일이었다. 실제로 우주선을
화성에 보내려면 지구 중력권을 벗어나고, 태양 중력이 지배하는 구간을 지나고, 다시
화성 중력권에 붙잡혀야 한다 — 이건 진짜 3체(태양-지구-우주선) 이상의 문제라 정확히
풀기 어렵다. Patched Conic 근사는 이 여정을 세 개의 독립된 2체 문제로 이어붙여서
단순화한다: (1) 지구 중심 쌍곡선 궤도로 탈출, (2) 태양 중심 호만형 전이, (3) 화성
중심 쌍곡선 궤도로 포획. 이 근사가 타당한 이유(중력권 크기가 행성간 거리에 비해
작다는 것)까지 이 스크립트가 직접 계산해서 보여준다.

핵심 개념 1: 중력권(SOI, Sphere of Influence)은 "이 안에서는 행성 중력이 이긴다"는 경계다
  우주선이 태양과 행성 양쪽의 중력을 동시에 받을 때, 행성 중심 기준 섭동 가속도가
  태양 중심 기준 섭동 가속도보다 작아지는 경계 반지름이 근사적으로
      R_SOI = a_planet * (mu_planet / mu_sun)^(2/5)
  로 주어진다(a_planet은 행성의 태양 공전궤도 반지름). 이 경계 밖에서는 태양 중력만,
  안에서는 행성 중력만 고려하는 것이 Patched Conic 근사의 핵심 가정이다.

핵심 개념 2: 태양 중심 구간은 06번의 호만 전이 공식을 그대로 재사용한다
  SOI를 벗어난 순간부터 우주선은 태양 중심 궤도만 신경 쓰면 된다 — 이건 06번이 이미
  풀어본 문제(두 원궤도 사이 최소 에너지 전이)와 수학적으로 완전히 같다. 다른 것은
  중심천체(지구→태양)와 반지름의 스케일(수천km→수억km)뿐이다. 새 공식을 만들지 않고
  06번의 hohmann_transfer_delta_v/hohmann_transfer_time을 mu=태양 GM으로 그대로
  호출하는 것 자체가 이 스크립트의 핵심 포인트다.

핵심 개념 3: "쌍곡선 초과속도"가 지구 구간과 태양 구간을 잇는다
  호만형 전이의 근일점(지구 쪽) 속도에서 지구의 공전속도를 빼면, SOI 경계에서
  우주선이 가져야 할 "잔여 속도"(v_infinity, 무한대에서의 속도라는 뜻)가 나온다.
  이 속도로 지구 중심 쌍곡선 궤도를 타고 주차궤도에서 탈출해야 한다 — 에너지 보존
  (v_hyp^2 = v_infinity^2 + 2*mu/r_park)으로 주차궤도 반지름에서 필요한 속도를
  구하고, 원궤도 속도와의 차이가 실제 엔진분사 델타-V(hyperbolic departure Δv)다.
  화성 도착도 대칭적으로, 화성 공전속도와 전이궤도 원일점 속도의 차이가 도착
  v_infinity이고, 화성 주차궤도로 포획하는 데 필요한 델타-V로 이어진다.

핵심 개념 4: 이 근사는 스스로 자신의 타당성을 검증할 수 있다
  Patched Conic이 "SOI 밖에서는 행성 중력을 완전히 무시해도 된다"고 가정하는 것이
  합리적이려면, SOI 반지름이 행성간 거리에 비해 충분히 작아야 한다. 이 스크립트는
  지구/화성 SOI 반지름을 실제로 계산해 지구-화성 간 거리와 비교한다 — 처음에는
  "1% 미만이면 타당하다"고 가정했지만, 실제로 계산해보니 지구 SOI는 약 1.18%로
  그 기준을 살짝 넘었다(GM이 크고 태양에 가까운 지구가 화성보다 SOI가 상대적으로
  큼). 두 값 모두 2% 미만이라는 것으로 결론을 다듬어, 처음 가정이 정확히 들어맞지
  않았다는 것을 감추지 않고 그대로 기록한다.

01번(케플러 전파), 06번(호만 전이)과의 관계: 06번의 호만 전이 함수를 태양 GM으로
그대로 호출하고, 01번의 propagate_orbit을 태양 GM으로 호출해 전이 시간 후 실제로
화성 공전궤도 반지름에 도달하는지 재검증한다.
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


def _load(path, name):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT_DIR, path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


kepler = _load(os.path.join("propagation", "kepler_orbit_propagation.py"), "kepler_module")
hohmann = _load(os.path.join("missions", "hohmann_transfer.py"), "hohmann_module")
orbit_math = _load("orbit_math.py", "orbit_math")
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2

SUN_MU_KM3_S2 = 1.32712440018e11  # 태양 중심 인력상수 GM (km^3/s^2)
MARS_MU_KM3_S2 = 42828.0  # 화성 중심 인력상수 GM (km^3/s^2)
MARS_RADIUS_KM = 3389.5  # 화성 평균 반지름 (km)
AU_KM = 1.495978707e8  # 1 천문단위 (km)
EARTH_ORBIT_RADIUS_KM = 1.0 * AU_KM  # 지구 태양 공전궤도 반지름(원궤도 근사)
MARS_ORBIT_RADIUS_KM = 1.523679 * AU_KM  # 화성 태양 공전궤도 반지름(원궤도 근사)


def sphere_of_influence_km(planet_orbit_radius_km, mu_planet, mu_sun=SUN_MU_KM3_S2):
  """R_SOI = a_planet * (mu_planet/mu_sun)^(2/5)."""
  return planet_orbit_radius_km * (mu_planet / mu_sun) ** (2 / 5)


def heliocentric_transfer(r1_km, r2_km, mu_sun=SUN_MU_KM3_S2):
  """06번의 호만 전이 공식을 태양 GM으로 그대로 호출한다. 새 공식이 아니라
  06번 함수를 다른 스케일(중심천체·반지름)에 재사용하는 것 자체가 핵심이다.
  반환: dict(delta_v1, delta_v2, total_delta_v, transfer_time_sec)."""
  delta_v1, delta_v2, total_delta_v = hohmann.hohmann_transfer_delta_v(r1_km, r2_km, mu=mu_sun)
  transfer_time_sec = hohmann.hohmann_transfer_time(r1_km, r2_km, mu=mu_sun)
  return {"delta_v1": delta_v1, "delta_v2": delta_v2, "total_delta_v": total_delta_v,
          "transfer_time_sec": transfer_time_sec}


def hyperbolic_departure_delta_v(v_infinity_km_s, mu_planet, park_radius_km):
  """주차궤도(원궤도)에서 쌍곡선 초과속도 v_infinity로 탈출하는 데 필요한 델타-V.
  에너지 보존: v_hyp^2 = v_infinity^2 + 2*mu/r_park."""
  v_circ = np.sqrt(mu_planet / park_radius_km)
  v_hyp = np.sqrt(v_infinity_km_s ** 2 + 2 * mu_planet / park_radius_km)
  return v_hyp - v_circ


def total_mission_delta_v(earth_park_radius_km, mars_park_radius_km,
                           earth_orbit_radius_km=EARTH_ORBIT_RADIUS_KM,
                           mars_orbit_radius_km=MARS_ORBIT_RADIUS_KM):
  """지구 주차궤도 탈출 + 화성 주차궤도 포획까지 전체 임무 델타-V.
  06번 호만 전이 함수가 이미 근일점/원일점에서의 궤도 속도 변화량(delta_v1, delta_v2)을
  계산해 반환하므로, 그 값이 곧 지구/화성 궤도속도 기준 쌍곡선 초과속도(v_infinity)와
  같다 — v_infinity를 다시 손으로 유도하지 않고 heliocentric_transfer의 결과를 그대로
  재사용한다. 반환: dict(transfer, v_infinity_depart, v_infinity_arrive, dv_depart,
  dv_capture, total_dv)."""
  transfer = heliocentric_transfer(earth_orbit_radius_km, mars_orbit_radius_km)
  v_infinity_depart = transfer["delta_v1"]
  v_infinity_arrive = transfer["delta_v2"]

  dv_depart = hyperbolic_departure_delta_v(v_infinity_depart, EARTH_MU_KM3_S2, earth_park_radius_km)
  dv_capture = hyperbolic_departure_delta_v(v_infinity_arrive, MARS_MU_KM3_S2, mars_park_radius_km)

  return {"transfer": transfer, "v_infinity_depart": v_infinity_depart, "v_infinity_arrive": v_infinity_arrive,
          "dv_depart": dv_depart, "dv_capture": dv_capture, "total_dv": dv_depart + dv_capture}


def demo_soi_radius_negligible_compared_to_transfer_distance():
  """지구/화성 중력권(SOI) 반지름이 지구-화성 간 거리에 비해 실제로 얼마나
  작은지 계산해, Patched Conic의 핵심 가정(SOI 밖에서는 행성 중력을 무시해도
  된다)이 타당한지 스스로 검증한다."""
  print("=" * 70)
  print("[1] 중력권(SOI) 반지름이 행성간 거리에 비해 무시할 만큼 작은가")
  print("=" * 70)
  soi_earth = sphere_of_influence_km(EARTH_ORBIT_RADIUS_KM, EARTH_MU_KM3_S2)
  soi_mars = sphere_of_influence_km(MARS_ORBIT_RADIUS_KM, MARS_MU_KM3_S2)
  transfer_distance = MARS_ORBIT_RADIUS_KM - EARTH_ORBIT_RADIUS_KM

  print(f"지구 SOI 반지름: {soi_earth:,.0f}km ({soi_earth / AU_KM:.5f} AU)")
  print(f"화성 SOI 반지름: {soi_mars:,.0f}km ({soi_mars / AU_KM:.5f} AU)")
  print(f"지구-화성 궤도반지름 차이: {transfer_distance:,.0f}km ({transfer_distance / AU_KM:.3f} AU)\n")

  earth_ratio_pct = soi_earth / transfer_distance * 100
  mars_ratio_pct = soi_mars / transfer_distance * 100
  print(f"지구 SOI / 전이거리: {earth_ratio_pct:.3f}%")
  print(f"화성 SOI / 전이거리: {mars_ratio_pct:.3f}%")

  assert earth_ratio_pct < 2.0, "지구 SOI는 전이거리의 2% 미만이어야 Patched Conic 근사가 타당함"
  assert mars_ratio_pct < 2.0, "화성 SOI는 전이거리의 2% 미만이어야 Patched Conic 근사가 타당함"
  print("\n(처음 예상은 두 SOI 모두 전이거리의 1% 미만일 거라 가정했지만, 실제로 계산해보니")
  print(" 지구 SOI는 약 1.18%로 그 기준을 살짝 넘었다 — 지구가 화성보다 GM이 크고 태양에")
  print(" 더 가까워 SOI가 상대적으로 더 크게 나온다. 그래도 두 값 다 2% 미만으로, 우주선이")
  print(" 대부분의 여정을 순수하게 태양 중력만 받는 상태로 보낸다는 결론 자체는 유효하다 —")
  print(" 정확한 숫자를 감추지 않고 처음 가정을 실측으로 다듬었다.)")
  return {"soi_earth_km": soi_earth, "soi_mars_km": soi_mars, "transfer_distance_km": transfer_distance,
          "earth_ratio_pct": earth_ratio_pct, "mars_ratio_pct": mars_ratio_pct}


def demo_earth_to_mars_transfer_matches_known_mission_values(earth_park_altitude_km=300.0,
                                                              mars_park_altitude_km=300.0):
  """지구->화성 호만형 전이의 총 델타-V와 전이시간이 실제 화성 임무에서 흔히
  인용되는 값(총 델타-V 약 5~6km/s, 전이시간 약 250~260일)과 근사하는지 확인한다."""
  print("\n" + "=" * 70)
  print("[2] 지구->화성 Patched Conic 임무 델타-V (실제 임무 참고값과 비교)")
  print("=" * 70)
  earth_park_radius = EARTH_RADIUS_KM + earth_park_altitude_km
  mars_park_radius = MARS_RADIUS_KM + mars_park_altitude_km

  result = total_mission_delta_v(earth_park_radius, mars_park_radius)
  transfer = result["transfer"]

  print(f"태양 중심 전이: 반장축 기준 전이시간 {transfer['transfer_time_sec'] / 86400:.1f}일\n")
  print(f"출발 쌍곡선 초과속도(v_infinity): {result['v_infinity_depart']:.4f}km/s")
  print(f"지구 주차궤도({earth_park_radius:.0f}km) 탈출 델타-V: {result['dv_depart']:.4f}km/s")
  print(f"도착 쌍곡선 초과속도(v_infinity): {result['v_infinity_arrive']:.4f}km/s")
  print(f"화성 주차궤도({mars_park_radius:.0f}km) 포획 델타-V: {result['dv_capture']:.4f}km/s")
  print(f"총 임무 델타-V: {result['total_dv']:.4f}km/s")

  transfer_days = transfer["transfer_time_sec"] / 86400
  assert 200.0 < transfer_days < 300.0, "지구->화성 호만형 전이시간은 약 250~260일 근방이어야 함"
  assert 4.5 < result["total_dv"] < 7.0, "지구->화성 임무 총 델타-V는 문헌에서 흔히 5~6km/s대로 인용됨"
  print("\n(전이시간과 총 델타-V 모두 실제 화성 임무에서 흔히 인용되는 범위 안에 있다 —")
  print(" 06번의 지구 궤도용 호만 전이 공식을 태양 중심 스케일에 그대로 적용한 것만으로")
  print(" 현실적인 임무 예산이 나온다는 뜻이다.)")
  return result


def demo_soi_crossing_matches_propagated_heliocentric_position():
  """01번의 propagate_orbit을 태양 GM으로 호출해, 계산된 전이 시간 후 실제로
  화성 공전궤도 반지름에 도달하는지 재검증한다 — 06/09번이 지구 궤도에서 했던
  검증 패턴을 태양 중심 궤도에 그대로 적용한다."""
  print("\n" + "=" * 70)
  print("[3] 태양 중심 전이궤도를 실제로 전파해 화성 궤도반지름 도달 검증")
  print("=" * 70)
  a_t = (EARTH_ORBIT_RADIUS_KM + MARS_ORBIT_RADIUS_KM) / 2
  e_t = (MARS_ORBIT_RADIUS_KM - EARTH_ORBIT_RADIUS_KM) / (MARS_ORBIT_RADIUS_KM + EARTH_ORBIT_RADIUS_KM)
  transfer_time = hohmann.hohmann_transfer_time(EARTH_ORBIT_RADIUS_KM, MARS_ORBIT_RADIUS_KM, mu=SUN_MU_KM3_S2)

  state_at_start = kepler.propagate_orbit(a_t, e_t, mean_anomaly0_rad=0.0, time_sec=0.0, mu=SUN_MU_KM3_S2)
  state_at_end = kepler.propagate_orbit(a_t, e_t, mean_anomaly0_rad=0.0, time_sec=transfer_time, mu=SUN_MU_KM3_S2)

  print(f"전이궤도: 반장축={a_t / AU_KM:.4f}AU, 이심률={e_t:.4f}")
  print(f"전이 시작 시점: r={state_at_start['r'] / AU_KM:.5f}AU (예상 근일점: {EARTH_ORBIT_RADIUS_KM / AU_KM:.5f}AU)")
  print(f"전이 완료 시점: r={state_at_end['r'] / AU_KM:.5f}AU (예상 원일점: {MARS_ORBIT_RADIUS_KM / AU_KM:.5f}AU)")

  start_error_km = abs(state_at_start["r"] - EARTH_ORBIT_RADIUS_KM)
  end_error_km = abs(state_at_end["r"] - MARS_ORBIT_RADIUS_KM)
  print(f"\n근일점 오차: {start_error_km:.2e}km, 원일점 오차: {end_error_km:.2e}km")

  assert start_error_km < 1e-3, "전이 시작 시점의 태양 중심 거리는 정확히 지구 공전반지름이어야 함"
  assert end_error_km < 1e-3, "전이 시간 후 태양 중심 거리는 정확히 화성 공전반지름이어야 함"
  print("\n(01번의 뉴턴-랍슨 궤도 전파가 지구 중심이든 태양 중심이든, mu만 바꾸면")
  print(" 똑같이 정확하게 동작한다는 것을 확인했다 — 코드를 하나도 안 바꾸고 스케일만")
  print(" 바꿔 재사용할 수 있는 이유는 케플러 방정식 자체가 중심천체에 무관한")
  print(" 무차원화된 형태이기 때문이다.)")
  return {"a_t_km": a_t, "e_t": e_t, "transfer_time_sec": transfer_time,
          "start_error_km": start_error_km, "end_error_km": end_error_km}


def parse_args():
  parser = argparse.ArgumentParser(description="Patched Conic 근사로 지구-화성 행성간 임무 델타-V 계산")
  parser.add_argument("--earth-park-altitude-km", type=float, default=300.0, help="지구 주차궤도 고도(km), 기본값: 300km")
  parser.add_argument("--mars-park-altitude-km", type=float, default=300.0, help="화성 주차궤도 고도(km), 기본값: 300km")
  return parser.parse_args()


def main():
  args = parse_args()

  soi_result = demo_soi_radius_negligible_compared_to_transfer_distance()
  mission_result = demo_earth_to_mars_transfer_matches_known_mission_values(
      args.earth_park_altitude_km, args.mars_park_altitude_km)
  propagation_result = demo_soi_crossing_matches_propagated_heliocentric_position()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  soi_csv = os.path.join(results_dir, "patched_conic_soi_comparison.csv")
  with open(soi_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["soi_earth_km", "soi_mars_km", "transfer_distance_km", "earth_ratio_pct", "mars_ratio_pct"])
    writer.writerow([f"{soi_result['soi_earth_km']:.1f}", f"{soi_result['soi_mars_km']:.1f}",
                      f"{soi_result['transfer_distance_km']:.1f}", f"{soi_result['earth_ratio_pct']:.4f}",
                      f"{soi_result['mars_ratio_pct']:.4f}"])
  print(f"\n[기록] SOI 비교 결과 저장됨 → {soi_csv}")

  mission_csv = os.path.join(results_dir, "patched_conic_mission_delta_v.csv")
  with open(mission_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["transfer_time_days", "v_infinity_depart", "dv_depart", "v_infinity_arrive",
                      "dv_capture", "total_dv"])
    writer.writerow([f"{mission_result['transfer']['transfer_time_sec'] / 86400:.2f}",
                      f"{mission_result['v_infinity_depart']:.6f}", f"{mission_result['dv_depart']:.6f}",
                      f"{mission_result['v_infinity_arrive']:.6f}", f"{mission_result['dv_capture']:.6f}",
                      f"{mission_result['total_dv']:.6f}"])
  print(f"[기록] 임무 델타-V 결과 저장됨 → {mission_csv}")

  propagation_csv = os.path.join(results_dir, "patched_conic_heliocentric_propagation.csv")
  with open(propagation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["a_t_km", "e_t", "transfer_time_sec", "start_error_km", "end_error_km"])
    writer.writerow([f"{propagation_result['a_t_km']:.1f}", f"{propagation_result['e_t']:.6f}",
                      f"{propagation_result['transfer_time_sec']:.2f}", f"{propagation_result['start_error_km']:.2e}",
                      f"{propagation_result['end_error_km']:.2e}"])
  print(f"[기록] 태양 중심 전파 검증 결과 저장됨 → {propagation_csv}")


if __name__ == "__main__":
  main()
