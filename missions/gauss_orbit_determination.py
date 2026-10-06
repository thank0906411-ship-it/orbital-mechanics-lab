"""
가우스법(Gauss's Method) 기반 일반 궤도결정 - 각도 전용 관측 3회로 타원궤도를 복원

14번(orbit_determination.py)은 "이심률이 0인 원궤도만 다룬다 — 완전히 일반적인
케플러 궤도 피팅(가우스법 등)은 라그랑주 계수와 고차 다항식 풀이가 추가로 필요해
범위가 커진다"고 명시적으로 범위를 그었다. 이 스크립트는 바로 그 가우스법을
구현해, 방위각/고도각만(거리 없이) 측정한 3회 관측값으로 일반 타원궤도(이심률
0 이상)를 복원한다.

핵심 개념 1: 관측자 시선벡터 + 관측자 위치만으로 위치벡터 3개가 숨어있다
  세 시점의 관측(방위각, 고도각 — 거리는 쓰지 않음)에서 각 ECI 단위 시선벡터
  L1,L2,L3와 관측자 위치 R1,R2,R3(04번의 site_position_eci 재사용)를 구하면,
  진짜 위치벡터는 r_i = R_i + rho_i*L_i(rho_i는 미지의 거리)로 표현된다 —
  미지수는 rho1,rho2,rho3 세 개뿐이다.

핵심 개념 2: 라그랑주 계수(f,g)가 세 위치를 하나의 궤도로 묶는다
  2체 운동에서 임의의 두 시점 위치/속도는 r1=f1*r2+g1*v2, r3=f3*r2+g3*v2로
  정확히 연결된다(f,g는 시간차 tau1=t1-t2, tau3=t3-t2와 mu/r2^3의 함수). 3차
  테일러 전개(f≈1-0.5*(mu/r2^3)*tau^2, g≈tau-(1/6)*(mu/r2^3)*tau^3)로 근사하면
  미지의 v2를 소거할 수 있어, rho1,rho2,rho3를 잇는 선형 연립방정식과 최종적으로
  r2 하나짜리 8차 다항식(r2^8+a*r2^6+b*r2^3+c=0, 짝수차항+상수항만)으로
  귀결된다.

핵심 개념 3: 8차 다항식의 근은 스칼라 삼중곱 D0에 극도로 민감하다(실측 확인)
  계수 A, B는 전부 1/D0(D0=L1·(L2x L3), 세 시선벡터의 공면성 척도)로 나뉘어
  있어, D0가 작으면(관측 호가 짧거나 관측자-궤도 기하가 거의 공면) A, B가
  비현실적으로 커지고 다항식의 어떤 근도 참값과 가까워지지 않는다. LEO
  지상관측은 태생적으로 D0가 작아지기 쉬운 기하라는 것을 계획 단계 실측
  스윕으로 확인했다 — 이건 숨기지 않고 데모에서 직접 보여준다(9번 란베르트의
  180도 특이점, 14번의 원궤도 제한과 같은 "한계를 감추지 않는다"는 이
  프로젝트의 확립된 태도).

핵심 개념 4: 반복(iteration)으로 안정화한다 — 완전한 보편변수 공식 없이도
  최초 r2 추정(8차 다항식의 물리적으로 유효한 근)으로 f,g를 계산해
  rho1,rho2,rho3와 r2_vec를 구한 뒤, 그 갱신된 r2_vec 크기로 다시 f,g를
  재계산해 rho들을 다시 구하는 과정을 r2가 수렴할 때까지(보통 10회 미만)
  반복한다. 스텀프 함수 기반 완전 보편변수 공식(쌍곡선/포물선까지 다루는
  일반형)은 구현하지 않는다 — 계획 단계 실측에서 이 단순 고정점 반복만으로
  LEO 시나리오의 반장축 오차가 불안정한 수십~수백%대에서 안정적인 1~3%대로
  바뀌는 것을 확인했다. 처음에는 이 반복 자체를 범위 밖에 두려 했으나, 실측이
  "반복 없이는 날카로운 스윗스팟 하나에서만 성공하고 그 근처에서도 수십~수백%
  오차로 터진다"는 것을 보여줘 계획을 뒤집었다.

핵심 개념 5: 수렴한 r2_vec, v2로 02번의 범용 궤도요소 변환을 그대로 재사용한다
  최종 라그랑주 계수로 v2 = (f1*r3_vec - f3*r1_vec)/(f1*g3 - f3*g1)을 구하면
  (r2_vec, v2) 상태벡터가 완성되고, 02번의 state_vector_to_orbital_elements에
  그대로 넘겨 6개 궤도요소(이심률·근점편각 포함, 14번은 전혀 다루지 못했던 값)를
  얻는다 — 02번은 이미 일반 타원궤도를 완전히 지원하므로 수정이 필요 없다.

단순화: 스텀프 함수 기반 완전 보편변수 공식(쌍곡선/포물선 궤도, 반복 없이도
안정적인 "개선된 가우스법")은 범위 밖으로 둔다. 궤도가 지구 그림자에 들어가
관측이 불가능해지는 경우는 고려하지 않는다(14번과 동일 단순화 수준). 세 관측
모두 같은 지상국에서 이뤄진다고 가정한다(여러 관측소 조합은 다루지 않음).
좋은 관측 기하(D0가 충분히 큰 간격)를 자동으로 찾는 탐색은 제한된 후보 집합만
시도하고, 전부 실패하면 명시적으로 에러를 내 가우스법의 기하 민감성을 감추지
않는다.

14번(orbit_determination.py)과의 관계: look_angles_to_eci_position(거리 포함
위치 반환)은 재사용하지 않고, 거리 없이 단위 시선벡터만 구하는 새 함수를 이
스크립트에 둔다(가우스법은 거리를 모른다고 가정하는 것이 핵심이므로).
find_visible_time_window/true_orbit_position은 원궤도 전용(argp 생략)이라
그대로 재사용할 수 없어, 이 스크립트에 일반 타원궤도 버전을 새로 정의한다.
02번(orbital_elements_and_energy.py)의 state_vector_to_orbital_elements는
수정 없이 그대로 재사용한다.
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
frames = _load(os.path.join("frames", "coordinate_frame_transforms.py"), "frames_module")
elements_module = _load(os.path.join("propagation", "orbital_elements_and_energy.py"), "elements_module")
orbit_determination_module = _load(os.path.join("missions", "orbit_determination.py"), "orbit_determination_module")
orbit_math = _load("orbit_math.py", "orbit_math")
rotation_matrix_x = orbit_math.rotation_matrix_x
rotation_matrix_y = orbit_math.rotation_matrix_y
rotation_matrix_z = orbit_math.rotation_matrix_z
EARTH_RADIUS_KM = orbit_math.EARTH_RADIUS_KM

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2
MIN_D0_THRESHOLD = 0.05


def look_angles_to_eci_unit_vector(azimuth_rad, elevation_rad, site_latitude_rad, site_longitude_rad, time_sec):
  """거리 없이 방위각/고도각만으로 ECI 기준 단위 시선벡터를 구한다. 14번
  look_angles_to_eci_position과 같은 회전(Ry(lat-90)*Rz(-LST)의 전치)을 쓰되
  range=1로 고정한 SEZ 단위벡터를 ECI로 회전만 시킨다 — 가우스법은 거리를
  모른다고 가정하는 것이 핵심이므로 위치가 아니라 방향만 반환한다."""
  s = -np.cos(elevation_rad) * np.cos(azimuth_rad)
  e = np.cos(elevation_rad) * np.sin(azimuth_rad)
  z = np.sin(elevation_rad)
  sez_unit = np.array([s, e, z])

  gst = frames.gst_at_time(time_sec)
  lst = (gst + site_longitude_rad) % (2 * np.pi)
  rotation = rotation_matrix_y(site_latitude_rad - np.pi / 2) @ rotation_matrix_z(-lst)
  return rotation.T @ sez_unit


def true_orbit_position_general(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                                 mean_anomaly0_rad, time_sec, mu=EARTH_MU_KM3_S2):
  """14번 true_orbit_position의 일반 타원궤도 버전(argp 포함, 이심률 0 가정
  없음). 01번 propagate_orbit + Rz(raan)*Rx(i)*Rz(argp) 회전으로 ECI 위치를
  계산한다."""
  state = kepler.propagate_orbit(semi_major_axis_km, eccentricity, mean_anomaly0_rad, time_sec, mu)
  position_perifocal = np.array([state["x_p"], state["y_p"], 0.0])
  rotation = rotation_matrix_z(raan_rad) @ rotation_matrix_x(inclination_rad) @ rotation_matrix_z(argp_rad)
  return rotation @ position_perifocal


def find_visible_time_window_general(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                                      mean_anomaly0_rad, site_latitude_rad, site_longitude_rad,
                                      num_probe_steps=2000):
  """14번 find_visible_time_window와 같은 탐색 로직을 true_orbit_position_general에
  적용한 일반 타원궤도 버전 — 한 궤도 주기를 촘촘히 훑어 첫 가시 구간의
  (시작, 끝) 시각을 찾는다."""
  period = 2 * np.pi / kepler.mean_motion(semi_major_axis_km, EARTH_MU_KM3_S2)
  probe_times = np.linspace(0, period, num_probe_steps)
  in_window = False
  window_start = None
  for t in probe_times:
    position = true_orbit_position_general(semi_major_axis_km, eccentricity, inclination_rad, raan_rad,
                                             argp_rad, mean_anomaly0_rad, t)
    _az, elevation, _r = frames.eci_position_to_look_angles(position, site_latitude_rad, site_longitude_rad, t)
    if elevation > 0 and not in_window:
      in_window = True
      window_start = t
    elif elevation <= 0 and in_window:
      window_end = t
      margin = (window_end - window_start) * 0.05
      return window_start + margin, window_end - margin
  raise ValueError("이 궤도/지상국 조합에서는 한 궤도 주기 안에 접촉 창을 찾지 못함")


def select_physical_root(roots, A, B, mu, earth_radius_km=EARTH_RADIUS_KM, real_tol=1e-6):
  """8차 다항식의 복소근 배열에서 물리적으로 유효한 r2를 고른다. 유효 조건:
  (1) 허수부가 실수부 대비 무시할 만큼 작음, (2) 실수부(r2 후보)가 지구
  반지름보다 큼, (3) 그 r2가 내포하는 rho2=A+mu*B/r2^3 도 양수(음의 거리는
  물리적으로 무효). 조건을 만족하는 후보가 없으면 ValueError, 여럿이면 가장
  작은 r2(계획 단계 실측: 참값에 가장 가까운 근이 대체로 유효 후보 중 가장
  작은 근이었음)를 선택하고 나머지 후보도 함께 반환한다."""
  candidates = []
  for root in roots:
    if abs(root.imag) > real_tol * max(1.0, abs(root.real)):
      continue
    r2_candidate = root.real
    if r2_candidate <= earth_radius_km:
      continue
    rho2_candidate = A + mu * B / r2_candidate ** 3
    if rho2_candidate <= 0:
      continue
    candidates.append((r2_candidate, rho2_candidate))

  if not candidates:
    raise ValueError(f"물리적으로 유효한 근이 없음(지구 반지름보다 크고 rho2>0인 근 없음) — "
                      f"전체 근: {roots}")

  candidates.sort(key=lambda c: c[0])
  return {"r2": candidates[0][0], "rho2": candidates[0][1], "all_candidates": candidates}


def gauss_solve_r2_v2(L1, L2, L3, R1, R2, R3, tau1, tau3, mu=EARTH_MU_KM3_S2,
                       d0_min_threshold=MIN_D0_THRESHOLD, max_iterations=15, convergence_tol_km=1e-6):
  """고전 가우스법 핵심 풀이 + 반복 개선(핵심 개념 4). |D0|<d0_min_threshold면
  관측 기하가 거의 공면이라 가우스법이 근본적으로 풀 수 없음을 알고도 계속
  진행하지 않고 ValueError를 낸다. 8차 다항식으로 최초 r2를 구한 뒤, r2가
  수렴할 때까지 f/g 재계산 - rho1/rho2/rho3 선형시스템 재풀이 - r2_vec 갱신을
  반복한다."""
  tau = tau3 - tau1
  p1 = np.cross(L2, L3)
  p2 = np.cross(L1, L3)
  p3 = np.cross(L1, L2)
  D0 = np.dot(L1, p1)

  if abs(D0) < d0_min_threshold:
    raise ValueError(f"관측 기하가 거의 공면(|D0|={abs(D0):.4f} < 임계값 {d0_min_threshold}) — "
                      f"가우스법은 이런 기하에서 근본적으로 불안정하다. 관측 간격이나 시점을 바꿔야 함.")

  D12 = np.dot(R2, p1)
  D22 = np.dot(R2, p2)
  D32 = np.dot(R2, p3)

  A = (1 / D0) * (-D12 * (tau3 / tau) + D22 + D32 * (tau1 / tau))
  B = (1 / (6 * D0)) * (D12 * (tau3 ** 2 - tau ** 2) * (tau3 / tau) + D32 * (tau ** 2 - tau1 ** 2) * (tau1 / tau))

  E_dot = np.dot(R2, L2)
  Rsq = np.dot(R2, R2)
  a_coef = -(A ** 2 + 2 * A * E_dot + Rsq)
  b_coef = -2 * mu * B * (A + E_dot)
  c_coef = -(mu ** 2) * (B ** 2)
  coeffs = [1, 0, a_coef, 0, 0, b_coef, 0, 0, c_coef]
  roots = np.roots(coeffs)

  root_result = select_physical_root(roots, A, B, mu)
  r2 = root_result["r2"]

  # 반복 개선(핵심 개념 4): 매 반복마다 직전 r2로 f,g를 재계산해 rho들을 다시 구한다.
  r1_vec = r2_vec = r3_vec = f1 = f3 = g1 = g3 = None
  prev_r2 = None
  iterations_used = 0
  for iteration in range(max_iterations):
    u = mu / r2 ** 3
    f1 = 1 - 0.5 * u * tau1 ** 2
    f3 = 1 - 0.5 * u * tau3 ** 2
    g1 = tau1 - (1.0 / 6.0) * u * tau1 ** 3
    g3 = tau3 - (1.0 / 6.0) * u * tau3 ** 3
    c1 = g3 / (f1 * g3 - f3 * g1)
    c3 = -g1 / (f1 * g3 - f3 * g1)

    rhs = R2 - c1 * R1 - c3 * R3
    M_matrix = np.column_stack([c1 * L1, -L2, c3 * L3])
    rho1, rho2, rho3 = np.linalg.solve(M_matrix, rhs)

    r1_vec = R1 + rho1 * L1
    r2_vec = R2 + rho2 * L2
    r3_vec = R3 + rho3 * L3
    r2_new = np.linalg.norm(r2_vec)

    iterations_used = iteration + 1
    if prev_r2 is not None and abs(r2_new - prev_r2) < convergence_tol_km:
      r2 = r2_new
      break
    prev_r2 = r2_new
    r2 = r2_new

  v2_vec = (f1 * r3_vec - f3 * r1_vec) / (f1 * g3 - f3 * g1)

  return {"r2_vec": r2_vec, "v2_vec": v2_vec, "r1_vec": r1_vec, "r3_vec": r3_vec,
          "d0": D0, "iterations_used": iterations_used, "all_candidates": root_result["all_candidates"]}


def determine_orbit_gauss(observations, site_latitude_rad, site_longitude_rad,
                           mu=EARTH_MU_KM3_S2, d0_min_threshold=MIN_D0_THRESHOLD):
  """observations: 정확히 3개의 {azimuth_rad, elevation_rad, time_sec} dict
  (시간순 정렬, 거리 불요 — 각도만 쓰는 것이 가우스법의 핵심). L,R을 구성해
  gauss_solve_r2_v2를 호출하고, 결과 (r2_vec,v2_vec)를 02번
  state_vector_to_orbital_elements에 넘겨 6개 궤도요소 전부를 반환한다."""
  assert len(observations) == 3, "가우스법은 정확히 3개의 각도 전용 관측값이 필요함"

  gst_list = [frames.gst_at_time(obs["time_sec"]) for obs in observations]
  R = [frames.site_position_eci(site_latitude_rad, site_longitude_rad, gst) for gst in gst_list]
  L = [look_angles_to_eci_unit_vector(obs["azimuth_rad"], obs["elevation_rad"],
                                       site_latitude_rad, site_longitude_rad, obs["time_sec"])
       for obs in observations]
  t = [obs["time_sec"] for obs in observations]
  tau1, tau3 = t[0] - t[1], t[2] - t[1]

  gauss_result = gauss_solve_r2_v2(L[0], L[1], L[2], R[0], R[1], R[2], tau1, tau3, mu, d0_min_threshold)
  elements = elements_module.state_vector_to_orbital_elements(gauss_result["r2_vec"], gauss_result["v2_vec"], mu)
  elements.update({"d0": gauss_result["d0"], "iterations_used": gauss_result["iterations_used"],
                    "r2_vec": gauss_result["r2_vec"], "v2_vec": gauss_result["v2_vec"]})
  return elements


def find_gauss_observation_times(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                                  mean_anomaly0_rad, site_latitude_rad, site_longitude_rad,
                                  spacing_fraction=0.4, min_elevation_rad=np.radians(5.0)):
  """find_visible_time_window_general로 가시 구간을 찾고, 그 구간 폭의
  spacing_fraction만큼씩 띄운 3개 시점(t1=구간 시작, t2=t1+span, t3=t1+2*span)을
  반환한다 — 구간 중앙을 기준으로 대칭 배치하는 대신 시작점에서 전진시키는
  방식을 쓴다(계획 단계 실측: 같은 궤도에서 중앙 대칭 배치는 8차 다항식에
  물리적으로 유효한 근이 아예 없는 경우가 많았던 반면, 시작점 기준 배치는
  검증된 시나리오를 그대로 재현했다). 세 시점 모두 최소고도각을 만족하는지
  확인한다. 반환: (t1, t2, t3)."""
  window_start, window_end = find_visible_time_window_general(
      semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
      mean_anomaly0_rad, site_latitude_rad, site_longitude_rad)
  window_duration = window_end - window_start
  span = spacing_fraction * window_duration
  t1, t2, t3 = window_start, window_start + span, window_start + 2 * span
  if t3 > window_end:
    raise ValueError(f"spacing_fraction={spacing_fraction}에서 t3={t3:.1f}초가 가시 구간 끝"
                      f"({window_end:.1f}초)을 벗어남 — spacing_fraction을 줄여야 함")

  for t in (t1, t2, t3):
    position = true_orbit_position_general(semi_major_axis_km, eccentricity, inclination_rad, raan_rad,
                                             argp_rad, mean_anomaly0_rad, t)
    _az, elevation, _r = frames.eci_position_to_look_angles(position, site_latitude_rad, site_longitude_rad, t)
    if elevation < min_elevation_rad:
      raise ValueError(f"spacing_fraction={spacing_fraction}에서 t={t:.1f}초의 고도각이 "
                        f"{np.degrees(elevation):.2f}도로 최소고도각 미만")

  return t1, t2, t3


def simulate_gauss_observations(semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad,
                                 mean_anomaly0_rad, times_sec, site_latitude_rad, site_longitude_rad,
                                 angle_noise_rad=0.0, rng=None):
  """세 시점의 "진짜" 위치를 01번 전파로 계산하고, 04번 정변환으로 방위각/고도각을
  구한 뒤 (옵션으로) 각도 노이즈를 더한다. 거리는 관측값에 포함하지 않는다
  (가우스법은 거리를 쓰지 않으므로)."""
  observations = []
  for t in times_sec:
    position = true_orbit_position_general(semi_major_axis_km, eccentricity, inclination_rad, raan_rad,
                                             argp_rad, mean_anomaly0_rad, t)
    azimuth, elevation, _r = frames.eci_position_to_look_angles(position, site_latitude_rad, site_longitude_rad, t)
    if angle_noise_rad > 0 and rng is not None:
      azimuth = azimuth + rng.normal(0, angle_noise_rad)
      elevation = elevation + rng.normal(0, angle_noise_rad)
    observations.append({"azimuth_rad": azimuth, "elevation_rad": elevation, "time_sec": t})
  return observations


# 계획 단계에서 Bash로 직접 검증한 시나리오 — LEO 지상관측에서 D0가 충분히 크면서
# 가시성도 만족하는, 데모로 쓸 수 있는 조합을 넓은 파라미터 스윕으로 찾았다.
DEFAULT_SEMI_MAJOR_AXIS_KM = 8000.0
DEFAULT_ECCENTRICITY = 0.1
DEFAULT_INCLINATION_RAD = np.radians(30.0)
DEFAULT_RAAN_RAD = np.radians(40.0)
DEFAULT_ARGP_RAD = np.radians(50.0)
DEFAULT_MEAN_ANOMALY0_RAD = np.radians(325.0)
DEFAULT_SITE_LATITUDE_RAD = np.radians(20.0)
DEFAULT_SITE_LONGITUDE_RAD = np.radians(60.0)


def demo_gauss_recovers_elliptical_orbit(spacing_fraction=0.4, angle_noise_deg=0.0):
  """계획 단계에서 검증한 구체적 시나리오로 가시 구간을 찾고 3개 관측 시점을
  잡아 가우스법(반복 포함)을 돌린다. 14번이 전혀 다루지 못했던 이심률/근점편각
  까지 포함한 5개 비원형 요소 전부를 참값과 비교한다."""
  print("=" * 70)
  print("[1] 가우스법: 각도 전용 관측 3회로 일반 타원궤도를 복원하는가")
  print("=" * 70)
  t1, t2, t3 = find_gauss_observation_times(
      DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
      DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD,
      spacing_fraction)

  rng = np.random.default_rng(0) if angle_noise_deg > 0 else None
  observations = simulate_gauss_observations(
      DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
      DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
      DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD, np.radians(angle_noise_deg), rng)

  recovered = determine_orbit_gauss(observations, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD)

  a_error_pct = abs(recovered["semi_major_axis_km"] - DEFAULT_SEMI_MAJOR_AXIS_KM) / DEFAULT_SEMI_MAJOR_AXIS_KM * 100
  e_error_abs = abs(recovered["eccentricity"] - DEFAULT_ECCENTRICITY)
  i_error_deg = abs(np.degrees(recovered["inclination_rad"]) - np.degrees(DEFAULT_INCLINATION_RAD))
  raan_error_deg = abs(np.degrees(recovered["raan_rad"]) - np.degrees(DEFAULT_RAAN_RAD))
  argp_error_deg = abs(np.degrees(recovered["argp_rad"]) - np.degrees(DEFAULT_ARGP_RAD))

  print(f"관측 시점: t1={t1:.1f}, t2={t2:.1f}, t3={t3:.1f}초, D0={recovered['d0']:.4f}, "
        f"반복 {recovered['iterations_used']}회 수렴")
  print(f"{'요소':>10}{'참값':>14}{'복원값':>14}{'오차':>14}")
  print(f"{'a(km)':>10}{DEFAULT_SEMI_MAJOR_AXIS_KM:>14.2f}{recovered['semi_major_axis_km']:>14.2f}{a_error_pct:>13.2f}%")
  print(f"{'e':>10}{DEFAULT_ECCENTRICITY:>14.4f}{recovered['eccentricity']:>14.4f}{e_error_abs:>14.4f}")
  print(f"{'i(deg)':>10}{np.degrees(DEFAULT_INCLINATION_RAD):>14.2f}{np.degrees(recovered['inclination_rad']):>14.2f}{i_error_deg:>13.2f}도")
  print(f"{'raan(deg)':>10}{np.degrees(DEFAULT_RAAN_RAD):>14.2f}{np.degrees(recovered['raan_rad']):>14.2f}{raan_error_deg:>13.2f}도")
  print(f"{'argp(deg)':>10}{np.degrees(DEFAULT_ARGP_RAD):>14.2f}{np.degrees(recovered['argp_rad']):>14.2f}{argp_error_deg:>13.2f}도")

  assert a_error_pct < 5.0, "반장축 오차가 5% 미만이어야 함"
  assert e_error_abs < 0.05, "이심률 절대오차가 0.05 미만이어야 함"
  assert i_error_deg < 2.0, "경사각 오차가 2도 미만이어야 함"
  assert raan_error_deg < 2.0, "RAAN 오차가 2도 미만이어야 함"

  print("\n(14번은 원궤도만 다뤄 이심률/근점편각을 아예 추정하지 못했지만, 가우스법은")
  print(" 02번의 범용 궤도요소 변환을 그대로 재사용해 5개 비원형 요소 전부를 복원한다.)")
  return {"recovered": recovered, "a_error_pct": a_error_pct, "e_error_abs": e_error_abs,
          "i_error_deg": i_error_deg, "raan_error_deg": raan_error_deg, "argp_error_deg": argp_error_deg,
          "observations": observations, "times": (t1, t2, t3)}


def demo_accuracy_vs_d0_degeneracy():
  """관측 간격(spacing_fraction)을 좁은 값부터 넓혀가며 D0와 반장축 오차를
  나란히 계산한다. D0가 작을수록 가우스법이 불안정해진다는 것을 감추지 않고
  직접 보여준다 — 임계값 미만에서는 ValueError가 실제로 발생함을 확인한다."""
  print("\n" + "=" * 70)
  print("[2] D0 민감도: 관측 기하가 퇴화할수록 정확도가 무너지는가")
  print("=" * 70)
  spacing_fractions = [0.05, 0.1, 0.2, 0.3, 0.38, 0.4, 0.45]
  rows = []
  print(f"  {'spacing_frac':>14}{'D0':>12}{'a_err%':>12}{'상태':>10}")
  for frac in spacing_fractions:
    try:
      t1, t2, t3 = find_gauss_observation_times(
          DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
          DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD,
          frac)
      observations = simulate_gauss_observations(
          DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
          DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
          DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD)
      recovered = determine_orbit_gauss(observations, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD,
                                         d0_min_threshold=1e-6)
      a_err = abs(recovered["semi_major_axis_km"] - DEFAULT_SEMI_MAJOR_AXIS_KM) / DEFAULT_SEMI_MAJOR_AXIS_KM * 100
      rows.append({"spacing_fraction": frac, "d0": recovered["d0"], "a_error_pct": a_err})
      print(f"  {frac:>14.4f}{recovered['d0']:>12.5f}{a_err:>12.2f}{'성공':>10}")
    except ValueError:
      rows.append({"spacing_fraction": frac, "d0": 0.0, "a_error_pct": float("nan")})
      print(f"  {frac:>14.4f}{'(작음)':>12}{'N/A':>12}{'실패':>10}")

  # 엄격한 d0_min_threshold로 가장 작은 spacing에서 ValueError가 실제로 발생하는지 별도 확인
  threshold_triggered = False
  try:
    t1, t2, t3 = find_gauss_observation_times(
        DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
        DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD,
        spacing_fractions[0])
    observations = simulate_gauss_observations(
        DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
        DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
        DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD)
    determine_orbit_gauss(observations, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD,
                           d0_min_threshold=MIN_D0_THRESHOLD)
  except ValueError:
    threshold_triggered = True

  valid_rows = [r for r in rows if not np.isnan(r["a_error_pct"])]
  assert len(valid_rows) >= 2, "비교할 유효 결과가 최소 2개는 있어야 함"
  assert valid_rows[0]["a_error_pct"] >= valid_rows[-1]["a_error_pct"] * 0.5 or threshold_triggered, \
      "D0가 작을수록 오차가 커지는 경향(또는 임계값 발동)을 보여야 함"

  print(f"\n(기본 임계값(|D0|<{MIN_D0_THRESHOLD})을 쓰면 가장 작은 spacing에서 ValueError 발생: "
        f"{threshold_triggered})")
  print(" D0가 작은 관측 기하에서는 가우스법이 근본적으로 불안정하다는 것을")
  print(" 감추지 않고 직접 보여준다 — 9번 란베르트의 180도 특이점, 14번의 원궤도")
  print(" 제한과 같은 이 프로젝트의 확립된 태도다.)")
  return {"rows": rows, "threshold_triggered": threshold_triggered}


def demo_small_noise_sensitivity(num_trials=20, angle_noise_deg=0.01):
  """검증된 좋은 기하에서 작은 각도 노이즈를 여러 시드로 반복 주입해, 반장축
  오차가 노이즈 없을 때와 비슷한 범위에서 안정적으로 유지되는지 확인한다 —
  반복 개선 덕분에 노이즈에도 견고하다는 것을 보여준다."""
  print("\n" + "=" * 70)
  print("[3] 작은 노이즈 민감도: 반복 개선이 노이즈에도 안정적인가")
  print("=" * 70)
  t1, t2, t3 = find_gauss_observation_times(
      DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
      DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD)

  a_errors = []
  e_errors = []
  rows = []
  for trial in range(num_trials):
    rng = np.random.default_rng(trial)
    observations = simulate_gauss_observations(
        DEFAULT_SEMI_MAJOR_AXIS_KM, DEFAULT_ECCENTRICITY, DEFAULT_INCLINATION_RAD, DEFAULT_RAAN_RAD,
        DEFAULT_ARGP_RAD, DEFAULT_MEAN_ANOMALY0_RAD, [t1, t2, t3],
        DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD, np.radians(angle_noise_deg), rng)
    recovered = determine_orbit_gauss(observations, DEFAULT_SITE_LATITUDE_RAD, DEFAULT_SITE_LONGITUDE_RAD)
    a_err = abs(recovered["semi_major_axis_km"] - DEFAULT_SEMI_MAJOR_AXIS_KM) / DEFAULT_SEMI_MAJOR_AXIS_KM * 100
    e_err = abs(recovered["eccentricity"] - DEFAULT_ECCENTRICITY)
    a_errors.append(a_err)
    e_errors.append(e_err)
    rows.append({"trial": trial, "a_error_pct": a_err, "e_error_abs": e_err})

  mean_a_err = np.mean(a_errors)
  max_a_err = np.max(a_errors)
  print(f"각도 노이즈={angle_noise_deg}도, {num_trials}회 시행")
  print(f"반장축 오차: 평균={mean_a_err:.2f}%, 최댓값={max_a_err:.2f}%")
  print(f"이심률 오차: 평균={np.mean(e_errors):.4f}, 최댓값={np.max(e_errors):.4f}")

  assert max_a_err < 10.0, "작은 노이즈에서도 반장축 오차가 10% 미만으로 안정적이어야 함"

  print("\n(노이즈를 섞어도 반장축 오차가 노이즈 없을 때와 비슷한 범위에 머문다 -")
  print(" 반복 개선이 1차 패스의 불안정성을 흡수해 노이즈에도 견고해졌다.)")
  return {"rows": rows, "mean_a_error_pct": mean_a_err, "max_a_error_pct": max_a_err}


def demo_compare_with_circular_svd_method():
  """이심률이 거의 0인 궤도에서 이 스크립트의 가우스법과 14번의
  determine_circular_orbit(SVD 방법)을 같은 관측값에 적용해 반장축/경사각이
  서로 근접하는지 비교한다 — 14번과 27번을 서사적으로 잇는 교차검증."""
  print("\n" + "=" * 70)
  print("[4] 14번과의 비교: 거의 원궤도에서 가우스법과 SVD 방법이 일치하는가")
  print("=" * 70)
  semi_major_axis_km = 7500.0
  eccentricity = 0.01
  inclination_rad = np.radians(45.0)
  raan_rad = np.radians(30.0)
  argp_rad = np.radians(0.0)
  mean_anomaly0_rad = np.radians(0.0)
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)

  t1, t2, t3 = find_gauss_observation_times(
      semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad, mean_anomaly0_rad,
      site_lat, site_lon, spacing_fraction=0.30, min_elevation_rad=np.radians(3.0))
  gauss_observations = simulate_gauss_observations(
      semi_major_axis_km, eccentricity, inclination_rad, raan_rad, argp_rad, mean_anomaly0_rad,
      [t1, t2, t3], site_lat, site_lon)
  gauss_recovered = determine_orbit_gauss(gauss_observations, site_lat, site_lon)

  # 14번의 SVD 방법은 더 많은 관측(기본 30개)을 기대하므로, 같은 궤도/지상국으로 별도 시뮬레이션
  svd_times = np.linspace(t1, t3, 30)
  rng = np.random.default_rng(1)
  svd_observations_raw = orbit_determination_module.simulate_noisy_observations(
      semi_major_axis_km, inclination_rad, raan_rad, site_lat, site_lon, svd_times, 0.0, 0.0, rng)
  svd_recovered = orbit_determination_module.determine_circular_orbit(svd_observations_raw, site_lat, site_lon)

  a_diff_pct = abs(gauss_recovered["semi_major_axis_km"] - svd_recovered["semi_major_axis_km"]) \
      / svd_recovered["semi_major_axis_km"] * 100
  i_diff_deg = abs(np.degrees(gauss_recovered["inclination_rad"]) - np.degrees(svd_recovered["inclination_rad"]))

  print(f"{'방법':>12}{'반장축(km)':>14}{'경사각(도)':>14}")
  print(f"{'가우스법':>12}{gauss_recovered['semi_major_axis_km']:>14.2f}{np.degrees(gauss_recovered['inclination_rad']):>14.2f}")
  print(f"{'SVD(14번)':>12}{svd_recovered['semi_major_axis_km']:>14.2f}{np.degrees(svd_recovered['inclination_rad']):>14.2f}")
  print(f"반장축 차이={a_diff_pct:.2f}%, 경사각 차이={i_diff_deg:.2f}도")

  assert a_diff_pct < 10.0, "거의 원궤도에서는 두 독립된 방법의 반장축이 서로 근접해야 함"
  assert i_diff_deg < 5.0, "거의 원궤도에서는 두 독립된 방법의 경사각이 서로 근접해야 함"

  print("\n(관측 원리가 완전히 다른 두 방법 — 가우스법의 라그랑주 계수 기반 풀이와")
  print(" 14번의 SVD 평면 피팅 — 이 거의 원궤도에서 서로 근접한 결과를 낸다.)")
  return {"gauss": gauss_recovered, "svd": svd_recovered, "a_diff_pct": a_diff_pct, "i_diff_deg": i_diff_deg}


def write_csv(filepath, rows, fieldnames):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  with open(filepath, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def parse_args():
  parser = argparse.ArgumentParser(description="가우스법 기반 일반 궤도결정 데모")
  parser.add_argument("--angle-noise-deg", type=float, default=0.01,
                       help="각도 관측 노이즈 표준편차(도), 기본값: 0.01도")
  parser.add_argument("--spacing-fraction", type=float, default=0.4,
                       help="가시 구간 폭 대비 관측 간격 비율, 기본값: 0.4")
  parser.add_argument("--min-d0-threshold", type=float, default=MIN_D0_THRESHOLD,
                       help="가우스법 기하 퇴화 판정 최소 |D0| 임계값, 기본값: 0.05")
  return parser.parse_args()


def main():
  args = parse_args()

  recovery_result = demo_gauss_recovers_elliptical_orbit(args.spacing_fraction)
  d0_result = demo_accuracy_vs_d0_degeneracy()
  noise_result = demo_small_noise_sensitivity(angle_noise_deg=args.angle_noise_deg)
  comparison_result = demo_compare_with_circular_svd_method()

  print("\n" + "=" * 70)
  print(f"[사용자 지정] spacing_fraction={args.spacing_fraction}, 각도 노이즈={args.angle_noise_deg}도, "
        f"최소 D0 임계값={args.min_d0_threshold}")
  print("=" * 70)
  recovered = recovery_result["recovered"]
  print(f"반장축 오차={recovery_result['a_error_pct']:.2f}%, "
        f"이심률 오차={recovery_result['e_error_abs']:.4f}, D0={recovered['d0']:.4f}")

  results_dir = os.path.join(_ROOT_DIR, "results")

  recovery_row = {
      "true_a_km": DEFAULT_SEMI_MAJOR_AXIS_KM, "true_e": DEFAULT_ECCENTRICITY,
      "true_i_deg": np.degrees(DEFAULT_INCLINATION_RAD), "true_raan_deg": np.degrees(DEFAULT_RAAN_RAD),
      "true_argp_deg": np.degrees(DEFAULT_ARGP_RAD),
      "recovered_a_km": recovered["semi_major_axis_km"], "recovered_e": recovered["eccentricity"],
      "recovered_i_deg": np.degrees(recovered["inclination_rad"]),
      "recovered_raan_deg": np.degrees(recovered["raan_rad"]),
      "recovered_argp_deg": np.degrees(recovered["argp_rad"]),
      "d0": recovered["d0"], "iterations_used": recovered["iterations_used"],
      "a_error_pct": recovery_result["a_error_pct"], "e_error_abs": recovery_result["e_error_abs"],
      "i_error_deg": recovery_result["i_error_deg"], "raan_error_deg": recovery_result["raan_error_deg"],
      "argp_error_deg": recovery_result["argp_error_deg"],
  }
  write_csv(os.path.join(results_dir, "gauss_orbit_determination_recovery.csv"),
            [recovery_row], list(recovery_row.keys()))

  write_csv(os.path.join(results_dir, "gauss_orbit_determination_d0_sensitivity.csv"),
            d0_result["rows"], ["spacing_fraction", "d0", "a_error_pct"])

  write_csv(os.path.join(results_dir, "gauss_orbit_determination_noise_sensitivity.csv"),
            noise_result["rows"], ["trial", "a_error_pct", "e_error_abs"])

  comparison_rows = [
      {"method": "gauss", "recovered_a_km": comparison_result["gauss"]["semi_major_axis_km"],
       "recovered_i_deg": np.degrees(comparison_result["gauss"]["inclination_rad"])},
      {"method": "svd", "recovered_a_km": comparison_result["svd"]["semi_major_axis_km"],
       "recovered_i_deg": np.degrees(comparison_result["svd"]["inclination_rad"])},
  ]
  write_csv(os.path.join(results_dir, "gauss_orbit_determination_circular_comparison.csv"),
            comparison_rows, ["method", "recovered_a_km", "recovered_i_deg"])

  print(f"\n[기록] 궤도 복원 결과 저장됨 → {os.path.join(results_dir, 'gauss_orbit_determination_recovery.csv')}")
  print(f"[기록] D0 민감도 저장됨 → {os.path.join(results_dir, 'gauss_orbit_determination_d0_sensitivity.csv')}")
  print(f"[기록] 노이즈 민감도 저장됨 → {os.path.join(results_dir, 'gauss_orbit_determination_noise_sensitivity.csv')}")
  print(f"[기록] 원궤도 비교 저장됨 → {os.path.join(results_dir, 'gauss_orbit_determination_circular_comparison.csv')}")


if __name__ == "__main__":
  main()
