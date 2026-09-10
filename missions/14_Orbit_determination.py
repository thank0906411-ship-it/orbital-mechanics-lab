"""
궤도 결정(Orbit Determination) - 노이즈 낀 레이더 관측값에서 궤도요소 역산

02번은 "궤도요소가 주어졌을 때 상태벡터를 구하는" 순문제와 그 역(상태벡터에서
궤도요소)을 다뤘지만, 그 역변환조차 "정확한" 상태벡터가 이미 손에 있다고
가정했다. 실제 위성 운영에서는 그런 사치가 없다 — 지상 레이더가 측정한 거리,
방위각, 고도각에는 항상 측정 오차(노이즈)가 섞여 있고, 궤도요소는 그 노이즈 낀
관측값들"만"으로 추정해야 한다. 이 스크립트는 그 방향의 문제를 다룬다: 04번의
관측 기하 변환을 거꾸로 뒤집어 노이즈 낀 관측값에서 순간 위치를 복원하고, 여러
시점의 복원된 위치들에 최소자승으로 원궤도를 피팅한다.

이 스크립트는 단순화를 위해 이심률이 0인 원궤도만 다룬다 — 완전히 일반적인
케플러 궤도 피팅(가우스법 등)은 라그랑주 계수와 고차 다항식 풀이가 추가로
필요해 범위가 커진다. 원궤도로 제한하면 "여러 관측 위치가 모두 같은 반지름의
구면 위, 하나의 평면 위에 있어야 한다"는 훨씬 단순한 기하 조건으로 최소자승
피팅을 할 수 있다.

핵심 개념 1: 레이더 관측(거리·방위각·고도각) → 순간 위치는 04번의 정확한 역변환이다
  04번의 eci_to_topocentric_sez는 SEZ = Ry(phi-90)*Rz(-LST)*(r_ECI - r_site)로
  ECI 위치를 관측자 기준 SEZ 좌표로 바꿨다. 회전행렬은 직교행렬이라 전치가 곧
  역행렬이므로(04번에서 이미 검증), 역변환은
      r_ECI = r_site + Rz(LST)*Ry(90-phi) * SEZ
  로 정확히 구해진다. SEZ 벡터 자체는 관측된 (방위각 Az, 고도각 El, 거리 r)에서
      S = -r*cos(El)*cos(Az),  E = r*cos(El)*sin(Az),  Z = r*sin(El)
  로 복원한다(04번의 Az=atan2(E,-S), El=asin(Z/r) 정의를 그대로 거꾸로 푼 것).

핵심 개념 2: 원궤도라는 가정 하에, 여러 위치는 "하나의 평면 + 하나의 반지름" 조건을 만족해야 한다
  실제 궤도가 원궤도라면, 모든 관측 시점의 위치벡터는 (1) 크기가 모두 같고(반지름
  a), (2) 모두 같은 평면(궤도면, 원점을 지나는 평면) 위에 있어야 한다. 노이즈가
  섞이면 이 두 조건이 정확히는 성립하지 않으므로 최소자승으로 근사한다: 궤도면
  법선은 위치벡터들이 이루는 행렬의 특이값 분해(SVD)에서 가장 작은 특이값에
  대응하는 벡터로 추정하고(그 방향이 "위치벡터들과 가장 수직에 가까운" 방향),
  반지름은 각 위치벡터 크기의 평균으로 추정한다.

핵심 개념 3: 궤도면 법선에서 경사각과 RAAN이 바로 나온다
  02번의 state_vector_to_orbital_elements가 각운동량 벡터(h_vec = r x v)에서
  경사각(cos(i) = h_z/|h|)과 RAAN(승교점 벡터 n = z_hat x h_vec의 방향)을 유도한
  것과 똑같은 공식을, 이번엔 속도가 없어도 "궤도면 법선"(부호가 h_vec와 같은
  방향인지는 알 수 없지만 축 방향은 같음)만으로 그대로 적용할 수 있다 — 경사각과
  RAAN은 궤도면의 방향만으로 결정되는 양이기 때문이다.

핵심 개념 4: 관측이 부정확할수록 궤도 결정도 부정확해진다 (당연하지만 실측 검증)
  관측 노이즈(거리·각도 측정 오차)가 커지면 복원된 위치들이 실제 궤도면에서 더
  많이 벗어나고, 따라서 SVD로 추정한 궤도면 법선의 오차도 커진다. 이 스크립트는
  노이즈 크기를 실제로 바꿔가며 복원된 경사각/RAAN의 오차가 커지는 것을 직접
  확인한다 — 그리고 관측 개수를 늘리면(2개→10개) 노이즈가 평균화되어 다시
  안정된다는 것도 함께 보인다.

01번(케플러 전파), 02번(상태벡터-궤도요소), 04번(좌표변환)과의 관계: 01번으로
"진짜" 궤도를 전파해 관측 시뮬레이션의 정답을 만들고, 04번의 정변환으로 노이즈
없는 관측값을 만든 뒤 노이즈를 더한다. 이 노이즈 낀 관측값을 다시 위치로
복원하는 역변환은 이 스크립트가 새로 구현하지만, 02번과 동일한 각운동량-경사각
-RAAN 공식을 그대로 재사용한다.
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


kepler = _load(os.path.join("propagation", "01_Kepler_orbit_propagation.py"), "kepler_module")
frames = _load(os.path.join("frames", "04_Coordinate_frame_transforms.py"), "frames_module")
orbit_math = _load("orbit_math.py", "orbit_math")
rotation_matrix_x = orbit_math.rotation_matrix_x
rotation_matrix_y = orbit_math.rotation_matrix_y
rotation_matrix_z = orbit_math.rotation_matrix_z

EARTH_MU_KM3_S2 = kepler.EARTH_MU_KM3_S2


def look_angles_to_eci_position(azimuth_rad, elevation_rad, range_km, site_latitude_rad, site_longitude_rad, time_sec):
  """04번의 eci_position_to_look_angles의 정확한 역함수: (방위각, 고도각, 거리) +
  관측자 위경도/시각 -> ECI 위치."""
  s = -range_km * np.cos(elevation_rad) * np.cos(azimuth_rad)
  e = range_km * np.cos(elevation_rad) * np.sin(azimuth_rad)
  z = range_km * np.sin(elevation_rad)
  sez = np.array([s, e, z])

  gst = frames.gst_at_time(time_sec)
  lst = (gst + site_longitude_rad) % (2 * np.pi)
  site_pos = frames.site_position_eci(site_latitude_rad, site_longitude_rad, gst)

  # eci_to_topocentric_sez의 회전(Ry(phi-90)*Rz(-LST))은 직교행렬이라 전치가 역행렬.
  rotation = rotation_matrix_y(site_latitude_rad - np.pi / 2) @ rotation_matrix_z(-lst)
  relative_position = rotation.T @ sez
  return site_pos + relative_position


def true_orbit_position(semi_major_axis_km, inclination_rad, raan_rad, time_sec):
  """01번 전파 + 05/11번과 동일한 회전 관례(Rz(RAAN)*Rx(i), 원궤도라 argp 생략)로
  "진짜" 궤도 위치(ECI)를 계산한다."""
  state = kepler.propagate_orbit(semi_major_axis_km, 0.0, mean_anomaly0_rad=0.0, time_sec=time_sec)
  position_perifocal = np.array([state["x_p"], state["y_p"], 0.0])
  rotation = rotation_matrix_z(raan_rad) @ rotation_matrix_x(inclination_rad)
  return rotation @ position_perifocal


def find_visible_time_window(semi_major_axis_km, inclination_rad, raan_rad, site_latitude_rad,
                              site_longitude_rad, num_probe_steps=2000):
  """한 궤도 주기를 촘촘히 훑어, 지상국 기준 고도각이 0을 넘는(지평선 위) 첫 접촉
  구간의 (시작, 끝) 시각을 찾는다. 05번의 접촉 창 개념을 재사용하되, 이 스크립트는
  "관측 가능한 시간대에서 몇 번 관측했는가"만 필요하므로 첫 구간 하나만 찾는다.
  구간의 양 끝은 고도각이 정확히 0(경계)이라 linspace로 끝점을 다시 샘플링하면
  부동소수점 오차로 지평선 아래로 떨어질 수 있으므로, 구간 폭의 5%만큼 안쪽으로
  여유를 두고 반환한다."""
  period = 2 * np.pi / kepler.mean_motion(semi_major_axis_km)
  probe_times = np.linspace(0, period, num_probe_steps)
  in_window = False
  window_start = None
  for t in probe_times:
    position = true_orbit_position(semi_major_axis_km, inclination_rad, raan_rad, t)
    _az, elevation, _r = frames.eci_position_to_look_angles(position, site_latitude_rad, site_longitude_rad, t)
    if elevation > 0 and not in_window:
      in_window = True
      window_start = t
    elif elevation <= 0 and in_window:
      window_end = t
      margin = (window_end - window_start) * 0.05
      return window_start + margin, window_end - margin
  raise ValueError("이 궤도/지상국 조합에서는 한 궤도 주기 안에 접촉 창을 찾지 못함")


def simulate_noisy_observations(semi_major_axis_km, inclination_rad, raan_rad, site_latitude_rad,
                                 site_longitude_rad, times_sec, range_noise_km, angle_noise_rad, rng):
  """01번 전파 + orbit_math 회전으로 "진짜" 궤도 위치를 만들고, 04번 정변환으로
  방위각/고도각/거리를 계산한 뒤 정규분포 노이즈를 더한다. 반환: 관측값 리스트
  [{azimuth_rad, elevation_rad, range_km, time_sec}, ...] (지평선 아래 시점은 제외)."""
  observations = []
  for t in times_sec:
    true_position = true_orbit_position(semi_major_axis_km, inclination_rad, raan_rad, t)
    azimuth, elevation, r = frames.eci_position_to_look_angles(true_position, site_latitude_rad, site_longitude_rad, t)
    if elevation <= 0:
      continue  # 지평선 아래(관측 불가능)인 시점은 관측값에서 제외

    noisy_azimuth = azimuth + rng.normal(0, angle_noise_rad)
    noisy_elevation = elevation + rng.normal(0, angle_noise_rad)
    noisy_range = r + rng.normal(0, range_noise_km)
    observations.append({"azimuth_rad": noisy_azimuth, "elevation_rad": noisy_elevation,
                          "range_km": noisy_range, "time_sec": t})
  return observations


def orbit_normal_from_positions(position_list):
  """여러 위치벡터로부터 궤도면 법선을 SVD로 추정한다 — 위치행렬의 가장 작은
  특이값에 대응하는 오른쪽 특이벡터가 위치벡터들에 가장 수직에 가까운 방향이다."""
  position_matrix = np.array(position_list)
  _u, _s, vt = np.linalg.svd(position_matrix)
  normal = vt[-1]
  return normal / np.linalg.norm(normal)


def determine_circular_orbit(observations, site_latitude_rad, site_longitude_rad):
  """관측값 리스트로부터 (반장축, 경사각, RAAN)을 추정한다. 02번과 동일한
  각운동량 벡터 -> 경사각/RAAN 공식을 궤도면 법선에 그대로 적용한다."""
  positions = [look_angles_to_eci_position(obs["azimuth_rad"], obs["elevation_rad"], obs["range_km"],
                                            site_latitude_rad, site_longitude_rad, obs["time_sec"])
               for obs in observations]

  semi_major_axis_km = np.mean([np.linalg.norm(p) for p in positions])
  normal = orbit_normal_from_positions(positions)

  # 02번 state_vector_to_orbital_elements와 동일한 공식: h_vec 대신 궤도면 법선을 쓴다.
  inclination = np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))
  z_hat = np.array([0.0, 0.0, 1.0])
  node_vec = np.cross(z_hat, normal if normal[2] >= 0 else -normal)
  node_norm = np.linalg.norm(node_vec)
  if node_norm < 1e-9:
    raan = 0.0  # 극궤도: 승교점이 정의되지 않음(02번과 동일한 특이 케이스)
  else:
    raan = np.arctan2(node_vec[1], node_vec[0]) % (2 * np.pi)

  return {"semi_major_axis_km": semi_major_axis_km, "inclination_rad": inclination, "raan_rad": raan,
          "num_observations": len(observations)}


def demo_look_angles_roundtrip():
  """look_angles_to_eci_position이 04번 eci_position_to_look_angles의 정확한
  역함수인지 여러 무작위 ECI 위치로 왕복 검증한다. 방향을 완전히 균일한 구면
  무작위로 뽑으면 지구 전체 입체각 중 지상국 상공(고도각>0)은 절반도 안 되는
  좁은 영역이라 대부분 지평선 아래로 걸러진다 — 방향을 지상국의 천정(그 시각의
  site 위치 방향) 쪽으로 편향시켜 뽑아 유효 표본 비율을 높인다."""
  print("=" * 70)
  print("[1] 관측값(방위각/고도각/거리) <-> ECI 위치 왕복 변환 검증")
  print("=" * 70)
  site_lat = np.radians(35.0)
  site_lon = np.radians(129.0)
  rng = np.random.default_rng(42)

  rows = []
  print(f"  {'시행':>6}{'원본 |r|(km)':>16}{'복원 오차(km)':>16}")
  for trial in range(20):
    t = rng.uniform(0, 3600.0)
    zenith_direction = frames.site_position_eci(site_lat, site_lon, frames.gst_at_time(t))
    zenith_direction /= np.linalg.norm(zenith_direction)
    wobble = rng.normal(size=3) * 0.3
    direction = zenith_direction + wobble
    direction /= np.linalg.norm(direction)
    radius = rng.uniform(7000.0, 20000.0)
    original_position = direction * radius

    azimuth, elevation, r = frames.eci_position_to_look_angles(original_position, site_lat, site_lon, t)
    if elevation <= 0:
      continue
    recovered_position = look_angles_to_eci_position(azimuth, elevation, r, site_lat, site_lon, t)
    error_km = np.linalg.norm(recovered_position - original_position)
    print(f"  {trial:>6}{radius:>16.2f}{error_km:>16.2e}")
    rows.append({"trial": trial, "original_r_km": radius, "roundtrip_error_km": error_km})

  assert all(row["roundtrip_error_km"] < 1e-6 for row in rows), "왕복 변환 오차는 부동소수점 정밀도 수준이어야 함"
  print("\n(방위각/고도각/거리에서 복원한 ECI 위치가 원본과 부동소수점 정밀도 수준으로")
  print(" 일치한다 — 이 역변환이 04번 정변환의 정확한 역함수라는 뜻이다.)")
  return rows


def demo_circular_orbit_recovered_from_noisy_observations():
  """알려진 원궤도를 관측·노이즈 주입한 뒤 궤도 결정을 수행해, 복원된 반장축/
  경사각/RAAN이 원래 값과 거의 일치하는지 확인한다(작은 노이즈 기준)."""
  print("\n" + "=" * 70)
  print("[2] 작은 노이즈에서 원궤도 요소 복원 검증")
  print("=" * 70)
  true_a, true_i, true_raan = 7000.0, np.radians(53.0), np.radians(80.0)
  site_lat, site_lon = np.radians(35.0), np.radians(129.0)
  window_start, window_end = find_visible_time_window(true_a, true_i, true_raan, site_lat, site_lon)
  times = np.linspace(window_start, window_end, 30)
  rng = np.random.default_rng(7)

  observations = simulate_noisy_observations(true_a, true_i, true_raan, site_lat, site_lon, times,
                                              range_noise_km=0.5, angle_noise_rad=np.radians(0.05), rng=rng)
  result = determine_circular_orbit(observations, site_lat, site_lon)

  print(f"진짜 궤도요소: a={true_a}km, i={np.degrees(true_i):.2f}도, RAAN={np.degrees(true_raan):.2f}도")
  print(f"복원된 궤도요소: a={result['semi_major_axis_km']:.2f}km, "
        f"i={np.degrees(result['inclination_rad']):.2f}도, RAAN={np.degrees(result['raan_rad']):.2f}도")
  print(f"사용된 관측 개수: {result['num_observations']}개(지평선 아래 시점 제외)")

  a_error_pct = abs(result["semi_major_axis_km"] - true_a) / true_a * 100
  i_error_deg = abs(np.degrees(result["inclination_rad"]) - np.degrees(true_i))
  print(f"\n반장축 오차: {a_error_pct:.3f}%, 경사각 오차: {i_error_deg:.3f}도")

  assert a_error_pct < 1.0, "작은 관측 노이즈에서는 반장축 오차가 1% 미만이어야 함"
  assert i_error_deg < 1.0, "작은 관측 노이즈에서는 경사각 오차가 1도 미만이어야 함"
  print("\n(레이더 관측값(거리/방위각/고도각)만으로, 그 관측에 쓰인 원래 궤도요소를")
  print(" 거의 정확하게 복원했다 — 02번의 순문제(궤도요소->상태벡터)를 거꾸로 푼 것이다.)")
  return {"true_a": true_a, "true_i_deg": np.degrees(true_i), "true_raan_deg": np.degrees(true_raan),
          "recovered_a": result["semi_major_axis_km"], "recovered_i_deg": np.degrees(result["inclination_rad"]),
          "recovered_raan_deg": np.degrees(result["raan_rad"]), "a_error_pct": a_error_pct, "i_error_deg": i_error_deg}


def demo_accuracy_degrades_with_observation_noise():
  """관측 노이즈 크기를 키워가며 복원된 경사각 오차가 커지는 경향을 확인한다."""
  print("\n" + "=" * 70)
  print("[3] 관측 노이즈가 커질수록 궤도 결정 정확도가 떨어진다")
  print("=" * 70)
  true_a, true_i, true_raan = 7000.0, np.radians(53.0), np.radians(80.0)
  site_lat, site_lon = np.radians(35.0), np.radians(129.0)
  window_start, window_end = find_visible_time_window(true_a, true_i, true_raan, site_lat, site_lon)
  times = np.linspace(window_start, window_end, 30)

  rows = []
  print(f"  {'각도 노이즈(도)':>16}{'경사각 오차(도)':>18}")
  for angle_noise_deg in [0.001, 0.05, 0.5, 2.0, 5.0]:
    rng = np.random.default_rng(123)  # 노이즈 크기 비교가 목적이므로 시드 고정
    observations = simulate_noisy_observations(true_a, true_i, true_raan, site_lat, site_lon, times,
                                                range_noise_km=0.5, angle_noise_rad=np.radians(angle_noise_deg), rng=rng)
    result = determine_circular_orbit(observations, site_lat, site_lon)
    i_error_deg = abs(np.degrees(result["inclination_rad"]) - np.degrees(true_i))
    print(f"  {angle_noise_deg:>16.3f}{i_error_deg:>18.4f}")
    rows.append({"angle_noise_deg": angle_noise_deg, "inclination_error_deg": i_error_deg})

  assert rows[-1]["inclination_error_deg"] > rows[0]["inclination_error_deg"], (
      "가장 큰 노이즈(5도)의 경사각 오차가 가장 작은 노이즈(0.001도)보다 커야 함"
  )
  print("\n(각도 관측 노이즈가 0.001도에서 5도로 커질수록 복원된 경사각 오차도 함께")
  print(" 커진다 — 궤도 결정의 정확도는 근본적으로 관측 장비의 정밀도에 좌우된다는")
  print(" 당연한 관계를, 실측 시뮬레이션으로 직접 확인했다.)")
  return rows


def demo_minimum_observations_needed():
  """관측 개수를 2개, 3개, 10개, 30개로 늘려가며 궤도면 법선(경사각) 추정이
  안정화되는 것을 보인다. 관측이 몇 개뿐일 때는 노이즈 하나하나의 요행에 따라
  결과가 시드마다 크게 요동치므로(2개짜리 단일 시행이 3개짜리보다 우연히 더
  정확하게 나오는 경우가 흔함), 시드를 여러 번 바꿔가며 평균 오차로 비교해야
  "개수가 늘수록 안정화된다"는 경향이 분산 감소로 드러난다."""
  print("\n" + "=" * 70)
  print("[4] 관측 개수가 늘수록 궤도 결정이 안정화된다 (여러 시행 평균)")
  print("=" * 70)
  true_a, true_i, true_raan = 7000.0, np.radians(53.0), np.radians(80.0)
  site_lat, site_lon = np.radians(35.0), np.radians(129.0)
  window_start, window_end = find_visible_time_window(true_a, true_i, true_raan, site_lat, site_lon)
  angle_noise_deg = 1.0
  num_trials = 30

  rows = []
  print(f"  {'관측 개수':>12}{'평균 경사각 오차(도)':>22}{'표준편차(도)':>16}")
  for num_obs in [2, 3, 10, 30]:
    times = np.linspace(window_start, window_end, num_obs)
    trial_errors = []
    actual_counts = []
    for trial in range(num_trials):
      rng = np.random.default_rng(1000 + trial)
      observations = simulate_noisy_observations(true_a, true_i, true_raan, site_lat, site_lon, times,
                                                  range_noise_km=0.5, angle_noise_rad=np.radians(angle_noise_deg), rng=rng)
      if len(observations) < 2:
        continue
      result = determine_circular_orbit(observations, site_lat, site_lon)
      trial_errors.append(abs(np.degrees(result["inclination_rad"]) - np.degrees(true_i)))
      actual_counts.append(result["num_observations"])
    mean_error_deg = np.mean(trial_errors)
    std_error_deg = np.std(trial_errors)
    print(f"  {int(np.mean(actual_counts)):>12}{mean_error_deg:>22.4f}{std_error_deg:>16.4f}")
    rows.append({"requested_num_obs": num_obs, "actual_num_obs": int(np.mean(actual_counts)),
                 "mean_inclination_error_deg": mean_error_deg, "std_inclination_error_deg": std_error_deg})

  assert rows[-1]["mean_inclination_error_deg"] < rows[0]["mean_inclination_error_deg"], (
      "관측 30개의 평균 오차가 관측 2개의 평균 오차보다 작아야 함"
  )
  print("\n(관측이 딱 2개뿐이면 궤도면 법선이 그 두 위치벡터의 외적 하나로만 정해져")
  print(" 노이즈에 매우 취약하지만(시행마다 오차가 크게 요동침), 관측 개수가 늘수록")
  print(" SVD가 여러 관측의 노이즈를 평균화해 평균 오차와 표준편차가 함께 줄어든다.)")
  return rows


def parse_args():
  parser = argparse.ArgumentParser(description="노이즈 낀 레이더 관측값(거리/방위각/고도각)으로부터 원궤도 요소 결정")
  parser.add_argument("--angle-noise-deg", type=float, default=0.05, help="각도 관측 노이즈 표준편차(도), 기본값: 0.05도")
  parser.add_argument("--range-noise-km", type=float, default=0.5, help="거리 관측 노이즈 표준편차(km), 기본값: 0.5km")
  return parser.parse_args()


def main():
  args = parse_args()

  roundtrip_rows = demo_look_angles_roundtrip()

  true_a, true_i, true_raan = 7000.0, np.radians(53.0), np.radians(80.0)
  site_lat, site_lon = np.radians(35.0), np.radians(129.0)
  window_start, window_end = find_visible_time_window(true_a, true_i, true_raan, site_lat, site_lon)
  times = np.linspace(window_start, window_end, 30)
  rng = np.random.default_rng(7)
  observations = simulate_noisy_observations(true_a, true_i, true_raan, site_lat, site_lon, times,
                                              range_noise_km=args.range_noise_km,
                                              angle_noise_rad=np.radians(args.angle_noise_deg), rng=rng)
  result = determine_circular_orbit(observations, site_lat, site_lon)
  print("\n" + "=" * 70)
  print(f"[2] 작은 노이즈에서 원궤도 요소 복원 검증 (CLI 인자 --angle-noise-deg={args.angle_noise_deg}, "
        f"--range-noise-km={args.range_noise_km} 반영)")
  print("=" * 70)
  a_error_pct = abs(result["semi_major_axis_km"] - true_a) / true_a * 100
  i_error_deg = abs(np.degrees(result["inclination_rad"]) - np.degrees(true_i))
  print(f"복원된 a={result['semi_major_axis_km']:.2f}km(오차 {a_error_pct:.3f}%), "
        f"i={np.degrees(result['inclination_rad']):.2f}도(오차 {i_error_deg:.3f}도)")
  recovery_result = {"true_a": true_a, "true_i_deg": np.degrees(true_i), "true_raan_deg": np.degrees(true_raan),
                      "recovered_a": result["semi_major_axis_km"], "recovered_i_deg": np.degrees(result["inclination_rad"]),
                      "recovered_raan_deg": np.degrees(result["raan_rad"]), "a_error_pct": a_error_pct,
                      "i_error_deg": i_error_deg}

  noise_rows = demo_accuracy_degrades_with_observation_noise()
  num_obs_rows = demo_minimum_observations_needed()

  results_dir = os.path.join(_ROOT_DIR, "results")
  os.makedirs(results_dir, exist_ok=True)

  roundtrip_csv = os.path.join(results_dir, "orbit_determination_roundtrip.csv")
  with open(roundtrip_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "original_r_km", "roundtrip_error_km"])
    for row in roundtrip_rows:
      writer.writerow([row["trial"], f"{row['original_r_km']:.4f}", f"{row['roundtrip_error_km']:.2e}"])
  print(f"\n[기록] 왕복 변환 검증 결과 저장됨 → {roundtrip_csv}")

  recovery_csv = os.path.join(results_dir, "orbit_determination_recovery.csv")
  with open(recovery_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true_a", "true_i_deg", "true_raan_deg", "recovered_a", "recovered_i_deg",
                      "recovered_raan_deg", "a_error_pct", "i_error_deg"])
    writer.writerow([recovery_result["true_a"], f"{recovery_result['true_i_deg']:.4f}",
                      f"{recovery_result['true_raan_deg']:.4f}", f"{recovery_result['recovered_a']:.4f}",
                      f"{recovery_result['recovered_i_deg']:.4f}", f"{recovery_result['recovered_raan_deg']:.4f}",
                      f"{recovery_result['a_error_pct']:.6f}", f"{recovery_result['i_error_deg']:.6f}"])
  print(f"[기록] 궤도요소 복원 결과 저장됨 → {recovery_csv}")

  noise_csv = os.path.join(results_dir, "orbit_determination_noise_sensitivity.csv")
  with open(noise_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["angle_noise_deg", "inclination_error_deg"])
    for row in noise_rows:
      writer.writerow([row["angle_noise_deg"], f"{row['inclination_error_deg']:.6f}"])
  print(f"[기록] 노이즈 민감도 결과 저장됨 → {noise_csv}")

  num_obs_csv = os.path.join(results_dir, "orbit_determination_observation_count.csv")
  with open(num_obs_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["requested_num_obs", "actual_num_obs", "mean_inclination_error_deg", "std_inclination_error_deg"])
    for row in num_obs_rows:
      writer.writerow([row["requested_num_obs"], row["actual_num_obs"],
                        f"{row['mean_inclination_error_deg']:.6f}", f"{row['std_inclination_error_deg']:.6f}"])
  print(f"[기록] 관측 개수별 안정성 결과 저장됨 → {num_obs_csv}")


if __name__ == "__main__":
  main()
