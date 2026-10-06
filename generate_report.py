"""
results/*.csv를 읽어 report_template.html의 플레이스홀더({{...}})를 실제 수치와
그래프 이미지(base64)로 채운 뒤 report.html을 생성한다.

run_all.sh / run_all.ps1의 마지막 단계에서 visualization/visualize_orbits.py 실행
직후 호출된다. CSV가 없는 항목(예: 아직 한 번도 해당 스크립트를 안 돌렸을 때)은
"-"로 표시하고 계속 진행한다. report.html/index.html 생성 후에는 README.md에
직접 박아넣은 대표 그래프 몇 장을 docs/images/에도 동기화한다.
"""

import base64
import csv
import datetime
import os
import re
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEMPLATE_PATH = os.path.join(BASE_DIR, "report_template.html")
OUTPUT_PATH = os.path.join(BASE_DIR, "report.html")
EARTH_MOON_PERIOD_DAYS_FOR_REPORT = 27.321661  # missions/lagrange_points.py의 EARTH_MOON_PERIOD_DAYS와 동일


def read_csv_rows(filename):
  path = os.path.join(RESULTS_DIR, filename)
  if not os.path.exists(path):
    return None
  with open(path, encoding="utf-8", newline="") as f:
    return list(csv.DictReader(f))


def img_to_data_uri(filename):
  path = os.path.join(RESULTS_DIR, filename)
  if not os.path.exists(path):
    return ""
  with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
  return f"data:image/png;base64,{b64}"


def count_pytest_tests():
  """tests/ 폴더의 테스트 개수를 pytest --collect-only로 직접 센다 (하드코딩 방지).
  마지막 "N tests collected" 요약 줄은 pytest 버전/로케일/플러그인에 따라 문구가
  바뀔 수 있어(실제로 메이저 버전 사이에 바뀐 적이 있음) 그 대신 -q 모드에서 수집된
  테스트마다 한 줄씩 찍히는 "path::test_name" 형식의 줄 개수를 직접 센다 — 이쪽이
  훨씬 안정적이다."""
  try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=BASE_DIR, capture_output=True, text=True, timeout=60,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    print(f"[경고] pytest 테스트 수 집계 실패({exc}), TEST_COUNT를 '-'로 표시합니다.")
    return "-"
  collected = [line for line in result.stdout.splitlines() if "::" in line]
  return str(len(collected))


def count_simulation_scripts():
  """run_all.sh에 등록된 시뮬레이션 스크립트 개수를 직접 센다 (하드코딩 방지).
  run_all.sh의 invoke_step 호출 중 pytest/visualize_orbits.py/generate_report.py는
  시뮬레이션 스크립트가 아니라 사전검증/시각화/리포트 단계이므로 제외한다 - 새
  스크립트가 추가될 때마다 이 숫자를 수동으로 맞출 필요가 없도록 run_all.sh 자체를
  소스 오브 트루스로 삼는다."""
  run_all_path = os.path.join(BASE_DIR, "run_all.sh")
  try:
    with open(run_all_path, encoding="utf-8") as f:
      lines = f.readlines()
  except OSError as exc:
    print(f"[경고] run_all.sh 읽기 실패({exc}), SCRIPT_COUNT를 '-'로 표시합니다.")
    return "-"

  excluded = {"pytest", "visualization/visualize_orbits.py", "generate_report.py"}
  count = 0
  for line in lines:
    if not line.startswith("invoke_step "):
      continue
    first_arg = line.split('"')[1]
    if first_arg not in excluded:
      count += 1
  return str(count)


def count_validation_extra_rows():
  """검증표의 숨겨진(접힌) tbody 안 행 개수를 템플릿 자체에서 직접 센다
  (하드코딩 방지) - "나머지 N개 더 보기" 버튼 문구가 실제 숨겨진 행 수와
  항상 일치하도록, 검증 항목이 추가/삭제될 때마다 수동으로 맞출 필요가 없다."""
  try:
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
      html = f.read()
  except OSError as exc:
    print(f"[경고] report_template.html 읽기 실패({exc}), VALIDATION_EXTRA_ROW_COUNT를 '-'로 표시합니다.")
    return "-"

  match = re.search(r'<tbody class="extra-rows" hidden>(.*?)</tbody>', html, re.DOTALL)
  if not match:
    return "0"
  return str(len(re.findall(r"<tr>", match.group(1))))


def compute_kepler_convergence():
  rows = read_csv_rows("kepler_convergence.csv")
  if not rows:
    return dict.fromkeys(["KEPLER_E0_ITERATIONS", "KEPLER_E099_ITERATIONS"], "-")
  by_e = {row["eccentricity"]: row["iterations"] for row in rows}
  return {
      "KEPLER_E0_ITERATIONS": by_e.get("0.0", "-"),
      "KEPLER_E099_ITERATIONS": by_e.get("0.99", "-"),
  }


def compute_conservation():
  """비에너지가 궤도 전체에서 보존된다면 모든 행의 값이 같아야 한다 — 그 최대/최소
  편차를 "보존 오차"로 쓴다. 02번의 데모가 어떤 반장축/GM을 썼는지 이 스크립트가
  다시 알 필요가 없다: CSV 자체가 이미 보존이 성립했는지 판단할 정보를 담고 있다."""
  rows = read_csv_rows("conserved_quantities.csv")
  if not rows:
    return {"CONSERVATION_MAX_ENERGY_ERROR": "-"}
  energies = [float(row["specific_energy"]) for row in rows]
  max_error = max(energies) - min(energies)
  return {"CONSERVATION_MAX_ENERGY_ERROR": f"{max_error:.2e}"}


def compute_rk4_convergence():
  rows = read_csv_rows("rk4_step_size_convergence.csv")
  if not rows or len(rows) < 2:
    return {"RK4_CONVERGENCE_RATIO": "-"}
  errors = [float(row["position_error_km"]) for row in rows]
  ratios = [errors[i - 1] / errors[i] for i in range(1, len(errors)) if errors[i] > 0]
  if not ratios:
    return {"RK4_CONVERGENCE_RATIO": "-"}
  avg_ratio = sum(ratios) / len(ratios)
  return {"RK4_CONVERGENCE_RATIO": f"{avg_ratio:.1f}"}


def compute_zenith_case():
  rows = read_csv_rows("zenith_case.csv")
  if not rows:
    return {"ZENITH_ELEVATION_DEG": "-"}
  return {"ZENITH_ELEVATION_DEG": f"{float(rows[0]['elevation_deg']):.4f}"}


def compute_contact_windows():
  rows = read_csv_rows("contact_windows.csv")
  if not rows:
    return {"CONTACT_WINDOW_COUNT": "0", "CONTACT_MAX_ELEVATION": "-"}
  max_elevation = max(float(row["max_elevation_deg"]) for row in rows)
  return {"CONTACT_WINDOW_COUNT": str(len(rows)), "CONTACT_MAX_ELEVATION": f"{max_elevation:.1f}"}


def compute_hohmann():
  rows = read_csv_rows("hohmann_leo_to_geo.csv")
  if not rows:
    return dict.fromkeys(["HOHMANN_TOTAL_DV", "HOHMANN_TRANSFER_HOURS"], "-")
  row = rows[0]
  return {
      "HOHMANN_TOTAL_DV": f"{float(row['total_delta_v']):.3f}",
      "HOHMANN_TRANSFER_HOURS": f"{float(row['transfer_time_sec']) / 3600:.2f}",
  }


def compute_j2_critical_inclination():
  rows = read_csv_rows("j2_critical_inclination_argp.csv")
  if not rows:
    return {"CRITICAL_INCLINATION_ARGP_RATE": "-"}
  critical_row = min(rows, key=lambda r: abs(float(r["argp_rate_deg_per_day"])))
  return {"CRITICAL_INCLINATION_ARGP_RATE": f"{float(critical_row['argp_rate_deg_per_day']):.6f}"}


def compute_j2_raan_drift():
  rows = read_csv_rows("j2_long_term_raan_drift.csv")
  if not rows:
    return {"RAAN_DRIFT_30DAYS_DEG": "-"}
  first, last = rows[0], rows[-1]
  drift = abs(float(last["raan_deg"]) - float(first["raan_deg"]))
  return {"RAAN_DRIFT_30DAYS_DEG": f"{drift:.2f}"}


def compute_lambert():
  rows = read_csv_rows("lambert_hohmann_cross_check.csv")
  if not rows:
    return {"LAMBERT_CROSS_CHECK_ERROR": "-"}
  return {"LAMBERT_CROSS_CHECK_ERROR": f"{float(rows[0]['error']):.4f}"}


def compute_constellation():
  rows = read_csv_rows("constellation_plane_comparison.csv")
  if not rows:
    return dict.fromkeys(["CONSTELLATION_SINGLE_PLANE_GAP", "CONSTELLATION_MULTI_PLANE_GAP"], "-")
  # "label"은 10번이 콘솔 출력용으로 쓴 한글 문구라 문구가 바뀌면 매칭이 깨진다 —
  # 대신 안정적인 숫자 컬럼인 num_planes로 단일/다중 평면을 구분한다.
  by_num_planes = {int(row["num_planes"]): row for row in rows}
  single = by_num_planes.get(1)
  multi = next((row for planes, row in by_num_planes.items() if planes > 1), None)
  if not single or not multi:
    return dict.fromkeys(["CONSTELLATION_SINGLE_PLANE_GAP", "CONSTELLATION_MULTI_PLANE_GAP"], "-")
  return {
      "CONSTELLATION_SINGLE_PLANE_GAP": f"{float(single['max_gap_min']):.1f}",
      "CONSTELLATION_MULTI_PLANE_GAP": f"{float(multi['max_gap_min']):.1f}",
  }


def compute_isl():
  result = dict.fromkeys(
      ["ISL_SAME_PLANE_BLOCKED_PCT", "ISL_CROSS_PLANE_BLOCKED_PCT", "ISL_OPPOSITE_DISTANCE"], "-")

  same_plane_rows = read_csv_rows("isl_same_plane_visibility.csv")
  if same_plane_rows:
    blocked_count = sum(1 for row in same_plane_rows if row["blocked"] == "True")
    result["ISL_SAME_PLANE_BLOCKED_PCT"] = f"{blocked_count / len(same_plane_rows) * 100:.1f}"

  cross_plane_rows = read_csv_rows("isl_polar_vs_equatorial.csv")
  if cross_plane_rows:
    blocked_count = sum(1 for row in cross_plane_rows if row["blocked"] == "True")
    result["ISL_CROSS_PLANE_BLOCKED_PCT"] = f"{blocked_count / len(cross_plane_rows) * 100:.1f}"

  opposite_rows = read_csv_rows("isl_opposite_side_edge_case.csv")
  if opposite_rows:
    result["ISL_OPPOSITE_DISTANCE"] = f"{float(opposite_rows[0]['closest_distance_km']):.4f}"

  return result


def compute_patched_conic():
  result = dict.fromkeys(
      ["PATCHED_CONIC_TRANSFER_DAYS", "PATCHED_CONIC_TOTAL_DV", "PATCHED_CONIC_SOI_EARTH_PCT"], "-")

  mission_rows = read_csv_rows("patched_conic_mission_delta_v.csv")
  if mission_rows:
    result["PATCHED_CONIC_TRANSFER_DAYS"] = f"{float(mission_rows[0]['transfer_time_days']):.1f}"
    result["PATCHED_CONIC_TOTAL_DV"] = f"{float(mission_rows[0]['total_dv']):.2f}"

  soi_rows = read_csv_rows("patched_conic_soi_comparison.csv")
  if soi_rows:
    result["PATCHED_CONIC_SOI_EARTH_PCT"] = f"{float(soi_rows[0]['earth_ratio_pct']):.2f}"

  return result


def compute_cw_rendezvous():
  result = dict.fromkeys(
      ["CW_RENDEZVOUS_DELTA_V_MS", "CW_APPROX_SMALL_ERROR_PCT", "CW_APPROX_LARGE_ERROR_PCT"], "-")

  rendezvous_rows = read_csv_rows("cw_rendezvous_delta_v.csv")
  if rendezvous_rows:
    result["CW_RENDEZVOUS_DELTA_V_MS"] = f"{float(rendezvous_rows[0]['delta_v_km_s']) * 1000:.2f}"

  approximation_rows = read_csv_rows("cw_approximation_validity.csv")
  if approximation_rows:
    result["CW_APPROX_SMALL_ERROR_PCT"] = f"{float(approximation_rows[0]['relative_error_pct']):.4f}"
    result["CW_APPROX_LARGE_ERROR_PCT"] = f"{float(approximation_rows[-1]['relative_error_pct']):.2f}"

  return result


def compute_orbit_determination():
  result = dict.fromkeys(
      ["OD_ROUNDTRIP_ERROR_KM", "OD_RECOVERY_A_ERROR_PCT", "OD_RECOVERY_I_ERROR_DEG",
       "OD_NOISE_SMALL_ERROR_DEG", "OD_NOISE_LARGE_ERROR_DEG",
       "OD_OBS_COUNT_SMALL_MEAN_DEG", "OD_OBS_COUNT_LARGE_MEAN_DEG"], "-")

  roundtrip_rows = read_csv_rows("orbit_determination_roundtrip.csv")
  if roundtrip_rows:
    max_error = max(float(row["roundtrip_error_km"]) for row in roundtrip_rows)
    result["OD_ROUNDTRIP_ERROR_KM"] = f"{max_error:.2e}"

  recovery_rows = read_csv_rows("orbit_determination_recovery.csv")
  if recovery_rows:
    result["OD_RECOVERY_A_ERROR_PCT"] = f"{float(recovery_rows[0]['a_error_pct']):.4f}"
    result["OD_RECOVERY_I_ERROR_DEG"] = f"{float(recovery_rows[0]['i_error_deg']):.4f}"

  noise_rows = read_csv_rows("orbit_determination_noise_sensitivity.csv")
  if noise_rows:
    result["OD_NOISE_SMALL_ERROR_DEG"] = f"{float(noise_rows[0]['inclination_error_deg']):.4f}"
    result["OD_NOISE_LARGE_ERROR_DEG"] = f"{float(noise_rows[-1]['inclination_error_deg']):.4f}"

  obs_count_rows = read_csv_rows("orbit_determination_observation_count.csv")
  if obs_count_rows:
    result["OD_OBS_COUNT_SMALL_MEAN_DEG"] = f"{float(obs_count_rows[0]['mean_inclination_error_deg']):.4f}"
    result["OD_OBS_COUNT_LARGE_MEAN_DEG"] = f"{float(obs_count_rows[-1]['mean_inclination_error_deg']):.4f}"

  return result


def compute_low_thrust_transfer():
  result = dict.fromkeys(
      ["LOW_THRUST_A_ERROR_PCT", "LOW_THRUST_ECCENTRICITY", "LOW_THRUST_EDELBAUM_ERROR_PCT",
       "LOW_THRUST_TIME_RATIO", "LOW_THRUST_DELTA_V_MS", "LOW_THRUST_HOHMANN_DELTA_V_MS"], "-")

  summary_rows = read_csv_rows("low_thrust_transfer_summary.csv")
  if summary_rows:
    row = summary_rows[0]
    result["LOW_THRUST_A_ERROR_PCT"] = f"{float(row['a_error_pct']):.4f}"
    result["LOW_THRUST_ECCENTRICITY"] = f"{float(row['achieved_e']):.6f}"
    result["LOW_THRUST_EDELBAUM_ERROR_PCT"] = f"{float(row['edelbaum_relative_error_pct']):.2f}"
    result["LOW_THRUST_TIME_RATIO"] = f"{float(row['time_ratio']):.1f}"
    result["LOW_THRUST_DELTA_V_MS"] = f"{float(row['numerical_delta_v_km_s']) * 1000:.2f}"
    result["LOW_THRUST_HOHMANN_DELTA_V_MS"] = f"{float(row['hohmann_delta_v_km_s']) * 1000:.2f}"

  return result


def compute_attitude_dynamics():
  result = dict.fromkeys(
      ["ATTITUDE_ENERGY_DRIFT", "ATTITUDE_MOMENTUM_DRIFT", "ATTITUDE_MINOR_RATIO",
       "ATTITUDE_MAJOR_RATIO", "ATTITUDE_INTERMEDIATE_RATIO"], "-")

  summary_rows = read_csv_rows("attitude_summary.csv")
  if summary_rows:
    row = summary_rows[0]
    result["ATTITUDE_ENERGY_DRIFT"] = row["energy_drift"]
    result["ATTITUDE_MOMENTUM_DRIFT"] = row["momentum_drift"]
    result["ATTITUDE_MINOR_RATIO"] = f"{float(row['minor_axis_ratio']):.1f}"
    result["ATTITUDE_MAJOR_RATIO"] = f"{float(row['major_axis_ratio']):.1f}"
    result["ATTITUDE_INTERMEDIATE_RATIO"] = f"{float(row['intermediate_axis_ratio']):.1f}"

  return result


def compute_station_keeping():
  result = dict.fromkeys(
      ["SK_UNCORRECTED_5YR_ERROR_DEG", "SK_NUM_BURNS_1YR", "SK_BURN_DELTA_V_MS",
       "SK_TOTAL_DELTA_V_MS", "SK_DELTA_V_SPREAD_PCT"], "-")

  drift_rows = read_csv_rows("station_keeping_uncorrected_drift.csv")
  if drift_rows:
    result["SK_UNCORRECTED_5YR_ERROR_DEG"] = f"{float(drift_rows[-1]['raan_error_deg']):.4f}"

  budget_rows = read_csv_rows("station_keeping_delta_v_budget.csv")
  if budget_rows:
    dv_values = [float(row["total_delta_v_ms"]) for row in budget_rows]
    spread_pct = (max(dv_values) - min(dv_values)) / min(dv_values) * 100
    result["SK_DELTA_V_SPREAD_PCT"] = f"{spread_pct:.1f}"
    result["SK_TOTAL_DELTA_V_MS"] = f"{dv_values[len(dv_values) // 2]:.1f}"

  frequency_rows = read_csv_rows("station_keeping_tolerance_vs_frequency.csv")
  if frequency_rows:
    for row in frequency_rows:
      if abs(float(row["tolerance_deg"]) - 0.1) < 1e-6:
        result["SK_NUM_BURNS_1YR"] = row["num_burns"]
        break

  if budget_rows:
    for row in budget_rows:
      if abs(float(row["tolerance_deg"]) - 0.1) < 1e-6 and int(row["num_burns"]) > 0:
        result["SK_BURN_DELTA_V_MS"] = f"{float(row['total_delta_v_ms']) / int(row['num_burns']):.2f}"
        break

  return result


def compute_orbital_decay():
  result = dict.fromkeys(
      ["DECAY_LIFETIME_DAYS", "DECAY_RATE_RATIO", "DECAY_BC_MIN_LIFETIME_DAYS",
       "DECAY_BC_MAX_LIFETIME_DAYS", "DECAY_ALT_MIN_LIFETIME_DAYS", "DECAY_ALT_MAX_LIFETIME_DAYS"], "-")

  summary_rows = read_csv_rows("orbital_decay_summary.csv")
  if summary_rows:
    row = summary_rows[0]
    result["DECAY_LIFETIME_DAYS"] = f"{float(row['lifetime_days']):.2f}"
    result["DECAY_RATE_RATIO"] = f"{float(row['rate_ratio']):.2f}"

  bc_rows = read_csv_rows("orbital_decay_ballistic_coefficient_vs_lifetime.csv")
  if bc_rows:
    result["DECAY_BC_MIN_LIFETIME_DAYS"] = f"{float(bc_rows[0]['lifetime_days']):.2f}"
    result["DECAY_BC_MAX_LIFETIME_DAYS"] = f"{float(bc_rows[-1]['lifetime_days']):.2f}"

  altitude_rows = read_csv_rows("orbital_decay_altitude_vs_lifetime.csv")
  if altitude_rows:
    result["DECAY_ALT_MIN_LIFETIME_DAYS"] = f"{float(altitude_rows[0]['lifetime_days']):.2f}"
    result["DECAY_ALT_MAX_LIFETIME_DAYS"] = f"{float(altitude_rows[-1]['lifetime_days']):.2f}"

  return result


def compute_lagrange_points():
  result = dict.fromkeys(
      ["LAGRANGE_L1_KM", "LAGRANGE_L4_MAX_REAL_EIGENVALUE", "LAGRANGE_L1_MAX_REAL_EIGENVALUE",
       "LAGRANGE_L1_L4_DIVERGENCE_RATIO"], "-")

  positions_rows = read_csv_rows("lagrange_points_positions.csv")
  if positions_rows:
    l1_row = next((row for row in positions_rows if row["point"] == "L1"), None)
    if l1_row:
      result["LAGRANGE_L1_KM"] = f"{float(l1_row['x_km']):,.0f}"

  stability_rows = read_csv_rows("lagrange_points_stability.csv")
  if stability_rows:
    l1_row = next((row for row in stability_rows if row["point"] == "L1"), None)
    l4_row = next((row for row in stability_rows if row["point"] == "L4"), None)
    if l1_row:
      result["LAGRANGE_L1_MAX_REAL_EIGENVALUE"] = f"{float(l1_row['max_real_eigenvalue']):.4f}"
    if l4_row:
      result["LAGRANGE_L4_MAX_REAL_EIGENVALUE"] = f"{float(l4_row['max_real_eigenvalue']):.6f}"

  l4_trajectory_rows = read_csv_rows("lagrange_points_l4_trajectory.csv")
  l1_trajectory_rows = read_csv_rows("lagrange_points_l1_trajectory.csv")
  if l4_trajectory_rows and l1_trajectory_rows:
    (x_l4, y_l4) = (float(l4_trajectory_rows[0]["x"]), float(l4_trajectory_rows[0]["y"]))
    (x_l1, y_l1) = (float(l1_trajectory_rows[0]["x"]), float(l1_trajectory_rows[0]["y"]))
    l4_max_dist = max(((float(r["x"]) - x_l4) ** 2 + (float(r["y"]) - y_l4) ** 2) ** 0.5
                       for r in l4_trajectory_rows)
    l1_max_dist = max(((float(r["x"]) - x_l1) ** 2 + (float(r["y"]) - y_l1) ** 2) ** 0.5
                       for r in l1_trajectory_rows)
    if l4_max_dist > 0:
      result["LAGRANGE_L1_L4_DIVERGENCE_RATIO"] = f"{l1_max_dist / l4_max_dist:.1f}"

  return result


def compute_lyapunov_orbits():
  result = dict.fromkeys(
      ["LYAPUNOV_L1_PERIOD_DAYS", "LYAPUNOV_MAX_DEVIATION_PCT",
       "LYAPUNOV_L1_L2_PERIOD_RATIO"], "-")

  family_rows = read_csv_rows("lyapunov_orbits_family.csv")
  if family_rows:
    l1_rows = [row for row in family_rows if row["libration_point"] == "L1" and row["period"]]
    if l1_rows:
      smallest = min(l1_rows, key=lambda r: float(r["amplitude"]))
      largest = max(l1_rows, key=lambda r: float(r["amplitude"]))
      period_days = float(smallest["period"]) * EARTH_MOON_PERIOD_DAYS_FOR_REPORT / (2 * 3.141592653589793)
      result["LYAPUNOV_L1_PERIOD_DAYS"] = f"{period_days:.2f}"
      linear_prediction = smallest.get("period_linear_prediction", "")
      if linear_prediction:
        linear_val = float(linear_prediction)
        deviation_pct = abs(float(largest["period"]) - linear_val) / linear_val * 100
        result["LYAPUNOV_MAX_DEVIATION_PCT"] = f"{deviation_pct:.3f}"

    l1_l2_rows = [row for row in family_rows if row["libration_point"] in ("L1", "L2") and not row["period_linear_prediction"] and row["period"]]
    l1_compare = next((row for row in l1_l2_rows if row["libration_point"] == "L1"), None)
    l2_compare = next((row for row in l1_l2_rows if row["libration_point"] == "L2"), None)
    if l1_compare and l2_compare:
      ratio = float(l2_compare["period"]) / float(l1_compare["period"])
      result["LYAPUNOV_L1_L2_PERIOD_RATIO"] = f"{ratio:.4f}"

  return result


def compute_pid_attitude_control():
  result = dict.fromkeys(
      ["PID_FINAL_ERROR_DEG", "PID_TUMBLE_UNCONTROLLED_DEG", "PID_TUMBLE_CONTROLLED_DEG",
       "PID_TUMBLE_SUPPRESSION_RATIO"], "-")

  summary_rows = read_csv_rows("pid_summary.csv")
  if summary_rows:
    row = summary_rows[0]
    result["PID_FINAL_ERROR_DEG"] = f"{float(row['final_error_deg']):.3f}"
    uncontrolled = float(row["tumble_max_uncontrolled_deg"])
    controlled = float(row["tumble_max_controlled_deg"])
    result["PID_TUMBLE_UNCONTROLLED_DEG"] = f"{uncontrolled:.1f}"
    result["PID_TUMBLE_CONTROLLED_DEG"] = f"{controlled:.1f}"
    if controlled > 0:
      result["PID_TUMBLE_SUPPRESSION_RATIO"] = f"{uncontrolled / controlled:.1f}"

  return result


def compute_orbit_raise_and_reorient():
  result = dict.fromkeys(
      ["MISSION_ORBIT_DELTA_V_MS", "MISSION_ORBIT_TRANSFER_HOURS",
       "MISSION_ATTITUDE_SETTLING_SEC", "MISSION_ATTITUDE_FINAL_ERROR_DEG",
       "MISSION_TOTAL_TIME_HOURS"], "-")

  rows = read_csv_rows("orbit_raise_and_reorient_handoff.csv")
  if rows:
    row = rows[0]
    result["MISSION_ORBIT_DELTA_V_MS"] = f"{float(row['orbit_delta_v_km_s']) * 1000:.2f}"
    result["MISSION_ORBIT_TRANSFER_HOURS"] = f"{float(row['orbit_transfer_time_sec']) / 3600:.2f}"
    result["MISSION_ATTITUDE_SETTLING_SEC"] = f"{float(row['attitude_settling_time_sec']):.1f}"
    result["MISSION_ATTITUDE_FINAL_ERROR_DEG"] = f"{float(row['attitude_final_error_deg']):.3f}"
    result["MISSION_TOTAL_TIME_HOURS"] = f"{float(row['total_mission_time_sec']) / 3600:.2f}"

  return result


def compute_third_body_perturbation():
  result = dict.fromkeys(
      ["THIRD_BODY_LUNAR_SOLAR_RATIO", "THIRD_BODY_RATIO_AT_LEO_PCT",
       "THIRD_BODY_RATIO_AT_GEO_PCT"], "-")

  magnitude_rows = read_csv_rows("third_body_sun_vs_moon_magnitude.csv")
  if magnitude_rows:
    by_body = {row["body"]: float(row["acceleration_km_s2"]) for row in magnitude_rows}
    result["THIRD_BODY_LUNAR_SOLAR_RATIO"] = f"{by_body['moon'] / by_body['sun']:.2f}"

  altitude_rows = read_csv_rows("third_body_j2_vs_altitude.csv")
  if altitude_rows:
    result["THIRD_BODY_RATIO_AT_LEO_PCT"] = f"{float(altitude_rows[0]['ratio_lunisolar_to_j2']) * 100:.4f}"
    result["THIRD_BODY_RATIO_AT_GEO_PCT"] = f"{float(altitude_rows[-1]['ratio_lunisolar_to_j2']) * 100:.2f}"

  return result


def compute_solar_radiation_pressure():
  result = dict.fromkeys(
      ["SRP_CROSSOVER_ALTITUDE_KM", "SRP_MAGNITUDE_KM_S2", "SRP_ECLIPSE_FRACTION_PCT"], "-")

  altitude_rows = read_csv_rows("solar_radiation_pressure_altitude_vs_drag.csv")
  if altitude_rows:
    for row in altitude_rows:
      if float(row["ratio_srp_to_drag"]) >= 1.0:
        result["SRP_CROSSOVER_ALTITUDE_KM"] = row["altitude_km"]
        break
    result["SRP_MAGNITUDE_KM_S2"] = f"{float(altitude_rows[-1]['srp_accel_km_s2']):.3e}"

  eclipse_rows = read_csv_rows("solar_radiation_pressure_eclipse_trajectory.csv")
  if eclipse_rows:
    in_shadow_count = sum(1 for r in eclipse_rows if r["in_shadow"] == "True")
    result["SRP_ECLIPSE_FRACTION_PCT"] = f"{in_shadow_count / len(eclipse_rows) * 100:.1f}"

  return result


def compute_reaction_wheel_desaturation():
  result = dict.fromkeys(
      ["WHEEL_TIME_TO_SATURATION_SEC", "WHEEL_DESAT_DELTA_V_MM_S",
       "WHEEL_DESAT_VS_ORBIT_RAISE_PCT"], "-")
  rows = read_csv_rows("reaction_wheel_summary.csv")
  if rows:
    row = rows[0]
    result["WHEEL_TIME_TO_SATURATION_SEC"] = f"{float(row['time_to_saturation_sec']):.1f}"
    result["WHEEL_DESAT_DELTA_V_MM_S"] = f"{float(row['desaturation_delta_v_km_s']) * 1e6:.3f}"
    result["WHEEL_DESAT_VS_ORBIT_RAISE_PCT"] = f"{float(row['desaturation_delta_v_km_s']) / float(row['orbit_raise_delta_v_km_s']) * 100:.4f}"
  return result


def main():
  with open(TEMPLATE_PATH, encoding="utf-8") as f:
    html = f.read()

  values = {
      "RUN_DATE": datetime.date.today().isoformat(),
      "TEST_COUNT": count_pytest_tests(),
      "SCRIPT_COUNT": count_simulation_scripts(),
      "VALIDATION_EXTRA_ROW_COUNT": count_validation_extra_rows(),
      "IMG_ORBIT_SHAPE": img_to_data_uri("01_kepler_orbit_shape.png"),
      "IMG_SECOND_LAW": img_to_data_uri("01_kepler_second_law_areas.png"),
      "IMG_ZENITH": img_to_data_uri("04_zenith_and_horizon_cases.png"),
      "IMG_ELEVATION": img_to_data_uri("05_elevation_over_time.png"),
      "IMG_CONTACT_GANTT": img_to_data_uri("05_contact_windows_gantt.png"),
      "IMG_HOHMANN_ORBIT": img_to_data_uri("06_hohmann_transfer_orbit.png"),
      "IMG_HOHMANN_RATIO": img_to_data_uri("06_hohmann_delta_v_vs_ratio.png"),
      "IMG_J2_RAAN": img_to_data_uri("07_j2_raan_precession.png"),
      "IMG_J2_DRIFT": img_to_data_uri("07_j2_ground_track_drift.png"),
      "IMG_LAMBERT_SHORT_LONG": img_to_data_uri("09_lambert_short_vs_long_way.png"),
      "IMG_CONSTELLATION_SIZE": img_to_data_uri("10_constellation_size_vs_gap.png"),
      "IMG_CONSTELLATION_PLANE": img_to_data_uri("10_constellation_plane_comparison.png"),
      "IMG_ISL_COMPARISON": img_to_data_uri("11_isl_visibility_comparison.png"),
      "IMG_PATCHED_CONIC_ORBIT": img_to_data_uri("12_patched_conic_transfer_orbit.png"),
      "IMG_PATCHED_CONIC_DV": img_to_data_uri("12_patched_conic_delta_v_breakdown.png"),
      "IMG_CW_ZERO_DRIFT": img_to_data_uri("13_cw_zero_drift_comparison.png"),
      "IMG_CW_APPROXIMATION": img_to_data_uri("13_cw_approximation_validity.png"),
      "IMG_OD_NOISE_SENSITIVITY": img_to_data_uri("14_orbit_determination_noise_sensitivity.png"),
      "IMG_OD_OBSERVATION_COUNT": img_to_data_uri("14_orbit_determination_observation_count.png"),
      "IMG_LOW_THRUST_SPIRAL": img_to_data_uri("15_low_thrust_spiral_trajectory.png"),
      "IMG_LOW_THRUST_TIME_VS_THRUST": img_to_data_uri("15_low_thrust_transfer_time_vs_thrust.png"),
      "IMG_ATTITUDE_INSTABILITY": img_to_data_uri("16_attitude_intermediate_axis_instability.png"),
      "IMG_ATTITUDE_GROWTH_COMPARISON": img_to_data_uri("16_attitude_perturbation_growth_comparison.png"),
      "IMG_SK_SAWTOOTH": img_to_data_uri("17_station_keeping_raan_sawtooth.png"),
      "IMG_SK_DELTA_V_BUDGET": img_to_data_uri("17_station_keeping_delta_v_budget.png"),
      "IMG_DECAY_TRAJECTORY": img_to_data_uri("18_orbital_decay_trajectory.png"),
      "IMG_DECAY_LIFETIME": img_to_data_uri("18_orbital_decay_lifetime_comparison.png"),
      "IMG_LAGRANGE_LAYOUT": img_to_data_uri("19_lagrange_points_layout.png"),
      "IMG_LAGRANGE_STABILITY": img_to_data_uri("19_lagrange_points_stability_trajectories.png"),
      "IMG_LYAPUNOV_TRAJECTORY": img_to_data_uri("20_lyapunov_orbit_trajectory.png"),
      "IMG_LYAPUNOV_AMPLITUDE_PERIOD": img_to_data_uri("20_lyapunov_amplitude_vs_period.png"),
      "IMG_PID_POINT_AND_HOLD": img_to_data_uri("21_pid_point_and_hold_error.png"),
      "IMG_PID_TUMBLE_COMPARISON": img_to_data_uri("21_pid_intermediate_axis_comparison.png"),
      "IMG_MISSION_TIMELINE": img_to_data_uri("22_mission_timeline_combined.png"),
      "IMG_THIRD_BODY_CROSSOVER": img_to_data_uri("23_third_body_j2_crossover.png"),
      "IMG_THIRD_BODY_OSCILLATION": img_to_data_uri("23_third_body_oscillation.png"),
      "IMG_SRP_CROSSOVER": img_to_data_uri("24_srp_drag_crossover.png"),
      "IMG_SRP_ECLIPSE": img_to_data_uri("24_srp_eclipse_timeseries.png"),
      "IMG_WHEEL_SATURATION": img_to_data_uri("25_reaction_wheel_saturation_momentum.png"),
      "IMG_WHEEL_POINTING_ERROR": img_to_data_uri("25_reaction_wheel_pointing_error.png"),
  }
  values.update(compute_kepler_convergence())
  values.update(compute_conservation())
  values.update(compute_rk4_convergence())
  values.update(compute_zenith_case())
  values.update(compute_contact_windows())
  values.update(compute_hohmann())
  values.update(compute_j2_critical_inclination())
  values.update(compute_j2_raan_drift())
  values.update(compute_lambert())
  values.update(compute_constellation())
  values.update(compute_isl())
  values.update(compute_patched_conic())
  values.update(compute_cw_rendezvous())
  values.update(compute_orbit_determination())
  values.update(compute_low_thrust_transfer())
  values.update(compute_attitude_dynamics())
  values.update(compute_station_keeping())
  values.update(compute_orbital_decay())
  values.update(compute_lagrange_points())
  values.update(compute_lyapunov_orbits())
  values.update(compute_pid_attitude_control())
  values.update(compute_orbit_raise_and_reorient())
  values.update(compute_third_body_perturbation())
  values.update(compute_solar_radiation_pressure())
  values.update(compute_reaction_wheel_desaturation())

  for key, val in values.items():
    token = "{{" + key + "}}"
    if token in html:
      html = html.replace(token, val)
  # 아직 스크립트를 안 돌려서 CSV가 없는 경우는 각 compute_*() 함수가 이미 "-"로 채워
  # 넣으므로, 여기서 남는 {{PLACEHOLDER}}는 그런 정상적인 경우가 아니라 템플릿에 새
  # 플레이스홀더를 추가했는데 values 딕셔너리에 그 키를 안 넣었거나 오타를 낸 경우뿐이다
  # — 경고만 찍고 넘어가면 report.html에 리터럴 {{PLACEHOLDER}} 텍스트가 그대로 박힌 채
  # exit code 0으로 끝나므로, 빌드를 실패시켜 CI에서 잡히게 한다.
  missing = [line.strip() for line in html.splitlines() if "{{" in line and "}}" in line]
  if missing:
    print("[오류] 채워지지 않은 플레이스홀더가 남아있습니다 (템플릿과 values 딕셔너리 불일치):")
    for line in missing:
      print("  ", line)
    raise SystemExit(1)

  with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

  # GitHub Pages는 저장소 루트의 index.html을 자동으로 서빙한다 — report.html과
  # 똑같은 내용을 index.html로도 저장해, 매 실행마다 별도 동기화 없이 GitHub Pages
  # 배포본이 항상 최신 report.html과 같은 상태를 유지하게 한다.
  index_path = os.path.join(BASE_DIR, "index.html")
  with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)

  print(f"[완료] report.html 생성됨 ({len(html):,}자) → {OUTPUT_PATH}")
  print(f"[완료] index.html 동기화됨 (GitHub Pages용) → {index_path}")

  sync_representative_images()


# README.md에 <img>로 직접 박아넣은 대표 그래프들 — results/는 .gitignore 대상이라
# 저장소에 커밋되지 않으므로, README에서 깨지지 않고 보이려면 이 별도 경로에
# 실제 파일로 커밋되어 있어야 한다. 목록은 README.md의 "주요 결과" 표와 일치시킨다.
REPRESENTATIVE_IMAGES = [
    "01_kepler_orbit_shape.png",
    "06_hohmann_transfer_orbit.png",
    "07_j2_raan_precession.png",
    "10_constellation_plane_comparison.png",
    "13_cw_approximation_validity.png",
    "14_orbit_determination_observation_count.png",
]


def sync_representative_images():
  docs_images_dir = os.path.join(BASE_DIR, "docs", "images")
  os.makedirs(docs_images_dir, exist_ok=True)
  for filename in REPRESENTATIVE_IMAGES:
    src = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(src):
      print(f"[건너뜀] {src} 없음 — README 대표 그래프 동기화에서 제외")
      continue
    dst = os.path.join(docs_images_dir, filename)
    with open(src, "rb") as f_src, open(dst, "wb") as f_dst:
      f_dst.write(f_src.read())
  print(f"[완료] README 대표 그래프 {len(REPRESENTATIVE_IMAGES)}개 동기화됨 → {docs_images_dir}")


if __name__ == "__main__":
  main()
