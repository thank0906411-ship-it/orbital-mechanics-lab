"""
propagation/kepler_orbit_propagation.py, frames/coordinate_frame_transforms.py,
missions/ground_station_visibility.py, missions/hohmann_transfer.py,
perturbations/j2_perturbation.py, missions/lambert_problem.py,
constellations/constellation_coverage.py,
constellations/intersatellite_link_visibility.py,
missions/patched_conic_interplanetary.py, missions/clohessy_wiltshire_rendezvous.py,
missions/orbit_determination.py, missions/low_thrust_transfer.py,
attitude/torque_free_rigid_body.py, missions/station_keeping.py,
missions/orbital_decay.py, missions/lagrange_points.py, missions/lyapunov_orbits.py,
attitude/pid_attitude_control.py, missions/orbit_raise_and_reorient.py,
perturbations/third_body_perturbation.py, perturbations/solar_radiation_pressure.py,
attitude/reaction_wheel_desaturation.py 시뮬레이션이 남긴 결과 CSV를 그래프로
그려주는 도구. 시뮬레이션 코드가 아니라 "결과를 눈으로 보기 위한" 별도 스크립트다.

실행 전에 먼저 위 22개 스크립트를 한 번 이상 실행해서 results/ 폴더에 CSV가
생성되어 있어야 한다. (해당 CSV가 없는 항목은 건너뛰고 나머지만 그린다.)

실행: python visualization/visualize_orbits.py
출력 파일명은 어느 스크립트가 만든 결과인지 한눈에 알 수 있도록 원래 번호 체계
(01~25, 스크립트 자체 파일명에서는 빠졌지만 결과물 파일명에는 남겨둠)를
접두사로 붙인다(예: 01_kepler_orbit_shape.png는 케플러 전파 스크립트의 결과):
      results/01_kepler_orbit_shape.png, results/01_kepler_second_law_areas.png,
      results/04_zenith_and_horizon_cases.png, results/05_elevation_over_time.png,
      results/05_contact_windows_gantt.png, results/06_hohmann_transfer_orbit.png,
      results/06_hohmann_delta_v_vs_ratio.png, results/07_j2_raan_precession.png,
      results/07_j2_ground_track_drift.png, results/09_lambert_short_vs_long_way.png,
      results/10_constellation_size_vs_gap.png, results/10_constellation_plane_comparison.png,
      results/11_isl_visibility_comparison.png, results/12_patched_conic_transfer_orbit.png,
      results/12_patched_conic_delta_v_breakdown.png, results/13_cw_zero_drift_comparison.png,
      results/13_cw_approximation_validity.png, results/14_orbit_determination_noise_sensitivity.png,
      results/14_orbit_determination_observation_count.png, results/15_low_thrust_spiral_trajectory.png,
      results/15_low_thrust_transfer_time_vs_thrust.png, results/16_attitude_intermediate_axis_instability.png,
      results/16_attitude_perturbation_growth_comparison.png, results/17_station_keeping_raan_sawtooth.png,
      results/17_station_keeping_delta_v_budget.png, results/18_orbital_decay_trajectory.png,
      results/18_orbital_decay_lifetime_comparison.png, results/19_lagrange_points_layout.png,
      results/19_lagrange_points_stability_trajectories.png, results/20_lyapunov_orbit_trajectory.png,
      results/20_lyapunov_amplitude_vs_period.png, results/21_pid_point_and_hold_error.png,
      results/21_pid_intermediate_axis_comparison.png, results/22_mission_timeline_combined.png,
      results/23_third_body_j2_crossover.png, results/23_third_body_oscillation.png,
      results/24_srp_drag_crossover.png, results/24_srp_eclipse_timeseries.png,
      results/25_reaction_wheel_saturation_momentum.png, results/25_reaction_wheel_pointing_error.png

참고: 이 스크립트가 만드는 그래프(축/제목/범례 라벨)는 의도적으로 영문으로 표기한다.
      나머지 콘솔 로그/주석은 한글이다.
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
  sys.stderr.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경(서버, CI)에서도 안전하게 PNG로 저장하기 위함
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 프로젝션 등록을 위해 필요)

# 기본 폰트 크기(10pt)로 저장하면 report.html에서 그래프가 실제 표시 크기보다
# 훨씬 크게 눌려 나와(그래프 이미지 자체는 크지만 화면에는 축소돼 보임) 축/제목
# 글자가 상대적으로 작아 잘 안 보인다 — 전역 rcParams로 폰트 크기를 일괄 확대해
# 27개 그래프 함수를 개별로 건드리지 않고 한 번에 해결한다.
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_THIS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def plot_kepler_orbit_shape():
  csv_path = os.path.join(RESULTS_DIR, "kepler_position_time_series.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 propagation/kepler_orbit_propagation.py를 실행하세요.")
    return

  xs, ys = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      xs.append(float(row["x_p_km"]))
      ys.append(float(row["y_p_km"]))

  fig, ax = plt.subplots(figsize=(8, 7))
  ax.plot(xs, ys, color="tab:blue", linewidth=1.5)
  ax.scatter([0], [0], color="tab:orange", s=100, marker="*", zorder=3, label="Focus (Earth)")
  ax.set_xlabel("Perifocal x (km)")
  ax.set_ylabel("Perifocal y (km)")
  ax.set_title("Kepler orbit propagation: orbital shape in the perifocal plane")
  ax.set_aspect("equal")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "01_kepler_orbit_shape.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_kepler_second_law_areas():
  csv_path = os.path.join(RESULTS_DIR, "circular_vs_elliptical.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 propagation/kepler_orbit_propagation.py를 실행하세요.")
    return

  # "orbit" 컬럼은 01번이 콘솔 출력용으로 쓴 한글 문구라 문구가 조금만 바뀌어도
  # 매칭이 깨진다 — 대신 안정적인 숫자 컬럼인 eccentricity로 원궤도/타원궤도를 구분한다.
  by_eccentricity = {}
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      e = float(row["eccentricity"])
      by_eccentricity.setdefault(e, {"fractions": [], "true_anomalies": []})
      by_eccentricity[e]["fractions"].append(float(row["time_fraction"]))
      by_eccentricity[e]["true_anomalies"].append(float(row["true_anomaly_deg"]))

  fig, ax = plt.subplots(figsize=(9, 5))
  for e, data in sorted(by_eccentricity.items()):
    label = "Circular" if e == 0.0 else "Elliptical"
    ax.plot(data["fractions"], data["true_anomalies"], marker="o", linewidth=2,
            label=f"{label} (e={e:.1f})")
  ax.set_xlabel("Time (fraction of orbital period)")
  ax.set_ylabel("True anomaly (deg)")
  ax.set_title("Kepler's 2nd law: elliptical orbit sweeps angle faster near perigee")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "01_kepler_second_law_areas.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_zenith_and_horizon_cases():
  horizon_path = os.path.join(RESULTS_DIR, "horizon_cases.csv")
  if not os.path.exists(horizon_path):
    print(f"[건너뜀] {horizon_path} 없음 — 먼저 frames/coordinate_frame_transforms.py를 실행하세요.")
    return

  # "label"은 04번이 콘솔 출력용으로 쓴 한글 문구라 문구가 바뀌면 매칭이 깨진다 —
  # 대신 이 그래프가 실제로 보여주려는 것 자체인 고도각의 부호로 라벨을 새로 만든다.
  labels, elevations = [], []
  with open(horizon_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      elevation = float(row["elevation_deg"])
      labels.append("Above horizon\n(visible)" if elevation > 0 else "Below horizon\n(not visible)")
      elevations.append(elevation)

  fig, ax = plt.subplots(figsize=(8, 5))
  colors = ["tab:green" if e > 0 else "tab:red" for e in elevations]
  bars = ax.bar(labels, elevations, color=colors, alpha=0.85)
  ax.axhline(0, color="black", linewidth=1)
  ax.set_ylabel("Elevation (deg)")
  ax.set_title("Elevation sign determines visibility: above vs below the horizon")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, e in zip(bars, elevations):
    ax.text(bar.get_x() + bar.get_width() / 2, e + (2 if e > 0 else -4), f"{e:.1f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "04_zenith_and_horizon_cases.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_elevation_over_time():
  csv_path = os.path.join(RESULTS_DIR, "elevation_time_series.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/ground_station_visibility.py를 실행하세요.")
    return

  times_min, elevations = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times_min.append(float(row["t_sec"]) / 60)
      elevations.append(float(row["elevation_deg"]))

  fig, ax = plt.subplots(figsize=(11, 5))
  ax.plot(times_min, elevations, color="tab:blue", linewidth=1.2)
  ax.axhline(10.0, color="tab:red", linestyle="--", linewidth=1.5, label="Min elevation (10 deg)")
  ax.fill_between(times_min, elevations, 10.0, where=[e > 10.0 for e in elevations],
                   color="tab:green", alpha=0.3, label="Contact window")
  ax.axhline(0.0, color="black", linewidth=0.8)
  ax.set_xlabel("Time (min)")
  ax.set_ylabel("Elevation (deg)")
  ax.set_title("Ground station elevation over time: real orbit propagation, not a square-wave approximation")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "05_elevation_over_time.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_contact_windows_gantt():
  csv_path = os.path.join(RESULTS_DIR, "contact_windows.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/ground_station_visibility.py를 실행하세요.")
    return

  windows = []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      windows.append((float(row["aos_sec"]) / 60, float(row["duration_sec"]) / 60, float(row["max_elevation_deg"])))

  if not windows:
    print(f"[건너뜀] {csv_path}에 접촉 창이 없음")
    return

  fig, ax = plt.subplots(figsize=(11, 4))
  for idx, (aos_min, duration_min, max_el) in enumerate(windows):
    ax.barh(idx, duration_min, left=aos_min, color="tab:blue", alpha=0.85)
    # 짧은 패스는 막대 안에 라벨이 다 안 들어가 잘리므로, 막대 바로 오른쪽에
    # 항상 폭과 무관하게 안전하게 보이도록 쓴다(막대 안 흰 글씨 대신).
    ax.text(aos_min + duration_min + 3, idx, f"max El={max_el:.0f}°", ha="left", va="center",
            color="black", fontsize=11)
  ax.set_yticks(range(len(windows)))
  ax.set_yticklabels([f"Pass {i + 1}" for i in range(len(windows))])
  ax.set_xlabel("Time (min)")
  ax.set_title("Contact windows (AOS-LOS) from real orbit propagation")
  ax.grid(True, axis="x", alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "05_contact_windows_gantt.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_hohmann_transfer_orbit():
  csv_path = os.path.join(RESULTS_DIR, "hohmann_propagated_verification.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/hohmann_transfer.py를 실행하세요.")
    return

  with open(csv_path, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
  r1, r2 = float(row["r1_km"]), float(row["r2_km"])
  a_t = (r1 + r2) / 2
  e_t = (r2 - r1) / (r2 + r1)

  theta = np.linspace(0, 2 * np.pi, 400)
  r_transfer = a_t * (1 - e_t ** 2) / (1 + e_t * np.cos(theta))
  x_transfer, y_transfer = r_transfer * np.cos(theta), r_transfer * np.sin(theta)

  fig, ax = plt.subplots(figsize=(8, 7))
  circle1 = plt.Circle((0, 0), r1, fill=False, color="tab:green", linewidth=2, label=f"Departure orbit (r={r1:.0f}km)")
  circle2 = plt.Circle((0, 0), r2, fill=False, color="tab:red", linewidth=2, label=f"Target orbit (r={r2:.0f}km)")
  ax.add_patch(circle1)
  ax.add_patch(circle2)
  ax.plot(x_transfer, y_transfer, color="tab:blue", linewidth=1.5, linestyle="--", label="Transfer ellipse")
  ax.scatter([0], [0], color="black", s=60, marker="*", zorder=3, label="Focus (Earth)")
  ax.set_xlabel("x (km)")
  ax.set_ylabel("y (km)")
  ax.set_title("Hohmann transfer: departure circle, transfer ellipse, target circle")
  ax.set_aspect("equal")
  ax.legend(fontsize=11)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "06_hohmann_transfer_orbit.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_hohmann_delta_v_vs_ratio():
  csv_path = os.path.join(RESULTS_DIR, "hohmann_delta_v_vs_ratio.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/hohmann_transfer.py를 실행하세요.")
    return

  ratios, totals = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      ratios.append(float(row["ratio"]))
      totals.append(float(row["total_delta_v"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(ratios, totals, marker="o", linewidth=2, color="tab:purple")
  ax.set_xlabel("Target/departure radius ratio (r2/r1)")
  ax.set_ylabel("Total delta-V (km/s)")
  ax.set_title("Hohmann transfer delta-V vs radius ratio")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "06_hohmann_delta_v_vs_ratio.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_j2_raan_precession():
  csv_path = os.path.join(RESULTS_DIR, "j2_raan_precession_by_inclination.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 perturbations/j2_perturbation.py를 실행하세요.")
    return

  inclinations, rates = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      inclinations.append(float(row["inclination_deg"]))
      rates.append(float(row["raan_rate_deg_per_day"]))

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.plot(inclinations, rates, marker="o", linewidth=2, color="tab:blue")
  ax.axhline(0, color="black", linewidth=1)
  ax.axvline(90, color="tab:gray", linestyle=":", label="Polar orbit (90 deg): zero precession")
  ax.set_xlabel("Inclination (deg)")
  ax.set_ylabel("RAAN precession rate (deg/day)")
  ax.set_title("J2 perturbation: RAAN precession direction flips at 90 deg inclination")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "07_j2_raan_precession.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_j2_ground_track_drift():
  csv_path = os.path.join(RESULTS_DIR, "j2_long_term_raan_drift.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 perturbations/j2_perturbation.py를 실행하세요.")
    return

  days, raans = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      days.append(float(row["day"]))
      raans.append(float(row["raan_deg"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(days, raans, marker="o", linewidth=2, color="tab:orange")
  ax.set_xlabel("Elapsed time (days)")
  ax.set_ylabel("RAAN (deg)")
  ax.set_title("J2 perturbation: long-term RAAN drift accumulates linearly")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "07_j2_ground_track_drift.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_lambert_short_vs_long_way():
  csv_path = os.path.join(RESULTS_DIR, "lambert_short_vs_long_way.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/lambert_problem.py를 실행하세요.")
    return

  labels_en = {"짧은 길(prograde)": "Short way\n(prograde)", "긴 길(retrograde 선택)": "Long way\n(retrograde)"}
  labels, v1_values = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      labels.append(labels_en.get(row["label"], row["label"]))
      v1_values.append(float(row["v1_km_s"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  bars = ax.bar(labels, v1_values, color=["tab:blue", "tab:red"], alpha=0.85)
  ax.set_ylabel("Departure speed |v1| (km/s)")
  ax.set_title("Lambert's problem: short way vs long way require different speeds")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, v1_values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "09_lambert_short_vs_long_way.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_constellation_size_vs_gap():
  csv_path = os.path.join(RESULTS_DIR, "constellation_size_vs_gap.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 constellations/constellation_coverage.py를 실행하세요.")
    return

  num_sats, max_gaps = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      num_sats.append(int(row["num_satellites"]))
      max_gaps.append(float(row["max_gap_min"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(num_sats, max_gaps, marker="o", linewidth=2, color="tab:green")
  ax.set_xlabel("Number of satellites in constellation")
  ax.set_ylabel("Max coverage gap (min)")
  ax.set_title("Constellation size vs max coverage gap at one ground site")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "10_constellation_size_vs_gap.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_constellation_plane_comparison():
  csv_path = os.path.join(RESULTS_DIR, "constellation_plane_comparison.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 constellations/constellation_coverage.py를 실행하세요.")
    return

  labels_en = {"단일 평면(1개 평면에 12기)": "Single plane\n(12 sats, 1 plane)",
               "다중 평면(3개 평면에 4기씩)": "Multi-plane\n(12 sats, 3 planes)"}
  labels, max_gaps = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      labels.append(labels_en.get(row["label"], row["label"]))
      max_gaps.append(float(row["max_gap_min"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  bars = ax.bar(labels, max_gaps, color=["tab:blue", "tab:orange"], alpha=0.85)
  ax.set_ylabel("Max coverage gap (min)")
  ax.set_title("Single plane can beat multi-plane for one fixed ground site\n(counter to the 'spread planes' intuition)")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, max_gaps):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "10_constellation_plane_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_isl_visibility_comparison():
  same_plane_path = os.path.join(RESULTS_DIR, "isl_same_plane_visibility.csv")
  cross_plane_path = os.path.join(RESULTS_DIR, "isl_polar_vs_equatorial.csv")
  if not os.path.exists(same_plane_path) or not os.path.exists(cross_plane_path):
    print(f"[건너뜀] {same_plane_path} 또는 {cross_plane_path} 없음 — "
          "먼저 constellations/intersatellite_link_visibility.py를 실행하세요.")
    return

  def read_series(path):
    times_min, blocked = [], []
    with open(path, newline="", encoding="utf-8") as f:
      for row in csv.DictReader(f):
        times_min.append(float(row["t_sec"]) / 60)
        blocked.append(row["blocked"] == "True")
    return times_min, blocked

  same_times, same_blocked = read_series(same_plane_path)
  cross_times, cross_blocked = read_series(cross_plane_path)

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

  ax1.fill_between(same_times, 0, 1, where=same_blocked, color="tab:red", alpha=0.6, step="post", label="Blocked")
  ax1.fill_between(same_times, 0, 1, where=[not b for b in same_blocked], color="tab:green", alpha=0.4,
                    step="post", label="Visible")
  ax1.set_ylim(0, 1)
  ax1.set_yticks([])
  ax1.set_title("ISL visibility: same orbital plane (30 deg phase offset)")
  ax1.legend(loc="upper right", fontsize=10)

  ax2.fill_between(cross_times, 0, 1, where=cross_blocked, color="tab:red", alpha=0.6, step="post", label="Blocked")
  ax2.fill_between(cross_times, 0, 1, where=[not b for b in cross_blocked], color="tab:green", alpha=0.4,
                    step="post", label="Visible")
  ax2.set_ylim(0, 1)
  ax2.set_yticks([])
  ax2.set_xlabel("Time (min)")
  ax2.set_title("ISL visibility: polar vs equatorial planes")
  ax2.legend(loc="upper right", fontsize=10)

  fig.suptitle("Inter-satellite link visibility: same plane vs crossing planes")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "11_isl_visibility_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_patched_conic_transfer_orbit():
  csv_path = os.path.join(RESULTS_DIR, "patched_conic_heliocentric_propagation.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/patched_conic_interplanetary.py를 실행하세요.")
    return

  AU_KM = 1.495978707e8
  with open(csv_path, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
  a_t, e_t = float(row["a_t_km"]), float(row["e_t"])
  r_earth = a_t * (1 - e_t)
  r_mars = a_t * (1 + e_t)

  theta = np.linspace(0, 2 * np.pi, 400)
  r_transfer = a_t * (1 - e_t ** 2) / (1 + e_t * np.cos(theta))
  x_transfer, y_transfer = r_transfer * np.cos(theta) / AU_KM, r_transfer * np.sin(theta) / AU_KM

  fig, ax = plt.subplots(figsize=(7, 7))
  circle_earth = plt.Circle((0, 0), r_earth / AU_KM, fill=False, color="tab:blue", linewidth=2, label="Earth orbit (1 AU)")
  circle_mars = plt.Circle((0, 0), r_mars / AU_KM, fill=False, color="tab:red", linewidth=2, label="Mars orbit (1.52 AU)")
  ax.add_patch(circle_earth)
  ax.add_patch(circle_mars)
  ax.plot(x_transfer, y_transfer, color="tab:purple", linewidth=1.5, linestyle="--", label="Heliocentric transfer ellipse")
  ax.scatter([0], [0], color="gold", s=150, marker="*", zorder=3, label="Sun")
  ax.set_xlabel("x (AU)")
  ax.set_ylabel("y (AU)")
  ax.set_title("Patched conic: Earth-to-Mars heliocentric transfer orbit")
  ax.set_aspect("equal")
  ax.legend(fontsize=11)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "12_patched_conic_transfer_orbit.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_patched_conic_delta_v_breakdown():
  csv_path = os.path.join(RESULTS_DIR, "patched_conic_mission_delta_v.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/patched_conic_interplanetary.py를 실행하세요.")
    return

  with open(csv_path, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
  dv_depart = float(row["dv_depart"])
  dv_capture = float(row["dv_capture"])
  total_dv = float(row["total_dv"])

  fig, ax = plt.subplots(figsize=(8, 5))
  labels = ["Earth departure\n(hyperbolic escape)", "Mars capture\n(hyperbolic capture)"]
  values = [dv_depart, dv_capture]
  bars = ax.bar(labels, values, color=["tab:blue", "tab:red"], alpha=0.85)
  ax.set_ylabel("Delta-V (km/s)")
  ax.set_title(f"Earth-to-Mars mission delta-V breakdown (total: {total_dv:.2f} km/s)")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "12_patched_conic_delta_v_breakdown.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_cw_zero_drift_comparison():
  csv_path = os.path.join(RESULTS_DIR, "cw_zero_drift_comparison.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/clohessy_wiltshire_rendezvous.py를 실행하세요.")
    return

  num_orbits, y_zero_drift, y_no_correction = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      num_orbits.append(int(row["num_orbits"]))
      y_zero_drift.append(float(row["y_zero_drift_km"]))
      y_no_correction.append(float(row["y_no_correction_km"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(num_orbits, y_zero_drift, marker="o", linewidth=2, color="tab:green", label="Zero-drift velocity")
  ax.plot(num_orbits, y_no_correction, marker="s", linewidth=2, color="tab:red", label="No correction (vy0=0)")
  ax.axhline(0, color="black", linewidth=0.8)
  ax.set_xlabel("Number of target orbits elapsed")
  ax.set_ylabel("Along-track offset y (km)")
  ax.set_title("CW zero-drift condition prevents secular divergence")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "13_cw_zero_drift_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_cw_approximation_validity():
  csv_path = os.path.join(RESULTS_DIR, "cw_approximation_validity.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/clohessy_wiltshire_rendezvous.py를 실행하세요.")
    return

  x0_values, errors = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      x0_values.append(float(row["x0_km"]))
      errors.append(float(row["relative_error_pct"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(x0_values, errors, marker="o", linewidth=2, color="tab:purple")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Initial radial offset x0 (km, log scale)")
  ax.set_ylabel("CW vs nonlinear propagation error (%, log scale)")
  ax.set_title("CW linearization error grows with separation distance")
  ax.grid(True, which="both", alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "13_cw_approximation_validity.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_orbit_determination_noise_sensitivity():
  csv_path = os.path.join(RESULTS_DIR, "orbit_determination_noise_sensitivity.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/orbit_determination.py를 실행하세요.")
    return

  angle_noise_deg, inclination_error_deg = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      angle_noise_deg.append(float(row["angle_noise_deg"]))
      inclination_error_deg.append(float(row["inclination_error_deg"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(angle_noise_deg, inclination_error_deg, marker="o", linewidth=2, color="tab:orange")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Observation angle noise std dev (deg, log scale)")
  ax.set_ylabel("Recovered inclination error (deg, log scale)")
  ax.set_title("Orbit determination accuracy degrades with observation noise")
  ax.grid(True, which="both", alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "14_orbit_determination_noise_sensitivity.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_orbit_determination_observation_count():
  csv_path = os.path.join(RESULTS_DIR, "orbit_determination_observation_count.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/orbit_determination.py를 실행하세요.")
    return

  num_obs, mean_error, std_error = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      num_obs.append(int(row["actual_num_obs"]))
      mean_error.append(float(row["mean_inclination_error_deg"]))
      std_error.append(float(row["std_inclination_error_deg"]))

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.errorbar(num_obs, mean_error, yerr=std_error, marker="o", linewidth=2, capsize=5, color="tab:blue")
  ax.set_xlabel("Number of observations used")
  ax.set_ylabel("Mean inclination error over trials (deg)")
  ax.set_title("More observations stabilize orbit determination (error bars = std dev)")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "14_orbit_determination_observation_count.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_low_thrust_spiral_trajectory():
  csv_path = os.path.join(RESULTS_DIR, "low_thrust_spiral_trajectory.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/low_thrust_transfer.py를 실행하세요.")
    return

  xs, ys = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      xs.append(float(row["x_km"]))
      ys.append(float(row["y_km"]))

  fig, ax = plt.subplots(figsize=(9, 7))
  ax.plot(xs, ys, color="tab:purple", linewidth=1.0)
  ax.scatter([xs[0]], [ys[0]], color="tab:green", s=80, zorder=3, label="Start (circular orbit)")
  ax.scatter([xs[-1]], [ys[-1]], color="tab:red", s=80, zorder=3, label="End (target orbit)")
  ax.scatter([0], [0], color="black", s=100, marker="*", zorder=3, label="Focus (Earth)")
  ax.set_xlabel("x (km)")
  ax.set_ylabel("y (km)")
  ax.set_title("Low-thrust spiral: continuous tangential thrust slowly raises a circular orbit")
  ax.set_aspect("equal")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "15_low_thrust_spiral_trajectory.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_low_thrust_transfer_time_vs_thrust():
  csv_path = os.path.join(RESULTS_DIR, "low_thrust_transfer_time_vs_thrust.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/low_thrust_transfer.py를 실행하세요.")
    return

  thrust_values, times_hr = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      thrust_values.append(float(row["thrust_accel_km_s2"]))
      times_hr.append(float(row["transfer_time_hr"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(thrust_values, times_hr, marker="o", linewidth=2, color="tab:orange")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Thrust acceleration (km/s^2, log scale)")
  ax.set_ylabel("Transfer time (hours, log scale)")
  ax.set_title("Low-thrust transfer: higher thrust means shorter transfer time")
  ax.grid(True, which="both", alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "15_low_thrust_transfer_time_vs_thrust.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_attitude_intermediate_axis_instability():
  csv_path = os.path.join(RESULTS_DIR, "attitude_unstable_axis_omega_history.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 attitude/torque_free_rigid_body.py를 실행하세요.")
    return

  times, omega1, omega2, omega3 = [], [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times.append(float(row["t_sec"]))
      omega1.append(float(row["omega1"]))
      omega2.append(float(row["omega2"]))
      omega3.append(float(row["omega3"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times, omega1, linewidth=1.5, label="omega1 (perturbed)")
  ax.plot(times, omega2, linewidth=1.5, label="omega2 (spin axis, intermediate I2)")
  ax.plot(times, omega3, linewidth=1.5, label="omega3 (perturbed)")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Angular velocity component (rad/s)")
  ax.set_title("Intermediate axis theorem: small perturbations grow explosively")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "16_attitude_intermediate_axis_instability.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_attitude_perturbation_growth_comparison():
  csv_path = os.path.join(RESULTS_DIR, "attitude_perturbation_growth_comparison.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 attitude/torque_free_rigid_body.py를 실행하세요.")
    return

  labels_en = {"최소축(I1)": "Minor axis\n(I1)", "중간축(I2)": "Intermediate axis\n(I2)",
               "최대축(I3)": "Major axis\n(I3)"}
  labels, ratios = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      labels.append(labels_en.get(row["axis_label"], row["axis_label"]))
      ratios.append(float(row["growth_ratio"]))

  fig, ax = plt.subplots(figsize=(7, 5))
  colors = ["tab:green", "tab:red", "tab:green"]
  bars = ax.bar(labels, ratios, color=colors, alpha=0.85)
  ax.set_ylabel("Perturbation growth ratio")
  ax.set_title("Only the intermediate axis is unstable")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(ratios) * 0.02, f"{v:.1f}x", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "16_attitude_perturbation_growth_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_station_keeping_raan_sawtooth():
  csv_path = os.path.join(RESULTS_DIR, "station_keeping_raan_history.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/station_keeping.py를 실행하세요.")
    return

  times_days, errors_deg = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times_days.append(float(row["t_sec"]) / 86400)
      errors_deg.append(float(row["raan_error_deg"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times_days, errors_deg, color="tab:blue", linewidth=1.5, marker="o", markersize=3)
  ax.set_xlabel("Time (days)")
  ax.set_ylabel("Residual RAAN error (deg)")
  ax.set_title("Station-keeping: periodic burns keep residual RAAN error bounded (sawtooth)")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "17_station_keeping_raan_sawtooth.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_station_keeping_delta_v_budget():
  csv_path = os.path.join(RESULTS_DIR, "station_keeping_delta_v_budget.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/station_keeping.py를 실행하세요.")
    return

  tolerances, num_burns, total_dv = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      tolerances.append(float(row["tolerance_deg"]))
      num_burns.append(int(row["num_burns"]))
      total_dv.append(float(row["total_delta_v_ms"]))

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
  ax1.plot(tolerances, num_burns, marker="o", linewidth=2, color="tab:orange")
  ax1.set_xlabel("RAAN tolerance (deg)")
  ax1.set_ylabel("Number of burns over mission")
  ax1.set_title("Tighter tolerance needs more frequent burns")
  ax1.grid(True, alpha=0.3)

  ax2.plot(tolerances, total_dv, marker="o", linewidth=2, color="tab:green")
  ax2.set_xlabel("RAAN tolerance (deg)")
  ax2.set_ylabel("Total delta-V over mission (m/s)")
  ax2.set_title("Total delta-V budget is roughly independent of tolerance")
  ax2.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "17_station_keeping_delta_v_budget.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_orbital_decay_trajectory():
  csv_path = os.path.join(RESULTS_DIR, "orbital_decay_trajectory.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/orbital_decay.py를 실행하세요.")
    return

  # 지구 반지름(~6378km) 대비 고도 변화(200km->100km)가 상대적으로 작아서
  # x-y 평면에 그대로 그리면 두꺼운 원 하나로 뭉개져 나선이 안 보인다(15번의
  # 저추력 상승 나선은 반지름이 60% 가까이 바뀌어 뚜렷했지만, 이 하강은 지구
  # 반지름 대비 1.5%p 안팎의 변화라 그렇다) — 그래서 x-y 평면 대신 시간에 따른
  # 고도 자체를 직접 그린다.
  times_hr, altitudes_km = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times_hr.append(float(row["t_sec"]) / 3600)
      altitudes_km.append(float(row["altitude_km"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times_hr, altitudes_km, color="tab:red", linewidth=1.5)
  ax.scatter([times_hr[0]], [altitudes_km[0]], color="tab:green", s=80, zorder=3, label="Start (circular orbit)")
  ax.scatter([times_hr[-1]], [altitudes_km[-1]], color="black", s=80, zorder=3, label="Reentry")
  ax.set_xlabel("Time (hours)")
  ax.set_ylabel("Altitude (km)")
  ax.set_title("Orbital decay: atmospheric drag slowly lowers altitude, accelerating near reentry")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "18_orbital_decay_trajectory.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_lagrange_points_layout():
  csv_path = os.path.join(RESULTS_DIR, "lagrange_points_positions.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/lagrange_points.py를 실행하세요.")
    return

  positions = {}
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      positions[row["point"]] = (float(row["x_dimensionless"]), float(row["y_dimensionless"]))

  mu_row_path = os.path.join(RESULTS_DIR, "lagrange_points_triangle_check.csv")
  mu = 0.012150585
  if os.path.exists(mu_row_path):
    with open(mu_row_path, newline="", encoding="utf-8") as f:
      mu = float(next(csv.DictReader(f))["mass_ratio"])

  fig, ax = plt.subplots(figsize=(8, 8))
  ax.scatter([-mu], [0], color="tab:blue", s=300, zorder=3, label="M1 (Earth)")
  ax.scatter([1 - mu], [0], color="tab:gray", s=100, zorder=3, label="M2 (Moon)")
  for name, (x, y) in positions.items():
    color = "tab:green" if name in ("L4", "L5") else "tab:red"
    ax.scatter([x], [y], color=color, s=60, zorder=3)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=13)
  ax.axhline(0, color="black", linewidth=0.5)
  ax.set_xlabel("x (dimensionless, corotating frame)")
  ax.set_ylabel("y (dimensionless, corotating frame)")
  ax.set_title("CR3BP Lagrange points in the Earth-Moon corotating frame\n(L4/L5 stable in green, L1/L2/L3 unstable in red)")
  ax.set_aspect("equal")
  ax.legend(loc="lower left")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "19_lagrange_points_layout.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_lagrange_points_stability_trajectories():
  l4_csv = os.path.join(RESULTS_DIR, "lagrange_points_l4_trajectory.csv")
  l1_csv = os.path.join(RESULTS_DIR, "lagrange_points_l1_trajectory.csv")
  if not os.path.exists(l4_csv) or not os.path.exists(l1_csv):
    print(f"[건너뜀] {l4_csv} 또는 {l1_csv} 없음 — 먼저 missions/lagrange_points.py를 실행하세요.")
    return

  def read_xy(path):
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8") as f:
      for row in csv.DictReader(f):
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))
    return xs, ys

  l4_xs, l4_ys = read_xy(l4_csv)
  l1_xs, l1_ys = read_xy(l1_csv)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
  ax1.plot(l4_xs, l4_ys, color="tab:green", linewidth=1.0)
  ax1.scatter([l4_xs[0]], [l4_ys[0]], color="black", s=60, zorder=3, label="L4 (start)")
  ax1.set_xlabel("x (dimensionless)")
  ax1.set_ylabel("y (dimensionless)")
  ax1.set_title("Near L4: perturbed trajectory stays bounded (stable)")
  ax1.set_aspect("equal")
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  ax2.plot(l1_xs, l1_ys, color="tab:red", linewidth=1.0)
  ax2.scatter([l1_xs[0]], [l1_ys[0]], color="black", s=60, zorder=3, label="L1 (start)")
  ax2.set_xlabel("x (dimensionless)")
  ax2.set_ylabel("y (dimensionless)")
  ax2.set_title("Near L1: perturbed trajectory diverges (unstable)")
  ax2.set_aspect("equal")
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  fig.suptitle("Same perturbation size, opposite fate: L4 (stable) vs L1 (unstable)")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "19_lagrange_points_stability_trajectories.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_orbital_decay_lifetime_comparison():
  bc_csv = os.path.join(RESULTS_DIR, "orbital_decay_ballistic_coefficient_vs_lifetime.csv")
  altitude_csv = os.path.join(RESULTS_DIR, "orbital_decay_altitude_vs_lifetime.csv")
  if not os.path.exists(bc_csv) or not os.path.exists(altitude_csv):
    print(f"[건너뜀] {bc_csv} 또는 {altitude_csv} 없음 — 먼저 missions/orbital_decay.py를 실행하세요.")
    return

  bcs, bc_lifetimes = [], []
  with open(bc_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      bcs.append(float(row["ballistic_coefficient_kg_m2"]))
      bc_lifetimes.append(float(row["lifetime_days"]))

  altitudes, alt_lifetimes = [], []
  with open(altitude_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      altitudes.append(float(row["initial_altitude_km"]))
      alt_lifetimes.append(float(row["lifetime_days"]))

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
  ax1.plot(bcs, bc_lifetimes, marker="o", linewidth=2, color="tab:blue")
  ax1.set_xlabel("Ballistic coefficient (kg/m^2)")
  ax1.set_ylabel("Orbital lifetime (days)")
  ax1.set_title("Larger ballistic coefficient extends lifetime")
  ax1.grid(True, alpha=0.3)

  ax2.plot(altitudes, alt_lifetimes, marker="o", linewidth=2, color="tab:purple")
  ax2.set_yscale("log")
  ax2.set_xlabel("Initial altitude (km)")
  ax2.set_ylabel("Orbital lifetime (days, log scale)")
  ax2.set_title("Lifetime grows exponentially with initial altitude")
  ax2.grid(True, which="both", alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "18_orbital_decay_lifetime_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_lyapunov_orbit_trajectory():
  trajectory_csv = os.path.join(RESULTS_DIR, "lyapunov_orbits_trajectory.csv")
  reference_csv = os.path.join(RESULTS_DIR, "lyapunov_orbits_lagrange_reference.csv")
  if not os.path.exists(trajectory_csv) or not os.path.exists(reference_csv):
    print(f"[건너뜀] {trajectory_csv} 또는 {reference_csv} 없음 — 먼저 missions/lyapunov_orbits.py를 실행하세요.")
    return

  xs, ys = [], []
  with open(trajectory_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      xs.append(float(row["x"]))
      ys.append(float(row["y"]))

  reference_points = {}
  with open(reference_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      reference_points[row["point"]] = float(row["x_dimensionless"])

  # 궤도 진폭(약 0.01 무차원)이 L1-달-L2 사이 거리(약 0.3 무차원)보다 훨씬
  # 작아서, 그 넓은 범위를 다 보여주면 궤도 자체가 점처럼 뭉개진다 — 그래서
  # 궤도 주변으로만 확대하고, 근처에 있는 라그랑주점(L1 또는 L2)만 함께 표시한다.
  x_center = sum(xs) / len(xs)
  nearest_point_name = min(("L1", "L2"), key=lambda name: abs(reference_points[name] - x_center))
  nearest_x = reference_points[nearest_point_name]

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(xs, ys, color="tab:blue", linewidth=1.5, label="Corrected periodic orbit")
  ax.scatter([xs[0]], [ys[0]], color="black", s=60, zorder=3, label="Start (y=0 crossing)")
  ax.scatter([nearest_x], [0], color="tab:red", s=80, marker="x", zorder=3, label=f"{nearest_point_name} (libration point)")
  ax.set_xlabel("x (dimensionless, corotating frame)")
  ax.set_ylabel("y (dimensionless, corotating frame)")
  ax.set_title(f"Lyapunov orbit around {nearest_point_name}: a closed periodic loop, not a single point")
  ax.set_aspect("equal", adjustable="datalim")
  ax.legend(loc="best", fontsize=10)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "20_lyapunov_orbit_trajectory.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_lyapunov_amplitude_vs_period():
  family_csv = os.path.join(RESULTS_DIR, "lyapunov_orbits_family.csv")
  if not os.path.exists(family_csv):
    print(f"[건너뜀] {family_csv} 없음 — 먼저 missions/lyapunov_orbits.py를 실행하세요.")
    return

  amplitudes, periods, linear_period = [], [], None
  with open(family_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      if row["libration_point"] != "L1" or not row["period"] or row["period_linear_prediction"] == "":
        continue
      amplitudes.append(float(row["amplitude"]))
      periods.append(float(row["period"]))
      if row["period_linear_prediction"]:
        linear_period = float(row["period_linear_prediction"])

  if not amplitudes:
    print(f"[건너뜀] {family_csv}에 유효한 L1 궤도 가족 데이터 없음")
    return

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(amplitudes, periods, marker="o", linewidth=2, color="tab:blue", label="Corrected nonlinear period")
  if linear_period is not None:
    ax.axhline(linear_period, color="tab:red", linestyle="--", linewidth=1.5, label="Linear prediction (2π/ω)")
  ax.set_xlabel("Orbit amplitude (dimensionless)")
  ax.set_ylabel("Period (dimensionless)")
  ax.set_title("Lyapunov orbit family near L1: period drifts from the linear prediction")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "20_lyapunov_amplitude_vs_period.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_pid_point_and_hold_error():
  csv_path = os.path.join(RESULTS_DIR, "pid_point_and_hold_error_history.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 attitude/pid_attitude_control.py를 실행하세요.")
    return

  times, errors = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times.append(float(row["t_sec"]))
      errors.append(float(row["error_angle_deg"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times, errors, color="tab:blue", linewidth=1.5)
  ax.axhline(2.0, color="tab:red", linestyle="--", linewidth=1.5, label="Settled threshold (2 deg)")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Attitude error angle (deg)")
  ax.set_title("PD point-and-hold: attitude error settles toward the target quaternion")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "21_pid_point_and_hold_error.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_pid_intermediate_axis_comparison():
  csv_path = os.path.join(RESULTS_DIR, "pid_intermediate_axis_comparison.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 attitude/pid_attitude_control.py를 실행하세요.")
    return

  times, errors_uncontrolled, errors_controlled = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times.append(float(row["t_sec"]))
      errors_uncontrolled.append(float(row["error_angle_deg_uncontrolled"]))
      errors_controlled.append(float(row["error_angle_deg_controlled"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times, errors_uncontrolled, color="tab:red", linewidth=1.5, label="Without control (tumbles)")
  ax.plot(times, errors_controlled, color="tab:green", linewidth=1.5, label="With PD control")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Attitude error angle (deg)")
  ax.set_title("PD control tames the intermediate-axis tumble from script 16")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "21_pid_intermediate_axis_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_mission_timeline_combined():
  csv_path = os.path.join(RESULTS_DIR, "orbit_raise_and_reorient_timeline.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 missions/orbit_raise_and_reorient.py를 실행하세요.")
    return

  # 궤도 전이(수십 분)와 자세 재정렬(수십 초)의 시간축 스케일이 1000배 가까이
  # 달라, 하나의 선형 시간축에 억지로 합치면 자세 곡선이 오른쪽 끝에 수직선으로
  # 뭉개진다 — 17번 station-keeping 델타-V 그래프처럼 좌우 분리된 독립 x축
  # 서브플롯으로 나누고, 하나의 suptitle로만 내러티브를 통합한다.
  orbit_t_min, orbit_r_km = [], []
  attitude_t_abs_sec, attitude_error_deg = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      if row["phase"] == "orbit":
        orbit_t_min.append(float(row["t_sec"]) / 60)
        orbit_r_km.append(float(row["metric_value"]))
      elif row["phase"] == "attitude":
        attitude_t_abs_sec.append(float(row["t_sec"]))
        attitude_error_deg.append(float(row["metric_value"]))

  transfer_time_min = orbit_t_min[-1] if orbit_t_min else 0.0
  # attitude 단계의 t_sec는 궤도 전이 완료 시각(t_offset_sec)이 이미 더해진
  # 절대 임무시각이므로, 전이 완료 후 경과시간으로 보려면 t_offset을 다시 뺀다.
  t_offset_sec = transfer_time_min * 60
  attitude_t_sec = [t - t_offset_sec for t in attitude_t_abs_sec]
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

  ax1.plot(orbit_t_min, orbit_r_km, marker="o", linestyle="--", color="tab:blue",
           linewidth=2, label="Hohmann transfer burn")
  ax1.set_xlabel("Mission time (minutes)")
  ax1.set_ylabel("Orbit radius (km)")
  ax1.set_title("Orbit raise phase")
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  ax2.plot(attitude_t_sec, attitude_error_deg, color="tab:purple", linewidth=1.5)
  ax2.axhline(2.0, color="tab:red", linestyle="--", linewidth=1.5, label="Settled threshold (2 deg)")
  ax2.set_xlabel(f"Time since orbit raise completion (s)\n(mission clock starts at T+{transfer_time_min:.1f}min)")
  ax2.set_ylabel("Attitude error angle (deg)")
  ax2.set_title("Attitude reorientation phase")
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  fig.suptitle("Integrated mission timeline: orbit raise -> attitude reorientation")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "22_mission_timeline_combined.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_third_body_j2_crossover():
  csv_path = os.path.join(RESULTS_DIR, "third_body_j2_vs_altitude.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 perturbations/third_body_perturbation.py를 실행하세요.")
    return

  altitudes, j2_accel, lunisolar_accel = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      altitudes.append(float(row["altitude_km"]))
      j2_accel.append(float(row["j2_accel_km_s2"]))
      lunisolar_accel.append(float(row["lunisolar_total_accel_km_s2"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.semilogy(altitudes, j2_accel, marker="o", color="tab:blue", linewidth=1.5, label="J2 perturbation")
  ax.semilogy(altitudes, lunisolar_accel, marker="o", color="tab:orange", linewidth=1.5, label="Lunisolar (3rd-body) perturbation")
  ax.set_xlabel("Altitude (km)")
  ax.set_ylabel("Acceleration magnitude (km/s^2, log scale)")
  ax.set_title("J2 dominates at LEO; lunisolar perturbation overtakes J2 near GEO")
  ax.legend()
  ax.grid(True, alpha=0.3, which="both")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "23_third_body_j2_crossover.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_third_body_oscillating_perturbation():
  csv_path = os.path.join(RESULTS_DIR, "third_body_trajectory_deviation.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 perturbations/third_body_perturbation.py를 실행하세요.")
    return

  times_days, accel_mags = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times_days.append(float(row["t_sec"]) / 86400)
      accel_mags.append(float(row["lunisolar_accel_mag_km_s2"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times_days, accel_mags, color="tab:purple", linewidth=1.5)
  ax.set_xlabel("Time (days)")
  ax.set_ylabel("Lunisolar perturbation acceleration magnitude (km/s^2)")
  ax.set_title("Lunisolar perturbation oscillates as the Sun and Moon move")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "23_third_body_oscillation.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_srp_drag_crossover():
  csv_path = os.path.join(RESULTS_DIR, "solar_radiation_pressure_altitude_vs_drag.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 perturbations/solar_radiation_pressure.py를 실행하세요.")
    return

  altitudes, drag_accel, srp_accel = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      altitudes.append(float(row["altitude_km"]))
      drag_accel.append(float(row["drag_accel_km_s2"]))
      srp_accel.append(float(row["srp_accel_km_s2"]))

  fig, ax = plt.subplots(figsize=(11, 5))
  ax.semilogy(altitudes, drag_accel, marker="o", color="tab:blue", linewidth=1.5, label="Atmospheric drag")
  ax.semilogy(altitudes, srp_accel, marker="o", color="tab:red", linewidth=1.5, label="Solar radiation pressure")
  ax.set_xlabel("Altitude (km)")
  ax.set_ylabel("Acceleration magnitude (km/s^2, log scale)")
  ax.set_title("Drag dominates at low altitude; SRP dominates at high altitude")
  ax.legend()
  ax.grid(True, alpha=0.3, which="both")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "24_srp_drag_crossover.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_srp_eclipse_timeseries():
  csv_path = os.path.join(RESULTS_DIR, "solar_radiation_pressure_eclipse_trajectory.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 perturbations/solar_radiation_pressure.py를 실행하세요.")
    return

  times_min, accel_mags = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times_min.append(float(row["t_sec"]) / 60)
      accel_mags.append(float(row["srp_accel_mag_km_s2"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times_min, accel_mags, color="tab:green", linewidth=1.5, drawstyle="steps-post")
  ax.set_xlabel("Time (minutes)")
  ax.set_ylabel("SRP acceleration magnitude (km/s^2)")
  ax.set_title("Solar radiation pressure drops to exactly zero during eclipse passes")
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "24_srp_eclipse_timeseries.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_reaction_wheel_saturation_and_desaturation():
  csv_path = os.path.join(RESULTS_DIR, "reaction_wheel_desaturation_recovery.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 attitude/reaction_wheel_desaturation.py를 실행하세요.")
    return

  times, h_mags = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times.append(float(row["t_sec"]))
      h_mags.append(float(row["h_wheel_mag_nms"]))
  capacity = max(h_mags) if h_mags else 0.0

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times, h_mags, color="tab:blue", linewidth=1.5)
  ax.axhline(capacity, color="tab:red", linestyle="--", linewidth=1.5, label="Wheel capacity")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Wheel momentum magnitude (N*m*s)")
  ax.set_title("Reaction wheel momentum ramps up, saturates, then resets at desaturation")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "25_reaction_wheel_saturation_momentum.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_reaction_wheel_pointing_error_through_saturation():
  csv_path = os.path.join(RESULTS_DIR, "reaction_wheel_desaturation_recovery.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 attitude/reaction_wheel_desaturation.py를 실행하세요.")
    return

  times, errors, phases = [], [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      times.append(float(row["t_sec"]))
      errors.append(float(row["error_angle_deg"]))
      phases.append(row["phase"])

  desat_t = next((t for t, p in zip(times, phases) if p == "after_desat"), times[-1])

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(times, errors, color="tab:purple", linewidth=1.5)
  ax.axvline(desat_t, color="tab:green", linestyle="--", linewidth=1.5, label="Desaturation burn")
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Attitude error angle (deg)")
  ax.set_title("Prompt desaturation at the moment of saturation keeps pointing error negligible")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "25_reaction_wheel_pointing_error.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


if __name__ == "__main__":
  os.makedirs(RESULTS_DIR, exist_ok=True)
  plot_kepler_orbit_shape()
  plot_kepler_second_law_areas()
  plot_zenith_and_horizon_cases()
  plot_elevation_over_time()
  plot_contact_windows_gantt()
  plot_hohmann_transfer_orbit()
  plot_hohmann_delta_v_vs_ratio()
  plot_j2_raan_precession()
  plot_j2_ground_track_drift()
  plot_lambert_short_vs_long_way()
  plot_constellation_size_vs_gap()
  plot_constellation_plane_comparison()
  plot_isl_visibility_comparison()
  plot_patched_conic_transfer_orbit()
  plot_patched_conic_delta_v_breakdown()
  plot_cw_zero_drift_comparison()
  plot_cw_approximation_validity()
  plot_orbit_determination_noise_sensitivity()
  plot_orbit_determination_observation_count()
  plot_low_thrust_spiral_trajectory()
  plot_low_thrust_transfer_time_vs_thrust()
  plot_attitude_intermediate_axis_instability()
  plot_attitude_perturbation_growth_comparison()
  plot_station_keeping_raan_sawtooth()
  plot_station_keeping_delta_v_budget()
  plot_orbital_decay_trajectory()
  plot_orbital_decay_lifetime_comparison()
  plot_lagrange_points_layout()
  plot_lagrange_points_stability_trajectories()
  plot_lyapunov_orbit_trajectory()
  plot_lyapunov_amplitude_vs_period()
  plot_pid_point_and_hold_error()
  plot_pid_intermediate_axis_comparison()
  plot_mission_timeline_combined()
  plot_third_body_j2_crossover()
  plot_third_body_oscillating_perturbation()
  plot_srp_drag_crossover()
  plot_srp_eclipse_timeseries()
  plot_reaction_wheel_saturation_and_desaturation()
  plot_reaction_wheel_pointing_error_through_saturation()
