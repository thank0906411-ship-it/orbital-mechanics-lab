"""
01, 04, 05, 06, 07, 09, 10, 11, 12번 시뮬레이션이 남긴 결과 CSV를 그래프로 그려주는
도구. 시뮬레이션 코드가 아니라 "결과를 눈으로 보기 위한" 별도 스크립트다.

실행 전에 먼저 01_Kepler_orbit_propagation.py, 04_Coordinate_frame_transforms.py,
05_Ground_station_visibility.py, 06_Hohmann_transfer.py, 07_J2_perturbation.py,
09_Lambert_problem.py, 10_Constellation_coverage.py,
11_Intersatellite_link_visibility.py, 12_Patched_conic_interplanetary.py를 한 번
이상 실행해서 results/ 폴더에 CSV가 생성되어 있어야 한다. (해당 CSV가 없는 항목은
건너뛰고 나머지만 그린다.)

실행: python 08_visualize_orbits.py
출력: results/kepler_orbit_shape.png, results/kepler_second_law_areas.png,
      results/zenith_and_horizon_cases.png, results/elevation_over_time.png,
      results/contact_windows_gantt.png, results/hohmann_transfer_orbit.png,
      results/hohmann_delta_v_vs_ratio.png, results/j2_raan_precession.png,
      results/j2_ground_track_drift.png, results/lambert_short_vs_long_way.png,
      results/constellation_size_vs_gap.png, results/constellation_plane_comparison.png,
      results/isl_visibility_comparison.png, results/patched_conic_transfer_orbit.png,
      results/patched_conic_delta_v_breakdown.png

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def plot_kepler_orbit_shape():
  csv_path = os.path.join(RESULTS_DIR, "kepler_position_time_series.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 01_Kepler_orbit_propagation.py를 실행하세요.")
    return

  xs, ys = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      xs.append(float(row["x_p_km"]))
      ys.append(float(row["y_p_km"]))

  fig, ax = plt.subplots(figsize=(7, 7))
  ax.plot(xs, ys, color="tab:blue", linewidth=1.5)
  ax.scatter([0], [0], color="tab:orange", s=100, marker="*", zorder=3, label="Focus (Earth)")
  ax.set_xlabel("Perifocal x (km)")
  ax.set_ylabel("Perifocal y (km)")
  ax.set_title("Kepler orbit propagation: orbital shape in the perifocal plane")
  ax.set_aspect("equal")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "kepler_orbit_shape.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_kepler_second_law_areas():
  csv_path = os.path.join(RESULTS_DIR, "circular_vs_elliptical.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 01_Kepler_orbit_propagation.py를 실행하세요.")
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

  out_path = os.path.join(RESULTS_DIR, "kepler_second_law_areas.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_zenith_and_horizon_cases():
  horizon_path = os.path.join(RESULTS_DIR, "horizon_cases.csv")
  if not os.path.exists(horizon_path):
    print(f"[건너뜀] {horizon_path} 없음 — 먼저 04_Coordinate_frame_transforms.py를 실행하세요.")
    return

  # "label"은 04번이 콘솔 출력용으로 쓴 한글 문구라 문구가 바뀌면 매칭이 깨진다 —
  # 대신 이 그래프가 실제로 보여주려는 것 자체인 고도각의 부호로 라벨을 새로 만든다.
  labels, elevations = [], []
  with open(horizon_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      elevation = float(row["elevation_deg"])
      labels.append("Above horizon\n(visible)" if elevation > 0 else "Below horizon\n(not visible)")
      elevations.append(elevation)

  fig, ax = plt.subplots(figsize=(7, 5))
  colors = ["tab:green" if e > 0 else "tab:red" for e in elevations]
  bars = ax.bar(labels, elevations, color=colors, alpha=0.85)
  ax.axhline(0, color="black", linewidth=1)
  ax.set_ylabel("Elevation (deg)")
  ax.set_title("Elevation sign determines visibility: above vs below the horizon")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, e in zip(bars, elevations):
    ax.text(bar.get_x() + bar.get_width() / 2, e + (2 if e > 0 else -4), f"{e:.1f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "zenith_and_horizon_cases.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_elevation_over_time():
  csv_path = os.path.join(RESULTS_DIR, "elevation_time_series.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 05_Ground_station_visibility.py를 실행하세요.")
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

  out_path = os.path.join(RESULTS_DIR, "elevation_over_time.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_contact_windows_gantt():
  csv_path = os.path.join(RESULTS_DIR, "contact_windows.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 05_Ground_station_visibility.py를 실행하세요.")
    return

  windows = []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      windows.append((float(row["aos_sec"]) / 60, float(row["duration_sec"]) / 60, float(row["max_elevation_deg"])))

  if not windows:
    print(f"[건너뜀] {csv_path}에 접촉 창이 없음")
    return

  fig, ax = plt.subplots(figsize=(9, 4))
  for idx, (aos_min, duration_min, max_el) in enumerate(windows):
    ax.barh(idx, duration_min, left=aos_min, color="tab:blue", alpha=0.85)
    ax.text(aos_min + duration_min / 2, idx, f"max El={max_el:.0f}°", ha="center", va="center",
            color="white", fontsize=8)
  ax.set_yticks(range(len(windows)))
  ax.set_yticklabels([f"Pass {i + 1}" for i in range(len(windows))])
  ax.set_xlabel("Time (min)")
  ax.set_title("Contact windows (AOS-LOS) from real orbit propagation")
  ax.grid(True, axis="x", alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "contact_windows_gantt.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_hohmann_transfer_orbit():
  csv_path = os.path.join(RESULTS_DIR, "hohmann_propagated_verification.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 06_Hohmann_transfer.py를 실행하세요.")
    return

  with open(csv_path, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
  r1, r2 = float(row["r1_km"]), float(row["r2_km"])
  a_t = (r1 + r2) / 2
  e_t = (r2 - r1) / (r2 + r1)

  theta = np.linspace(0, 2 * np.pi, 400)
  r_transfer = a_t * (1 - e_t ** 2) / (1 + e_t * np.cos(theta))
  x_transfer, y_transfer = r_transfer * np.cos(theta), r_transfer * np.sin(theta)

  fig, ax = plt.subplots(figsize=(7, 7))
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
  ax.legend(fontsize=9)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "hohmann_transfer_orbit.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_hohmann_delta_v_vs_ratio():
  csv_path = os.path.join(RESULTS_DIR, "hohmann_delta_v_vs_ratio.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 06_Hohmann_transfer.py를 실행하세요.")
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

  out_path = os.path.join(RESULTS_DIR, "hohmann_delta_v_vs_ratio.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_j2_raan_precession():
  csv_path = os.path.join(RESULTS_DIR, "j2_raan_precession_by_inclination.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 07_J2_perturbation.py를 실행하세요.")
    return

  inclinations, rates = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      inclinations.append(float(row["inclination_deg"]))
      rates.append(float(row["raan_rate_deg_per_day"]))

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(inclinations, rates, marker="o", linewidth=2, color="tab:blue")
  ax.axhline(0, color="black", linewidth=1)
  ax.axvline(90, color="tab:gray", linestyle=":", label="Polar orbit (90 deg): zero precession")
  ax.set_xlabel("Inclination (deg)")
  ax.set_ylabel("RAAN precession rate (deg/day)")
  ax.set_title("J2 perturbation: RAAN precession direction flips at 90 deg inclination")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "j2_raan_precession.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_j2_ground_track_drift():
  csv_path = os.path.join(RESULTS_DIR, "j2_long_term_raan_drift.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 07_J2_perturbation.py를 실행하세요.")
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

  out_path = os.path.join(RESULTS_DIR, "j2_ground_track_drift.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_lambert_short_vs_long_way():
  csv_path = os.path.join(RESULTS_DIR, "lambert_short_vs_long_way.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 09_Lambert_problem.py를 실행하세요.")
    return

  labels_en = {"짧은 길(prograde)": "Short way\n(prograde)", "긴 길(retrograde 선택)": "Long way\n(retrograde)"}
  labels, v1_values = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      labels.append(labels_en.get(row["label"], row["label"]))
      v1_values.append(float(row["v1_km_s"]))

  fig, ax = plt.subplots(figsize=(7, 5))
  bars = ax.bar(labels, v1_values, color=["tab:blue", "tab:red"], alpha=0.85)
  ax.set_ylabel("Departure speed |v1| (km/s)")
  ax.set_title("Lambert's problem: short way vs long way require different speeds")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, v1_values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "lambert_short_vs_long_way.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_constellation_size_vs_gap():
  csv_path = os.path.join(RESULTS_DIR, "constellation_size_vs_gap.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 10_Constellation_coverage.py를 실행하세요.")
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

  out_path = os.path.join(RESULTS_DIR, "constellation_size_vs_gap.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_constellation_plane_comparison():
  csv_path = os.path.join(RESULTS_DIR, "constellation_plane_comparison.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 10_Constellation_coverage.py를 실행하세요.")
    return

  labels_en = {"단일 평면(1개 평면에 12기)": "Single plane\n(12 sats, 1 plane)",
               "다중 평면(3개 평면에 4기씩)": "Multi-plane\n(12 sats, 3 planes)"}
  labels, max_gaps = [], []
  with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      labels.append(labels_en.get(row["label"], row["label"]))
      max_gaps.append(float(row["max_gap_min"]))

  fig, ax = plt.subplots(figsize=(7, 5))
  bars = ax.bar(labels, max_gaps, color=["tab:blue", "tab:orange"], alpha=0.85)
  ax.set_ylabel("Max coverage gap (min)")
  ax.set_title("Single plane can beat multi-plane for one fixed ground site\n(counter to the 'spread planes' intuition)")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, max_gaps):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "constellation_plane_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_isl_visibility_comparison():
  same_plane_path = os.path.join(RESULTS_DIR, "isl_same_plane_visibility.csv")
  cross_plane_path = os.path.join(RESULTS_DIR, "isl_polar_vs_equatorial.csv")
  if not os.path.exists(same_plane_path) or not os.path.exists(cross_plane_path):
    print(f"[건너뜀] {same_plane_path} 또는 {cross_plane_path} 없음 — "
          "먼저 11_Intersatellite_link_visibility.py를 실행하세요.")
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
  ax1.legend(loc="upper right", fontsize=8)

  ax2.fill_between(cross_times, 0, 1, where=cross_blocked, color="tab:red", alpha=0.6, step="post", label="Blocked")
  ax2.fill_between(cross_times, 0, 1, where=[not b for b in cross_blocked], color="tab:green", alpha=0.4,
                    step="post", label="Visible")
  ax2.set_ylim(0, 1)
  ax2.set_yticks([])
  ax2.set_xlabel("Time (min)")
  ax2.set_title("ISL visibility: polar vs equatorial planes")
  ax2.legend(loc="upper right", fontsize=8)

  fig.suptitle("Inter-satellite link visibility: same plane vs crossing planes")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "isl_visibility_comparison.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_patched_conic_transfer_orbit():
  csv_path = os.path.join(RESULTS_DIR, "patched_conic_heliocentric_propagation.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 12_Patched_conic_interplanetary.py를 실행하세요.")
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
  ax.legend(fontsize=9)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "patched_conic_transfer_orbit.png")
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"[저장됨] {out_path}")


def plot_patched_conic_delta_v_breakdown():
  csv_path = os.path.join(RESULTS_DIR, "patched_conic_mission_delta_v.csv")
  if not os.path.exists(csv_path):
    print(f"[건너뜀] {csv_path} 없음 — 먼저 12_Patched_conic_interplanetary.py를 실행하세요.")
    return

  with open(csv_path, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
  dv_depart = float(row["dv_depart"])
  dv_capture = float(row["dv_capture"])
  total_dv = float(row["total_dv"])

  fig, ax = plt.subplots(figsize=(7, 5))
  labels = ["Earth departure\n(hyperbolic escape)", "Mars capture\n(hyperbolic capture)"]
  values = [dv_depart, dv_capture]
  bars = ax.bar(labels, values, color=["tab:blue", "tab:red"], alpha=0.85)
  ax.set_ylabel("Delta-V (km/s)")
  ax.set_title(f"Earth-to-Mars mission delta-V breakdown (total: {total_dv:.2f} km/s)")
  ax.grid(True, axis="y", alpha=0.3)
  for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
  fig.tight_layout()

  out_path = os.path.join(RESULTS_DIR, "patched_conic_delta_v_breakdown.png")
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
