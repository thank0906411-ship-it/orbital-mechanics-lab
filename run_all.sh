#!/usr/bin/env bash
# 01~12번 시뮬레이션을 순서대로 한 번에 실행하고, 마지막에 report.html까지 재생성하는
# 스크립트 (리눅스/macOS/Git Bash용). run_all.ps1의 bash 이식판 — 로직과 단계 구성은 동일하다.
# 실행: ./run_all.sh (또는 bash run_all.sh)

set -u  # 정의 안 된 변수 참조 시 즉시 에러

cd "$(dirname "$0")"

write_section() {
  echo ""
  printf '=%.0s' {1..70}
  echo ""
  echo "$1"
  printf '=%.0s' {1..70}
  echo ""
}

# 매 단계마다 종료 코드를 직접 확인해 실패 시 즉시 중단한다 — 안 그러면 중간 단계가
# 죽어도 스크립트가 끝까지 돌고 "완료" 메시지가 뜨는 조용한 실패가 발생한다.
invoke_step() {
  local description="$1"
  shift
  "$@"
  local exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo ""
    echo "[중단] ${description} 단계가 종료 코드 ${exit_code} 로 실패했습니다. 파이프라인을 멈춥니다."
    exit "$exit_code"
  fi
}

# results/를 매 실행 전에 비운다 — 안 그러면 예전에 스크립트를 리팩터링/삭제하면서 더
# 이상 어떤 스크립트도 만들지 않는 CSV/PNG가 폴더에 계속 남아 최신 결과와 뒤섞인다.
# 번호 있는 단계(1/14~)에 넣지 않은 이유: 이건 "시뮬레이션을 실행"하는 게 아니라 그
# 전에 실행 환경을 정리하는 준비 작업이라, 파이프라인 진행률 표시에 포함시키지 않는다.
echo "[정리] results/의 이전 실행 잔재물을 비웁니다."
rm -rf results

write_section "1/14 자동 테스트 (pytest) - 코드가 깨진 상태로 시뮬레이션을 돌리지 않기 위한 사전 검증"
invoke_step "pytest" python -m pytest tests/ -q

write_section "2/14 케플러 방정식과 궤도 전파"
invoke_step "01_Kepler_orbit_propagation.py" python 01_Kepler_orbit_propagation.py

write_section "3/14 궤도요소-상태벡터 변환과 보존량(에너지/각운동량)"
invoke_step "02_Orbital_elements_and_energy.py" python 02_Orbital_elements_and_energy.py

write_section "4/14 2체 문제 수치적분(RK4)과 해석해 검증"
invoke_step "03_Two_body_numerical_integration.py" python 03_Two_body_numerical_integration.py

write_section "5/14 좌표계 변환 (ECI/ECEF/SEZ)과 방위각/고도각"
invoke_step "04_Coordinate_frame_transforms.py" python 04_Coordinate_frame_transforms.py

write_section "6/14 지상국 가시성: 실측 궤도 전파로 계산하는 접촉 창"
invoke_step "05_Ground_station_visibility.py" python 05_Ground_station_visibility.py

write_section "7/14 호만 전이(Hohmann Transfer) 델타-V와 전이시간"
invoke_step "06_Hohmann_transfer.py" python 06_Hohmann_transfer.py

write_section "8/14 J2 섭동: RAAN/근점편각 세차"
invoke_step "07_J2_perturbation.py" python 07_J2_perturbation.py

write_section "9/14 란베르트 문제: 두 위치-비행시간으로 궤도 속도 계산"
invoke_step "09_Lambert_problem.py" python 09_Lambert_problem.py

write_section "10/14 다중 위성 성좌 커버리지: 워커 델타 패턴과 재방문 공백"
invoke_step "10_Constellation_coverage.py" python 10_Constellation_coverage.py

write_section "11/14 위성간 링크(ISL) 가시선: 지구 차단 기하 판정"
invoke_step "11_Intersatellite_link_visibility.py" python 11_Intersatellite_link_visibility.py

write_section "12/14 행성간 궤적: Patched Conic 근사로 지구-화성 임무 델타-V"
invoke_step "12_Patched_conic_interplanetary.py" python 12_Patched_conic_interplanetary.py

write_section "13/14 결과 시각화 (results/*.png 생성)"
invoke_step "08_visualize_orbits.py" python 08_visualize_orbits.py

write_section "14/14 프로젝트 요약 리포트 생성 (report.html)"
invoke_step "generate_report.py" python generate_report.py

write_section "전체 시뮬레이션 실행 완료"
