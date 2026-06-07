#!/usr/bin/env bash
# 긴 매트릭스 sweep을 tmux session으로 detach 실행. 연결이 끊겨도 계속 동작.
# 사용:
#   bash launch_matrix_tmux.sh                # 새 세션 시작
#   bash launch_matrix_tmux.sh --resume       # 기존 master.csv에서 이어 실행
#   bash launch_matrix_tmux.sh --status       # 진행상황 폴링(셀별 [CELL]/[ETA] 마커)
#   bash launch_matrix_tmux.sh --tail         # tmux 출력 라이브 보기
#   bash launch_matrix_tmux.sh --attach       # 세션 부착(Ctrl-b d 로 detach)
#   bash launch_matrix_tmux.sh --kill         # 세션 종료

set -u
SESSION=${SESSION:-tsmatrix}
WT="/home/mgkyung/ts/.claude/worktrees/elastic-allen-cc712e"
LOG="${WT}/paper_data/matrix/run.log"
SCRIPT="${WT}/gangnam4_cuda/run_matrix.py"
mkdir -p "$(dirname "$LOG")"

action="${1:-start}"
case "$action" in
  --status)
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[status] 세션 '$SESSION' 없음"; ls -la "$LOG" 2>/dev/null
    else
      echo "[status] 세션 '$SESSION' 실행중"
    fi
    echo "---- 최근 진행 마커 ----"; grep -E "^\[CELL\]|^\[ETA\]|^\[LOG\] 매트릭스" "$LOG" 2>/dev/null | tail -20
    echo "---- master 행 수 ----"; [[ -f "$WT/paper_data/matrix/matrix.csv" ]] && wc -l "$WT/paper_data/matrix/matrix.csv" || echo "matrix.csv 없음"
    ;;
  --tail)
    tail -F "$LOG"
    ;;
  --attach)
    tmux attach -t "$SESSION"
    ;;
  --kill)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed $SESSION" || echo "no session"
    ;;
  --resume|start|"")
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[start] 세션 '$SESSION' 이미 실행중. --status / --tail / --attach 로 확인하세요."; exit 1
    fi
    RESUME_ARG=""; [[ "$action" == "--resume" ]] && RESUME_ARG="--resume"
    cmd="cd /home/mgkyung/ts && python3 -u '$SCRIPT' $RESUME_ARG 2>&1 | tee '$LOG'"
    tmux new-session -d -s "$SESSION" "$cmd"
    echo "[start] tmux 세션 '$SESSION' detached 실행중"
    echo "  log:    $LOG"
    echo "  status: bash $0 --status"
    echo "  attach: bash $0 --attach   (Ctrl-b d 로 detach)"
    ;;
  *)
    echo "unknown action: $action"; exit 2;;
esac
