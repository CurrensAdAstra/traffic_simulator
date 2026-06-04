#!/usr/bin/env python3
"""
교통 시뮬레이션 검증용 표준 메트릭 (논문 E1-E7 공용).

순수 numpy 구현(scipy 의존 없음) — Docker 이미지/CI 어디서나 동작.
 - pearson_r, spearman_rho      : 선형/순위 상관
 - rmsn                         : RMS normalized error (교통 표준)
 - geh, geh_fraction            : GEH 통계 + GEH<threshold 비율(링크 flow 검증 표준)
 - ks_statistic                 : 두 표본 분포 거리(통행시간 CDF 비교)
 - precision_recall_at_k        : 혼잡 hotspot 검출(상/하위 K)
 - mape, bias                   : 통행시간 절대오차/편향
모든 함수는 numpy array를 받는다.
"""

from __future__ import annotations

import numpy as np


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2:
        return float("nan")
    a0 = a - a.mean(); b0 = b - b.mean()
    da = float(np.sqrt((a0 * a0).sum())); db = float(np.sqrt((b0 * b0).sum()))
    if da <= 0 or db <= 0:
        return float("nan")
    return float((a0 * b0).sum() / (da * db))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """평균 순위(동점 평균) — Spearman용."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # 동점 평균 처리
    sx = x[order]
    i = 0
    n = x.size
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0  # 1-base 평균
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2:
        return float("nan")
    return pearson_r(_rankdata(a), _rankdata(b))


def rmsn(ref: np.ndarray, test: np.ndarray) -> float:
    """RMS normalized error = sqrt(N·Σ(t-r)^2) / Σr. 교통 보정 표준 지표."""
    ref = np.asarray(ref, float); test = np.asarray(test, float)
    n = ref.size
    denom = float(ref.sum())
    if n == 0 or denom == 0:
        return float("nan")
    return float(np.sqrt(n * np.sum((test - ref) ** 2)) / denom)


def geh(model: np.ndarray, count: np.ndarray) -> np.ndarray:
    """GEH = sqrt(2(M-C)^2 / (M+C)). M,C는 시간당 볼륨(veh/h) 권장."""
    model = np.asarray(model, float); count = np.asarray(count, float)
    s = model + count
    out = np.zeros_like(s)
    nz = s > 0
    out[nz] = np.sqrt(2.0 * (model[nz] - count[nz]) ** 2 / s[nz])
    return out


def geh_fraction(model: np.ndarray, count: np.ndarray, threshold: float = 5.0) -> float:
    """GEH < threshold 인 링크 비율(교통 표준: 85% 이상이면 합격)."""
    g = geh(model, count)
    if g.size == 0:
        return float("nan")
    return float((g < threshold).mean())


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """두 표본 KS 거리 = max|F_a - F_b|. 분포 형태 일치도."""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def precision_recall_at_k(ref_score: np.ndarray, test_score: np.ndarray, k: int,
                          lowest: bool = True) -> tuple[float, float]:
    """하위(lowest=True) 또는 상위 K 집합의 precision/recall.
    혼잡 hotspot 검출(속도 하위 K). ref/test 동일 길이, 같은 edge 순서 가정.
    """
    ref_score = np.asarray(ref_score, float); test_score = np.asarray(test_score, float)
    n = ref_score.size
    k = min(k, n)
    if k == 0:
        return (float("nan"), float("nan"))
    order = (np.argsort) if lowest else (lambda x: np.argsort(-x))
    ref_set = set(np.argsort(ref_score)[:k] if lowest else np.argsort(-ref_score)[:k])
    test_set = set(np.argsort(test_score)[:k] if lowest else np.argsort(-test_score)[:k])
    tp = len(ref_set & test_set)
    precision = tp / len(test_set)
    recall = tp / len(ref_set)
    return (precision, recall)


def mape(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean absolute percentage error (ref 기준). %."""
    ref = np.asarray(ref, float); test = np.asarray(test, float)
    nz = ref != 0
    if nz.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((test[nz] - ref[nz]) / ref[nz])))


def bias(ref: np.ndarray, test: np.ndarray) -> float:
    """평균 편향 test - ref."""
    ref = np.asarray(ref, float); test = np.asarray(test, float)
    if ref.size == 0:
        return float("nan")
    return float(np.mean(test - ref))
