"""Tests for the profiling harness helpers."""

from __future__ import annotations

from superglm.profiling.harness import (
    SystemSample,
    flatten_system_sample,
    summarize_system_samples,
)


def test_flatten_system_sample_expands_per_core_metrics() -> None:
    sample = SystemSample(
        t_rel_s=1.25,
        rss_bytes=2_000_000,
        vms_bytes=3_000_000,
        uss_bytes=1_500_000,
        child_rss_bytes=250_000,
        process_cpu_percent=87.5,
        load_avg_1m=9.0,
        load_avg_5m=8.0,
        load_avg_15m=7.0,
        thread_count=12,
        voluntary_ctx_switches=10,
        involuntary_ctx_switches=2,
        read_bytes=1024,
        write_bytes=2048,
        available_memory_bytes=16_000_000,
        gc_gen0=4,
        gc_gen1=5,
        gc_gen2=6,
        cpu_percent_per_core=(10.0, 20.0, 30.0),
    )

    row = flatten_system_sample(sample)

    assert row["t_rel_s"] == 1.25
    assert row["rss_bytes"] == 2_000_000
    assert row["child_rss_bytes"] == 250_000
    assert row["cpu_core_0_percent"] == 10.0
    assert row["cpu_core_1_percent"] == 20.0
    assert row["cpu_core_2_percent"] == 30.0


def test_summarize_system_samples_reports_peaks_and_means() -> None:
    samples = [
        SystemSample(
            t_rel_s=0.0,
            rss_bytes=2_200,
            vms_bytes=3_100,
            uss_bytes=1_800,
            child_rss_bytes=100,
            process_cpu_percent=25.0,
            load_avg_1m=4.0,
            load_avg_5m=3.0,
            load_avg_15m=2.0,
            thread_count=8,
            voluntary_ctx_switches=1,
            involuntary_ctx_switches=0,
            read_bytes=10,
            write_bytes=20,
            available_memory_bytes=20_000,
            gc_gen0=1,
            gc_gen1=2,
            gc_gen2=3,
            cpu_percent_per_core=(10.0, 20.0),
        ),
        SystemSample(
            t_rel_s=0.5,
            rss_bytes=3_400,
            vms_bytes=4_400,
            uss_bytes=2_700,
            child_rss_bytes=300,
            process_cpu_percent=75.0,
            load_avg_1m=10.0,
            load_avg_5m=8.0,
            load_avg_15m=6.0,
            thread_count=14,
            voluntary_ctx_switches=4,
            involuntary_ctx_switches=1,
            read_bytes=30,
            write_bytes=80,
            available_memory_bytes=18_000,
            gc_gen0=2,
            gc_gen1=3,
            gc_gen2=4,
            cpu_percent_per_core=(30.0, 50.0),
        ),
    ]

    summary = summarize_system_samples(samples)

    assert summary["n_samples"] == 2
    assert summary["rss_peak_bytes"] == 3_400
    assert summary["rss_delta_bytes"] == 1_200
    assert summary["process_cpu_mean_percent"] == 50.0
    assert summary["process_cpu_peak_percent"] == 75.0
    assert summary["load_avg_1m_peak"] == 10.0
    assert summary["thread_count_peak"] == 14
    assert summary["cpu_core_1_peak_percent"] == 50.0
