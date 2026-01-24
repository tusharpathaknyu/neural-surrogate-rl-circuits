"""
Utility functions for waveform processing and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict
import torch


def compute_fft(waveform: np.ndarray, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude spectrum.
    
    Returns:
        freqs: Frequency bins
        magnitude: Magnitude spectrum
    """
    n = len(waveform)
    fft = np.fft.rfft(waveform)
    magnitude = np.abs(fft) * 2 / n
    freqs = np.fft.rfftfreq(n, 1/sample_rate)
    return freqs, magnitude


def compute_thd(waveform: np.ndarray, fundamental_freq: float, sample_rate: float, num_harmonics: int = 10) -> float:
    """
    Compute Total Harmonic Distortion.
    
    THD = sqrt(sum(V_h^2)) / V_fundamental
    """
    freqs, magnitude = compute_fft(waveform, sample_rate)
    
    # Find fundamental
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fund_mag = magnitude[fund_idx]
    
    # Find harmonics
    harmonic_power = 0
    for h in range(2, num_harmonics + 1):
        h_freq = h * fundamental_freq
        h_idx = np.argmin(np.abs(freqs - h_freq))
        harmonic_power += magnitude[h_idx] ** 2
    
    thd = np.sqrt(harmonic_power) / fund_mag if fund_mag > 0 else 0
    return thd * 100  # Return as percentage


def compute_ripple(waveform: np.ndarray, method: str = 'pp') -> Dict:
    """
    Compute ripple characteristics.
    
    Args:
        method: 'pp' (peak-to-peak), 'rms' (RMS), or 'both'
    
    Returns:
        Dictionary with ripple metrics
    """
    dc = np.mean(waveform)
    ac = waveform - dc
    
    pp_ripple = np.max(waveform) - np.min(waveform)
    rms_ripple = np.sqrt(np.mean(ac ** 2))
    
    return {
        'dc_value': dc,
        'pp_ripple': pp_ripple,
        'pp_ripple_percent': pp_ripple / dc * 100 if dc != 0 else 0,
        'rms_ripple': rms_ripple,
        'rms_ripple_percent': rms_ripple / dc * 100 if dc != 0 else 0,
    }


def compute_settling_time(
    waveform: np.ndarray, 
    target_value: float,
    tolerance: float = 0.02,
    sample_rate: float = 1.0
) -> float:
    """
    Compute settling time to within tolerance of target.
    
    Returns:
        Settling time in seconds (or samples if sample_rate=1)
    """
    error = np.abs(waveform - target_value) / target_value
    within_tolerance = error < tolerance
    
    # Find last time it leaves tolerance band
    if np.all(within_tolerance):
        return 0
    
    # Find where it finally settles
    for i in range(len(waveform) - 1, -1, -1):
        if not within_tolerance[i]:
            return (i + 1) / sample_rate
    
    return len(waveform) / sample_rate


def compute_overshoot(waveform: np.ndarray, target_value: float) -> float:
    """
    Compute percentage overshoot.
    """
    max_val = np.max(waveform)
    if target_value == 0:
        return 0
    overshoot = (max_val - target_value) / target_value * 100
    return max(0, overshoot)


def compute_rise_time(
    waveform: np.ndarray,
    low_pct: float = 0.1,
    high_pct: float = 0.9,
    sample_rate: float = 1.0
) -> float:
    """
    Compute rise time (10% to 90% by default).
    """
    min_val = np.min(waveform)
    max_val = np.max(waveform)
    
    low_threshold = min_val + low_pct * (max_val - min_val)
    high_threshold = min_val + high_pct * (max_val - min_val)
    
    # Find crossings
    low_idx = np.argmax(waveform >= low_threshold)
    high_idx = np.argmax(waveform >= high_threshold)
    
    if high_idx <= low_idx:
        return 0
    
    return (high_idx - low_idx) / sample_rate


def resample_waveform(
    waveform: np.ndarray,
    original_time: np.ndarray,
    target_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample waveform to fixed number of points.
    """
    interp_func = interp1d(original_time, waveform, kind='linear')
    new_time = np.linspace(original_time[0], original_time[-1], target_points)
    new_waveform = interp_func(new_time)
    return new_time, new_waveform


def extract_steady_state(
    waveform: np.ndarray,
    num_periods: int = 3,
    period_samples: int = None
) -> np.ndarray:
    """
    Extract steady-state portion of waveform.
    Uses last N periods.
    """
    if period_samples is None:
        # Estimate period from autocorrelation
        autocorr = np.correlate(waveform - np.mean(waveform), 
                               waveform - np.mean(waveform), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after zero
        peaks, _ = signal.find_peaks(autocorr)
        if len(peaks) > 0:
            period_samples = peaks[0]
        else:
            period_samples = len(waveform) // 10
    
    samples_needed = num_periods * period_samples
    return waveform[-samples_needed:]


def plot_waveform_analysis(
    waveform: np.ndarray,
    sample_rate: float = 1e6,
    title: str = "Waveform Analysis",
    save_path: Optional[str] = None
):
    """
    Comprehensive waveform analysis plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time = np.arange(len(waveform)) / sample_rate * 1e6  # Convert to µs
    
    # Time domain
    ax1 = axes[0, 0]
    ax1.plot(time, waveform, 'b-', linewidth=1)
    ax1.set_xlabel('Time (µs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Time Domain')
    ax1.grid(True)
    
    # Zoom on ripple (last 10%)
    ax2 = axes[0, 1]
    start_idx = int(0.9 * len(waveform))
    ax2.plot(time[start_idx:], waveform[start_idx:], 'b-', linewidth=1)
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('Steady-State Ripple (Zoomed)')
    ax2.grid(True)
    
    # Frequency domain
    ax3 = axes[1, 0]
    freqs, magnitude = compute_fft(waveform, sample_rate)
    freqs_khz = freqs / 1e3
    ax3.semilogy(freqs_khz, magnitude)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Frequency Spectrum')
    ax3.set_xlim([0, min(1000, freqs_khz[-1])])
    ax3.grid(True)
    
    # Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    ripple = compute_ripple(waveform)
    overshoot = compute_overshoot(waveform, ripple['dc_value'])
    rise_time = compute_rise_time(waveform, sample_rate=sample_rate)
    settling = compute_settling_time(waveform, ripple['dc_value'], sample_rate=sample_rate)
    
    metrics_text = f"""
    WAVEFORM METRICS
    ================
    
    DC Value:       {ripple['dc_value']:.3f} V
    
    Ripple (P-P):   {ripple['pp_ripple']*1000:.2f} mV
    Ripple (%):     {ripple['pp_ripple_percent']:.2f}%
    
    Ripple (RMS):   {ripple['rms_ripple']*1000:.2f} mV
    
    Overshoot:      {overshoot:.1f}%
    Rise Time:      {rise_time*1e6:.2f} µs
    Settling Time:  {settling*1e6:.2f} µs
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=12, fontfamily='monospace',
             verticalalignment='center')
    ax4.set_title('Metrics Summary')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_waveforms(
    waveform1: np.ndarray,
    waveform2: np.ndarray,
    labels: Tuple[str, str] = ('Target', 'Predicted'),
    sample_rate: float = 1e6,
    title: str = "Waveform Comparison",
    save_path: Optional[str] = None
):
    """
    Compare two waveforms side by side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time = np.arange(len(waveform1)) / sample_rate * 1e6
    
    # Overlay
    ax1 = axes[0, 0]
    ax1.plot(time, waveform1, 'b-', label=labels[0], linewidth=2)
    ax1.plot(time, waveform2, 'r--', label=labels[1], linewidth=2)
    ax1.set_xlabel('Time (µs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Waveform Overlay')
    ax1.legend()
    ax1.grid(True)
    
    # Error
    ax2 = axes[0, 1]
    error = waveform2 - waveform1
    ax2.plot(time, error * 1000, 'g-', linewidth=1)  # mV
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Error (mV)')
    ax2.set_title(f'Prediction Error (RMS: {np.sqrt(np.mean(error**2))*1000:.2f} mV)')
    ax2.grid(True)
    
    # Frequency comparison
    ax3 = axes[1, 0]
    freqs1, mag1 = compute_fft(waveform1, sample_rate)
    freqs2, mag2 = compute_fft(waveform2, sample_rate)
    ax3.semilogy(freqs1/1e3, mag1, 'b-', label=labels[0], alpha=0.7)
    ax3.semilogy(freqs2/1e3, mag2, 'r--', label=labels[1], alpha=0.7)
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Spectrum Comparison')
    ax3.set_xlim([0, 500])
    ax3.legend()
    ax3.grid(True)
    
    # Metrics comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    ripple1 = compute_ripple(waveform1)
    ripple2 = compute_ripple(waveform2)
    
    metrics_text = f"""
    COMPARISON
    ==========
    
                    {labels[0]:>12}  {labels[1]:>12}
    DC Value (V):   {ripple1['dc_value']:>12.3f}  {ripple2['dc_value']:>12.3f}
    Ripple (mV):    {ripple1['pp_ripple']*1000:>12.2f}  {ripple2['pp_ripple']*1000:>12.2f}
    Ripple (%):     {ripple1['pp_ripple_percent']:>12.2f}  {ripple2['pp_ripple_percent']:>12.2f}
    
    MSE:            {np.mean((waveform1 - waveform2)**2):.6f}
    MAE:            {np.mean(np.abs(waveform1 - waveform2)):.6f}
    Max Error:      {np.max(np.abs(waveform1 - waveform2)):.6f}
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace',
             verticalalignment='center')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Test utilities
if __name__ == '__main__':
    # Generate test waveform (simulated buck converter output)
    sample_rate = 1e6  # 1 MHz sampling
    t = np.arange(0, 0.001, 1/sample_rate)  # 1 ms
    
    # DC + ripple + some transient at start
    dc = 5.0
    ripple_freq = 100e3  # 100 kHz switching
    ripple_amp = 0.05
    
    waveform = dc + ripple_amp * np.sin(2 * np.pi * ripple_freq * t)
    # Add transient
    waveform[:100] = np.linspace(0, dc, 100) + 0.5 * np.exp(-np.arange(100)/20)
    
    # Analyze
    plot_waveform_analysis(waveform, sample_rate, "Test Buck Converter Output")
    
    # Compare with noisy version
    noisy = waveform + np.random.normal(0, 0.01, len(waveform))
    compare_waveforms(waveform, noisy, ('Original', 'Noisy'), sample_rate)
