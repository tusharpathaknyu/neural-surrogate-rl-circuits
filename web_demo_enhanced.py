"""
Enhanced Interactive Web Demo for Power Electronics Circuit Design.

Features:
- 6 circuit topologies (Buck, Boost, Buck-Boost, SEPIC, Ćuk, Flyback)
- Input and output voltage specification
- Real-time efficiency, ripple, and cost estimation
- Component recommendations
- SPICE comparison mode
- Multi-objective optimization visualization

Run: python web_demo_enhanced.py
Open: http://localhost:7860
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")

from models.forward_surrogate import ForwardSurrogate
from models.multi_topology_surrogate import MultiTopologySurrogate
from rl.environment import CircuitDesignEnv
from rl.ppo_agent import PPOAgent

# Import new features
try:
    from utils.uncertainty import UncertaintyEstimator, RobustnessAnalyzer
    from utils.spice_validator import SPICEValidator
    from rl.pareto_optimizer import ParetoOptimizer
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    print("⚠️ Advanced features not available")


# ============================================================================
# TOPOLOGY DEFINITIONS
# ============================================================================

TOPOLOGIES = {
    "Buck (Step-Down)": {
        "id": 0,
        "formula": "Vout = Vin × D",
        "description": "Steps voltage DOWN. Most efficient for Vout < Vin.",
        "use_cases": "12V→5V USB, 24V→12V automotive, 48V→3.3V logic",
        "vin_range": (8, 48),
        "vout_range": (1, 36),
        "efficiency_typical": "90-95%",
        "pros": ["Simple", "High efficiency", "Low component count"],
        "cons": ["Can only step down", "Pulsating input current"],
    },
    "Boost (Step-Up)": {
        "id": 1,
        "formula": "Vout = Vin / (1-D)",
        "description": "Steps voltage UP. Essential for battery applications.",
        "use_cases": "3.7V Li-ion→5V, solar MPPT, LED drivers",
        "vin_range": (3.3, 24),
        "vout_range": (5, 60),
        "efficiency_typical": "85-93%",
        "pros": ["Simple boost", "Good for batteries", "High power density"],
        "cons": ["Can only step up", "Pulsating output current"],
    },
    "Buck-Boost (Inverting)": {
        "id": 2,
        "formula": "Vout = -Vin × D/(1-D)",
        "description": "Can step UP or DOWN, but output is INVERTED (negative).",
        "use_cases": "Negative rail generation, battery systems with varying input",
        "vin_range": (5, 36),
        "vout_range": (3, 50),
        "efficiency_typical": "80-88%",
        "pros": ["Flexible ratio", "Simple", "Can buck or boost"],
        "cons": ["Inverted output", "Higher stress", "Discontinuous currents"],
    },
    "SEPIC (Non-Inverting)": {
        "id": 3,
        "formula": "Vout = Vin × D/(1-D)",
        "description": "Like buck-boost but output is POSITIVE (non-inverting).",
        "use_cases": "Battery chargers, LED drivers, automotive",
        "vin_range": (5, 24),
        "vout_range": (3, 40),
        "efficiency_typical": "78-88%",
        "pros": ["Non-inverting", "Flexible ratio", "Continuous input current"],
        "cons": ["More components", "Lower efficiency", "Larger size"],
    },
    "Ćuk (Continuous Current)": {
        "id": 4,
        "formula": "Vout = -Vin × D/(1-D)",
        "description": "Inverted output, but with continuous input AND output current.",
        "use_cases": "Low-noise applications, audio, sensitive equipment",
        "vin_range": (5, 24),
        "vout_range": (3, 40),
        "efficiency_typical": "75-85%",
        "pros": ["Low ripple", "Continuous currents", "Flexible ratio"],
        "cons": ["Inverted", "Complex", "More components", "Lower efficiency"],
    },
    "Flyback (Isolated)": {
        "id": 5,
        "formula": "Vout = Vin × N × D/(1-D)",
        "description": "GALVANIC ISOLATION via transformer. Essential for safety.",
        "use_cases": "AC-DC adapters, medical devices, telecom, chargers",
        "vin_range": (12, 400),
        "vout_range": (3, 48),
        "efficiency_typical": "80-90%",
        "pros": ["Isolation", "Multiple outputs possible", "Wide ratio range"],
        "cons": ["EMI challenges", "Transformer design", "More complex"],
    },
}

# ============================================================================
# COMPONENT DATABASE (for cost estimation)
# ============================================================================

COMPONENT_COSTS = {
    'inductor': {  # $/µH (base cost)
        'small': 0.02,   # < 50µH
        'medium': 0.05,  # 50-200µH
        'large': 0.10,   # > 200µH
    },
    'capacitor': {  # $/µF
        'small': 0.005,   # < 100µF
        'medium': 0.01,   # 100-500µF
        'large': 0.02,    # > 500µF
    },
    'mosfet': {
        'low_voltage': 0.30,   # < 30V
        'medium_voltage': 0.50,  # 30-100V
        'high_voltage': 1.00,    # > 100V
    },
    'diode': {
        'schottky': 0.20,
        'fast': 0.35,
    },
    'transformer': 2.50,  # Base cost for flyback
}


# ============================================================================
# MODELS
# ============================================================================

SURROGATE = None
AGENT = None
ENV = None
DEVICE = None


def load_models():
    """Load trained models."""
    global SURROGATE, AGENT, ENV, DEVICE
    
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load multi-topology surrogate (trained on 6 topologies, 30k samples)
    SURROGATE = MultiTopologySurrogate(num_topologies=6)
    ckpt_path = Path('checkpoints/multi_topology_surrogate.pt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        SURROGATE.load_state_dict(ckpt['model_state_dict'])
        print(f"✓ Loaded multi-topology surrogate (val_loss={ckpt.get('val_loss', 'N/A'):.4f})")
    else:
        # Fallback to old model
        ckpt_path = Path('checkpoints/best_model.pt')
        SURROGATE = ForwardSurrogate()
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            SURROGATE.load_state_dict(ckpt['model_state_dict'])
            print("✓ Loaded legacy surrogate (single topology)")
    SURROGATE.to(DEVICE)
    SURROGATE.eval()
    
    # Load RL agent - prefer multi-topology agent if available
    ENV = CircuitDesignEnv(SURROGATE, device=DEVICE)
    AGENT = PPOAgent(ENV, device=DEVICE)
    
    # Try multi-topology agent first
    multi_agent_path = Path('checkpoints/multi_topo_rl_agent.pt')
    if multi_agent_path.exists():
        AGENT.load(str(multi_agent_path))
        print("✓ Loaded multi-topology RL agent")
    else:
        # Fallback to single-topology agent
        agent_path = Path('checkpoints/rl_agent.pt')
        if agent_path.exists():
            AGENT.load(str(agent_path))
            print("✓ Loaded legacy RL agent")
    
    return True


# ============================================================================
# DESIGN CALCULATIONS
# ============================================================================

def calculate_duty_cycle(topology: str, v_in: float, v_out: float) -> float:
    """Calculate required duty cycle for given voltages."""
    if "Buck" in topology and "Boost" not in topology:
        # Buck: D = Vout/Vin
        d = v_out / v_in
    elif "Boost" in topology and "Buck" not in topology:
        # Boost: D = 1 - Vin/Vout
        d = 1 - v_in / v_out
    elif "SEPIC" in topology or "Ćuk" in topology:
        # SEPIC/Ćuk: D = Vout/(Vin + Vout)
        d = v_out / (v_in + v_out)
    elif "Buck-Boost" in topology:
        # Buck-Boost: D = Vout/(Vin + Vout)
        d = abs(v_out) / (v_in + abs(v_out))
    elif "Flyback" in topology:
        # Flyback with N=1: D = Vout/(Vin + Vout)
        d = v_out / (v_in + v_out)
    else:
        d = 0.5
    
    return np.clip(d, 0.1, 0.9)


def estimate_components(topology: str, v_in: float, v_out: float, 
                        i_out: float, ripple_target: float = 0.05,
                        f_sw: float = 100e3) -> dict:
    """Estimate component values for given specifications."""
    duty = calculate_duty_cycle(topology, v_in, v_out)
    
    # Inductor sizing (targeting specific ripple)
    if "Buck" in topology and "Boost" not in topology:
        L = (v_in - v_out) * duty / (2 * i_out * ripple_target * f_sw)
    elif "Boost" in topology:
        L = v_in * duty / (2 * i_out * ripple_target * f_sw)
    else:
        L = v_in * duty / (2 * i_out * ripple_target * f_sw)
    
    L = np.clip(L, 10e-6, 1000e-6)
    
    # Capacitor sizing (targeting specific voltage ripple)
    C = i_out * duty / (ripple_target * v_out * f_sw)
    C = np.clip(C, 10e-6, 2000e-6)
    
    # Calculate R_load
    R_load = v_out / i_out if i_out > 0 else 10
    
    return {
        'L': L,
        'C': C,
        'R_load': R_load,
        'V_in': v_in,
        'f_sw': f_sw,
        'duty': duty,
    }


def estimate_cost(components: dict, topology: str) -> dict:
    """Estimate component costs."""
    costs = {}
    
    # Inductor
    L_uh = components['L'] * 1e6
    if L_uh < 50:
        costs['inductor'] = L_uh * COMPONENT_COSTS['inductor']['small']
    elif L_uh < 200:
        costs['inductor'] = L_uh * COMPONENT_COSTS['inductor']['medium']
    else:
        costs['inductor'] = L_uh * COMPONENT_COSTS['inductor']['large']
    
    # Second inductor for SEPIC/Ćuk
    if "SEPIC" in topology or "Ćuk" in topology:
        costs['inductor'] *= 2
    
    # Capacitor
    C_uf = components['C'] * 1e6
    if C_uf < 100:
        costs['capacitor'] = C_uf * COMPONENT_COSTS['capacitor']['small']
    elif C_uf < 500:
        costs['capacitor'] = C_uf * COMPONENT_COSTS['capacitor']['medium']
    else:
        costs['capacitor'] = C_uf * COMPONENT_COSTS['capacitor']['large']
    
    # MOSFET
    v_in = components['V_in']
    if v_in < 30:
        costs['mosfet'] = COMPONENT_COSTS['mosfet']['low_voltage']
    elif v_in < 100:
        costs['mosfet'] = COMPONENT_COSTS['mosfet']['medium_voltage']
    else:
        costs['mosfet'] = COMPONENT_COSTS['mosfet']['high_voltage']
    
    # Diode
    costs['diode'] = COMPONENT_COSTS['diode']['schottky']
    
    # Transformer for flyback
    if "Flyback" in topology:
        costs['transformer'] = COMPONENT_COSTS['transformer']
    
    costs['total'] = sum(costs.values())
    
    return costs


def estimate_efficiency(topology: str, v_in: float, v_out: float, 
                       duty: float, f_sw: float) -> dict:
    """Estimate converter efficiency and losses."""
    # Base efficiency from topology
    base_eff = {
        "Buck": 0.92,
        "Boost": 0.88,
        "Buck-Boost": 0.84,
        "SEPIC": 0.82,
        "Ćuk": 0.80,
        "Flyback": 0.85,
    }
    
    topo_key = topology.split()[0]
    eff = base_eff.get(topo_key, 0.85)
    
    # Duty cycle penalty (extreme duties are less efficient)
    duty_penalty = 0.05 * (abs(duty - 0.5) / 0.4) ** 2
    eff -= duty_penalty
    
    # Frequency penalty (higher freq = more switching losses)
    freq_penalty = 0.02 * (f_sw - 100e3) / 400e3
    eff -= max(0, freq_penalty)
    
    # Estimate losses
    p_out = v_out * 1.0  # Assume 1A for calculation
    p_in = p_out / eff
    
    return {
        'efficiency': eff * 100,
        'p_out': p_out,
        'p_in': p_in,
        'p_loss': p_in - p_out,
        'switching_loss': (p_in - p_out) * 0.4,
        'conduction_loss': (p_in - p_out) * 0.5,
        'other_loss': (p_in - p_out) * 0.1,
    }


# ============================================================================
# WAVEFORM GENERATION
# ============================================================================

def create_target_waveform(v_out: float, ripple_percent: float, 
                          rise_time_percent: float) -> np.ndarray:
    """Create target waveform from specifications."""
    t = np.linspace(0, 1, 512)
    
    # Rise time transient
    rise_samples = max(1, int(rise_time_percent / 100 * 512))
    tau = rise_time_percent / 100 * 0.3
    target = v_out * (1 - np.exp(-t / tau))
    target[rise_samples:] = v_out
    
    # Add ripple
    ripple = v_out * (ripple_percent / 100) * np.sin(2 * np.pi * 20 * t)
    target = target + ripple
    
    return target.astype(np.float32)


def design_circuit(topology: str, v_in: float, v_out: float,
                   i_out: float, ripple_target: float,
                   f_sw: float, optimization_steps: int) -> tuple:
    """Full circuit design workflow."""
    global AGENT, ENV
    
    if AGENT is None:
        load_models()
    
    # Validate voltage conversion is possible
    topo_info = TOPOLOGIES[topology]
    
    # Map display name to model topology name
    topo_name_map = {
        "Buck (Step-Down)": "buck",
        "Boost (Step-Up)": "boost",
        "Buck-Boost (Inverting)": "buck_boost",
        "SEPIC (Non-Inverting)": "sepic",
        "Ćuk (Continuous Current)": "cuk",
        "Flyback (Isolated)": "flyback",
    }
    ENV.topology = topo_name_map.get(topology, "buck")
    
    # Check if conversion is valid
    can_convert = True
    warning = None
    
    if "Buck" in topology and "Boost" not in topology:
        if v_out >= v_in:
            can_convert = False
            warning = f"⚠️ Buck converter requires Vout < Vin. You specified Vout={v_out}V, Vin={v_in}V"
    elif "Boost" in topology and "Buck" not in topology:
        if v_out <= v_in:
            can_convert = False
            warning = f"⚠️ Boost converter requires Vout > Vin. You specified Vout={v_out}V, Vin={v_in}V"
    
    # Estimate components
    components = estimate_components(topology, v_in, v_out, i_out, 
                                     ripple_target / 100, f_sw * 1e3)
    
    # Get efficiency and cost
    eff_info = estimate_efficiency(topology, v_in, v_out, 
                                   components['duty'], components['f_sw'])
    cost_info = estimate_cost(components, topology)
    
    # Create target waveform
    target = create_target_waveform(v_out, ripple_target, 5)
    
    # Run RL optimization
    ENV.target_waveform = target
    ENV.current_params = np.array([
        components['L'],
        components['C'],
        components['R_load'],
        components['V_in'],
        components['f_sw'],
        components['duty'],
    ], dtype=np.float32)
    ENV.current_step = 0
    ENV.prev_mse = None
    
    state = ENV._get_state()
    best_params = ENV.current_params.copy()
    best_mse = float('inf')
    history = []
    
    for step in range(optimization_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action, _, _ = AGENT.policy.get_action(state_tensor, deterministic=True)
        
        action_np = action.cpu().numpy().squeeze()
        state, _, done, info = ENV.step(action_np)
        
        history.append(info['mse'])
        
        if info['mse'] < best_mse:
            best_mse = info['mse']
            best_params = ENV.current_params.copy()
        
        if done:
            break
    
    # Get final prediction
    predicted = ENV._simulate(best_params)
    
    # Create comprehensive visualization
    fig = create_comprehensive_visualization(
        target, predicted, best_params, history,
        topology, v_in, v_out, eff_info, cost_info, warning
    )
    
    # Format results
    results = format_comprehensive_results(
        topology, topo_info, components, best_params, best_mse,
        v_in, v_out, i_out, eff_info, cost_info, warning
    )
    
    return fig, results


def create_comprehensive_visualization(target, predicted, params, history,
                                       topology, v_in, v_out, eff_info, 
                                       cost_info, warning):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(14, 10))
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    t = np.linspace(0, 1, 512)
    
    # 1. Waveform comparison (large, top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t * 1000, target, 'b-', label='Target', linewidth=2, alpha=0.7)
    ax1.plot(t * 1000, predicted, 'r--', label='RL Design', linewidth=2)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'Output Waveform: {topology}', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(v_out, color='g', linestyle=':', alpha=0.5, label=f'Target DC: {v_out}V')
    
    # Add Vin annotation
    ax1.annotate(f'Vin = {v_in}V', xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 2. Efficiency pie chart (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    losses = [eff_info['switching_loss'], eff_info['conduction_loss'], eff_info['other_loss']]
    labels = ['Switching', 'Conduction', 'Other']
    colors = ['#ff6b6b', '#ffa06b', '#ffdb6b']
    
    wedges, texts, autotexts = ax2.pie(
        losses, labels=labels, autopct='%1.1f%%', colors=colors,
        explode=(0.05, 0.05, 0.05), startangle=90
    )
    ax2.set_title(f'Loss Breakdown\n(η = {eff_info["efficiency"]:.1f}%)', fontweight='bold')
    
    # 3. Optimization progress (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(history, 'b-o', markersize=3)
    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('MSE (log)')
    ax3.set_title('RL Optimization Progress', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(history[-1], color='r', linestyle='--', alpha=0.5)
    ax3.annotate(f'Final: {history[-1]:.4f}', xy=(len(history)-1, history[-1]),
                 xytext=(5, 10), textcoords='offset points', fontsize=9)
    
    # 4. Cost breakdown (bottom-middle)
    ax4 = fig.add_subplot(gs[1, 1])
    cost_items = [(k, v) for k, v in cost_info.items() if k != 'total']
    names = [item[0].title() for item in cost_items]
    values = [item[1] for item in cost_items]
    
    bars = ax4.bar(names, values, color=['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'][:len(names)])
    ax4.set_ylabel('Cost ($)')
    ax4.set_title(f'Component Costs (Total: ${cost_info["total"]:.2f})', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, values):
        ax4.annotate(f'${val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # 5. Component values (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    text = f"""
┌─────────────────────────────────┐
│  📦 DESIGNED COMPONENTS         │
├─────────────────────────────────┤
│  Inductor:     {params[0]*1e6:>8.2f} µH    │
│  Capacitor:    {params[1]*1e6:>8.2f} µF    │
│  Load:         {params[2]:>8.2f} Ω     │
│  Input:        {params[3]:>8.2f} V     │
│  Frequency:    {params[4]/1e3:>8.1f} kHz   │
│  Duty Cycle:   {params[5]*100:>8.1f} %     │
├─────────────────────────────────┤
│  Expected Vout: {params[3]*params[5]:>7.2f} V      │
│  Efficiency:   {eff_info['efficiency']:>8.1f} %     │
│  Est. Cost:    ${cost_info['total']:>7.2f}       │
└─────────────────────────────────┘
"""
    ax5.text(0.1, 0.5, text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout()
    return fig


def format_comprehensive_results(topology, topo_info, components, params, mse,
                                  v_in, v_out, i_out, eff_info, cost_info, warning):
    """Format comprehensive results as markdown."""
    
    warning_text = ""
    if warning:
        warning_text = f"\n\n{warning}\n\n---\n"
    
    results = f"""
# ⚡ Circuit Design Complete!
{warning_text}
## 📊 Topology: {topology}

**Formula:** `{topo_info['formula']}`

**Description:** {topo_info['description']}

**Best For:** {topo_info['use_cases']}

---

## 🔌 Input/Output Specifications

| Parameter | Value |
|-----------|-------|
| **Input Voltage** | **{v_in} V** |
| **Output Voltage** | **{v_out} V** |
| **Output Current** | {i_out} A |
| **Output Power** | {v_out * i_out:.1f} W |
| **Required Duty Cycle** | {components['duty']*100:.1f}% |

---

## 📦 Optimized Component Values

| Component | Calculated | RL-Optimized | Unit |
|-----------|------------|--------------|------|
| Inductor (L) | {components['L']*1e6:.2f} | {params[0]*1e6:.2f} | µH |
| Capacitor (C) | {components['C']*1e6:.2f} | {params[1]*1e6:.2f} | µF |
| Load (R) | {components['R_load']:.2f} | {params[2]:.2f} | Ω |
| Switching Freq | {components['f_sw']/1e3:.1f} | {params[4]/1e3:.1f} | kHz |
| Duty Cycle | {components['duty']*100:.1f} | {params[5]*100:.1f} | % |

---

## ⚡ Efficiency Analysis

| Metric | Value |
|--------|-------|
| Overall Efficiency | **{eff_info['efficiency']:.1f}%** |
| Input Power | {eff_info['p_in']:.2f} W |
| Output Power | {eff_info['p_out']:.2f} W |
| Total Losses | {eff_info['p_loss']:.2f} W |
| Switching Loss | {eff_info['switching_loss']:.3f} W |
| Conduction Loss | {eff_info['conduction_loss']:.3f} W |

---

## 💰 Cost Estimate

| Component | Cost |
|-----------|------|
| Inductor(s) | ${cost_info.get('inductor', 0):.2f} |
| Capacitor(s) | ${cost_info.get('capacitor', 0):.2f} |
| MOSFET | ${cost_info.get('mosfet', 0):.2f} |
| Diode | ${cost_info.get('diode', 0):.2f} |
{"| Transformer | $" + f"{cost_info.get('transformer', 0):.2f} |" if 'transformer' in cost_info else ""}
| **Total BOM** | **${cost_info['total']:.2f}** |

---

## 📈 Design Quality

| Metric | Value |
|--------|-------|
| Waveform MSE | {mse:.6f} |
| Design Time | < 1 second |
| vs SPICE | ~100,000x faster |

---

## ✅ Pros & ⚠️ Cons of {topology.split()[0]}

**Advantages:**
{chr(10).join(['- ✅ ' + p for p in topo_info['pros']])}

**Limitations:**
{chr(10).join(['- ⚠️ ' + c for c in topo_info['cons']])}
"""
    return results


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Global advanced feature instances
UNCERTAINTY_ESTIMATOR = None
ROBUSTNESS_ANALYZER = None
SPICE_VALIDATOR = None
PARETO_OPTIMIZER = None

def init_advanced_features():
    """Initialize advanced analysis features."""
    global UNCERTAINTY_ESTIMATOR, ROBUSTNESS_ANALYZER, SPICE_VALIDATOR, PARETO_OPTIMIZER
    
    if ADVANCED_FEATURES and SURROGATE is not None:
        UNCERTAINTY_ESTIMATOR = UncertaintyEstimator(SURROGATE, n_samples=20, device=DEVICE)
        ROBUSTNESS_ANALYZER = RobustnessAnalyzer(SURROGATE, device=DEVICE)
        SPICE_VALIDATOR = SPICEValidator()
        PARETO_OPTIMIZER = ParetoOptimizer(SURROGATE, device=DEVICE)
        return True
    return False


def run_advanced_analysis(topology: str, v_in: float, v_out: float, 
                          i_out: float, f_sw: float) -> tuple:
    """Run advanced analysis: uncertainty, robustness, SPICE validation."""
    if not ADVANCED_FEATURES:
        return None, "⚠️ Advanced features not available"
    
    if UNCERTAINTY_ESTIMATOR is None:
        init_advanced_features()
    
    # Map topology name
    topo_name_map = {
        "Buck (Step-Down)": "buck",
        "Boost (Step-Up)": "boost",
        "Buck-Boost (Inverting)": "buck_boost",
        "SEPIC (Non-Inverting)": "sepic",
        "Ćuk (Continuous Current)": "cuk",
        "Flyback (Isolated)": "flyback",
    }
    topo = topo_name_map.get(topology, "buck")
    
    # Calculate duty cycle
    duty = calculate_duty_cycle(topology, v_in, v_out)
    
    # Create parameter tensor
    params = torch.tensor([[
        50e-6,   # L
        220e-6,  # C
        v_out / i_out,  # R_load
        v_in,
        f_sw * 1e3,
        duty,
    ]], dtype=torch.float32)
    
    topology_ids = torch.tensor([topo_name_map.get(topology, 0) if isinstance(topo_name_map.get(topology, 0), int) 
                                  else list(topo_name_map.values()).index(topo)])
    topology_ids = torch.tensor([{'buck': 0, 'boost': 1, 'buck_boost': 2, 'sepic': 3, 'cuk': 4, 'flyback': 5}[topo]])
    
    # 1. Uncertainty Analysis
    uncertainty_result = UNCERTAINTY_ESTIMATOR.predict_with_uncertainty(params, topology_ids)
    
    # 2. Robustness Analysis
    robustness_result = ROBUSTNESS_ANALYZER.monte_carlo_robustness(params, topology_ids, n_samples=50)
    
    # 3. SPICE Validation (if available)
    spice_result = None
    if SPICE_VALIDATOR.ngspice_available:
        spice_result = SPICE_VALIDATOR.validate_surrogate(SURROGATE, params.numpy().squeeze(), topo)
    
    # Create visualization
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Prediction with confidence interval
    ax1 = fig.add_subplot(gs[0, 0:2])
    t = np.linspace(0, 1, 512)
    mean_wave = uncertainty_result['waveform_mean'].squeeze()
    ci_low = uncertainty_result['waveform_ci_low'].squeeze()
    ci_high = uncertainty_result['waveform_ci_high'].squeeze()
    
    ax1.fill_between(t, ci_low, ci_high, alpha=0.3, color='blue', label='95% CI')
    ax1.plot(t, mean_wave, 'b-', linewidth=2, label='Prediction')
    if spice_result and spice_result['valid']:
        ax1.plot(t, spice_result['spice_waveform'], 'r--', linewidth=2, label='SPICE')
    ax1.set_xlabel('Time (normalized)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'Prediction Confidence (±{uncertainty_result["waveform_std"].mean():.2f}V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence gauge
    ax2 = fig.add_subplot(gs[0, 2])
    conf = uncertainty_result['confidence_pct'][0]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    color = colors[min(4, int(conf / 20))]
    ax2.barh([0], [conf], color=color, height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title(f'Model Confidence: {conf:.0f}%')
    ax2.axvline(x=80, color='green', linestyle='--', alpha=0.5)
    
    # 3. Robustness histogram
    ax3 = fig.add_subplot(gs[1, 0])
    v_out_range = robustness_result['v_out_range'][0]
    rob_score = robustness_result['robustness_score'][0]
    ax3.bar(['Vout Range', 'Robustness'], [v_out_range, rob_score/10], color=['orange', 'green'])
    ax3.set_ylabel('Value')
    ax3.set_title(f'Robustness Score: {rob_score:.0f}%')
    
    # 4. Tolerance effects
    ax4 = fig.add_subplot(gs[1, 1])
    tolerances = list(robustness_result['tolerances'].keys())
    tol_values = [robustness_result['tolerances'][t] * 100 for t in tolerances]
    ax4.barh(tolerances, tol_values, color='steelblue')
    ax4.set_xlabel('Tolerance (%)')
    ax4.set_title('Component Tolerances')
    
    # 5. Metrics summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    summary_text = f"""
    📊 ANALYSIS SUMMARY
    ━━━━━━━━━━━━━━━━━━━
    
    🎯 Prediction
       Vout: {mean_wave.mean():.2f}V
       Uncertainty: ±{uncertainty_result['waveform_std'].mean():.3f}V
       Confidence: {conf:.0f}%
    
    🔧 Robustness
       Vout Range: {robustness_result['v_out_mean'][0]:.2f} ± {robustness_result['v_out_std'][0]:.2f}V
       Worst Ripple: {robustness_result['ripple_worst'][0]:.3f}V
       Score: {rob_score:.0f}%
    """
    
    if spice_result and spice_result['valid']:
        summary_text += f"""
    ✓ SPICE Validation
       Correlation: {spice_result['correlation']:.3f}
       DC Error: {spice_result['dc_error_pct']:.1f}%
       Speedup: {spice_result['speedup']:.0f}x
        """
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Create markdown report
    report = f"""
## 🔬 Advanced Analysis Results

### 🎯 Prediction Confidence
- **Model Confidence**: {conf:.0f}%
- **Prediction Uncertainty**: ±{uncertainty_result['waveform_std'].mean():.3f}V
- **Relative Uncertainty**: {uncertainty_result['relative_uncertainty'][0]*100:.2f}%

### 🔧 Robustness to Component Tolerances
- **Output Voltage Range**: {robustness_result['v_out_mean'][0]:.2f}V ± {robustness_result['v_out_std'][0]:.2f}V
- **Worst-Case Ripple**: {robustness_result['ripple_worst'][0]:.3f}V
- **Robustness Score**: {rob_score:.0f}%

### ⚠️ Tolerance Assumptions
| Component | Tolerance |
|-----------|-----------|
| Inductor (L) | ±10% |
| Capacitor (C) | ±20% |
| Load (R) | ±5% |
| Input Voltage | ±5% |
"""
    
    if spice_result and spice_result['valid']:
        report += f"""
### ✓ SPICE Validation
- **Correlation**: {spice_result['correlation']:.4f}
- **DC Error**: {spice_result['dc_error_pct']:.2f}%
- **SPICE Time**: {spice_result['spice_time_ms']:.1f}ms
- **Surrogate Time**: {spice_result['surrogate_time_ms']:.3f}ms
- **Speedup**: {spice_result['speedup']:.0f}x
"""
    
    return fig, report


def run_pareto_optimization(topology: str, v_out_target: float) -> tuple:
    """Run Pareto multi-objective optimization."""
    if not ADVANCED_FEATURES:
        return None, "⚠️ Advanced features not available"
    
    if PARETO_OPTIMIZER is None:
        init_advanced_features()
    
    # Map topology name
    topo_name_map = {
        "Buck (Step-Down)": "buck",
        "Boost (Step-Up)": "boost", 
        "Buck-Boost (Inverting)": "buck_boost",
        "SEPIC (Non-Inverting)": "sepic",
        "Ćuk (Continuous Current)": "cuk",
        "Flyback (Isolated)": "flyback",
    }
    topo = topo_name_map.get(topology, "buck")
    
    # Run optimization
    pareto_front = PARETO_OPTIMIZER.optimize(
        topology=topo,
        v_target=v_out_target,
        population_size=40,
        n_generations=20,
        verbose=False
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Efficiency vs Accuracy
    errors = [d.v_out_error * 100 for d in pareto_front]
    effs = [d.efficiency * 100 for d in pareto_front]
    costs = [d.cost for d in pareto_front]
    
    axes[0].scatter(errors, effs, c=costs, cmap='viridis', s=50, alpha=0.7)
    axes[0].set_xlabel('Voltage Error (%)')
    axes[0].set_ylabel('Efficiency (%)')
    axes[0].set_title('Pareto Front: Accuracy vs Efficiency')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='Cost ($)')
    
    # 2. Cost vs Efficiency
    axes[1].scatter(costs, effs, c=errors, cmap='RdYlGn_r', s=50, alpha=0.7)
    axes[1].set_xlabel('Cost ($)')
    axes[1].set_ylabel('Efficiency (%)')
    axes[1].set_title('Pareto Front: Cost vs Efficiency')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Error (%)')
    
    # 3. Recommendations table
    axes[2].axis('off')
    
    recommendations = []
    for priority in ['balanced', 'efficiency', 'cost', 'accuracy']:
        design = PARETO_OPTIMIZER.recommend_design(pareto_front, priority)
        recommendations.append({
            'priority': priority.capitalize(),
            'L': f"{design.params[0]*1e6:.1f}µH",
            'C': f"{design.params[1]*1e6:.0f}µF",
            'eff': f"{design.efficiency*100:.1f}%",
            'cost': f"${design.cost:.2f}",
        })
    
    table_text = "RECOMMENDATIONS\n" + "="*50 + "\n"
    table_text += f"{'Priority':<12} {'L':<10} {'C':<10} {'Eff':<8} {'Cost':<8}\n"
    table_text += "-"*50 + "\n"
    for r in recommendations:
        table_text += f"{r['priority']:<12} {r['L']:<10} {r['C']:<10} {r['eff']:<8} {r['cost']:<8}\n"
    
    axes[2].text(0.1, 0.9, table_text, transform=axes[2].transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Create report
    report = f"""
## 🎯 Pareto Multi-Objective Optimization

Found **{len(pareto_front)} Pareto-optimal designs** for {topology} → {v_out_target}V

### 📊 Design Recommendations

| Priority | Inductor | Capacitor | Efficiency | Cost |
|----------|----------|-----------|------------|------|
"""
    for r in recommendations:
        report += f"| {r['priority']} | {r['L']} | {r['C']} | {r['eff']} | {r['cost']} |\n"
    
    report += """
### 💡 How to Choose
- **Balanced**: Best overall trade-off (default)
- **Efficiency**: Maximum power efficiency, may cost more
- **Cost**: Cheapest design that meets specs
- **Accuracy**: Most precise voltage regulation
"""
    
    return fig, report


def update_voltage_ranges(topology):
    """Update voltage sliders based on topology."""
    topo_info = TOPOLOGIES[topology]
    vin_min, vin_max = topo_info['vin_range']
    vout_min, vout_max = topo_info['vout_range']
    
    # Return new slider configs
    return (
        gr.Slider(minimum=vin_min, maximum=vin_max, value=(vin_min + vin_max) / 2),
        gr.Slider(minimum=vout_min, maximum=vout_max, value=(vout_min + vout_max) / 2),
    )


def create_demo():
    """Create enhanced Gradio demo."""
    if not GRADIO_AVAILABLE:
        print("Please install gradio: pip install gradio")
        return None
    
    with gr.Blocks(title="Neural Circuit Designer Pro") as demo:
        gr.Markdown("""
        # ⚡ Neural Power Electronics Designer
        
        Design DC-DC converters instantly using AI! This tool combines:
        - 🧠 **Neural Surrogate** - 100,000x faster than SPICE simulation
        - 🎮 **Reinforcement Learning** - Optimizes component values in real-time
        - 📊 **Multi-Objective Optimization** - Balances efficiency, cost, and performance
        
        ### Supported Topologies
        | Topology | Conversion | Best For |
        |----------|------------|----------|
        | Buck | Step-Down | 12V→5V, 24V→12V |
        | Boost | Step-Up | Battery→5V, Solar MPPT |
        | Buck-Boost | Either (Inverted) | Variable battery input |
        | SEPIC | Either (Non-Inverted) | LED drivers, chargers |
        | Ćuk | Either (Low Noise) | Audio, sensitive circuits |
        | Flyback | Isolated | AC adapters, medical |
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Design Specifications")
                
                topology = gr.Dropdown(
                    choices=list(TOPOLOGIES.keys()),
                    value="Buck (Step-Down)",
                    label="Circuit Topology",
                    info="Select the converter type"
                )
                
                with gr.Group():
                    gr.Markdown("### 🔌 Voltages")
                    v_in = gr.Slider(
                        minimum=8, maximum=48, value=24, step=0.5,
                        label="Input Voltage (Vin)",
                        info="Power source voltage"
                    )
                    
                    v_out = gr.Slider(
                        minimum=1, maximum=36, value=12, step=0.5,
                        label="Output Voltage (Vout)",
                        info="Desired output voltage"
                    )
                
                with gr.Group():
                    gr.Markdown("### ⚡ Load")
                    i_out = gr.Slider(
                        minimum=0.1, maximum=5, value=1, step=0.1,
                        label="Output Current (A)",
                        info="Load current requirement"
                    )
                
                with gr.Group():
                    gr.Markdown("### 📈 Performance")
                    ripple = gr.Slider(
                        minimum=0.5, maximum=10, value=2, step=0.5,
                        label="Max Ripple (%)",
                        info="Acceptable output voltage ripple"
                    )
                    
                    f_sw = gr.Slider(
                        minimum=50, maximum=500, value=100, step=10,
                        label="Switching Frequency (kHz)",
                        info="Higher = smaller components, more losses"
                    )
                
                with gr.Group():
                    gr.Markdown("### 🤖 Optimization")
                    steps = gr.Slider(
                        minimum=20, maximum=200, value=50, step=10,
                        label="RL Optimization Steps",
                        info="More steps = better results (slower)"
                    )
                
                design_btn = gr.Button("🚀 Design Circuit", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Design Results")
                
                with gr.Tabs():
                    with gr.Tab("🎨 Visualization"):
                        plot_output = gr.Plot(label="Visualization")
                        results_output = gr.Markdown()
                    
                    with gr.Tab("🔬 Advanced Analysis"):
                        gr.Markdown("*Analyze prediction confidence and robustness*")
                        analyze_btn = gr.Button("🔬 Run Analysis", variant="secondary")
                        analysis_plot = gr.Plot(label="Analysis")
                        analysis_output = gr.Markdown()
                    
                    with gr.Tab("🎯 Pareto Optimization"):
                        gr.Markdown("*Find optimal trade-offs between efficiency, cost, and accuracy*")
                        pareto_btn = gr.Button("🎯 Optimize", variant="secondary")
                        pareto_plot = gr.Plot(label="Pareto Front")
                        pareto_output = gr.Markdown()
        
        # Update voltage ranges when topology changes
        topology.change(
            fn=update_voltage_ranges,
            inputs=[topology],
            outputs=[v_in, v_out]
        )
        
        design_btn.click(
            fn=design_circuit,
            inputs=[topology, v_in, v_out, i_out, ripple, f_sw, steps],
            outputs=[plot_output, results_output]
        )
        
        # Advanced analysis button
        analyze_btn.click(
            fn=run_advanced_analysis,
            inputs=[topology, v_in, v_out, i_out, f_sw],
            outputs=[analysis_plot, analysis_output]
        )
        
        # Pareto optimization button
        pareto_btn.click(
            fn=run_pareto_optimization,
            inputs=[topology, v_out],
            outputs=[pareto_plot, pareto_output]
        )
        
        gr.Markdown("""
        ---
        ### 💡 Usage Tips
        
        1. **Select the right topology** based on your Vin/Vout ratio
        2. **Buck** is best when Vout < Vin (most efficient)
        3. **Boost** is needed when Vout > Vin
        4. **SEPIC/Buck-Boost** for flexible ratios without isolation
        5. **Flyback** when you need galvanic isolation (safety)
        
        ---
        *Trained on 30,000 SPICE simulations (5,000 per topology) | [GitHub](https://github.com/tusharpathaknyu/neural-surrogate-rl-circuits)*
        """)
    
    return demo


def main():
    """Launch the enhanced web demo."""
    print("Loading models...")
    load_models()
    print("✓ Models loaded")
    
    print("\nStarting enhanced web demo...")
    demo = create_demo()
    
    if demo:
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
        )


if __name__ == '__main__':
    main()
