import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# --- SAFETY CHECK: Detect if run incorrectly ---
if __name__ == "__main__" and "streamlit" not in sys.modules:
    print("\nError: You are running this as a normal Python script.")
    print("-------------------------------------------------------")
    print("To view the web app, you must run it with the Streamlit command.")
    print("Please open your terminal and type:")
    print("\n    streamlit run fermi_app.py\n")
    print("-------------------------------------------------------")
    sys.exit()

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quantum Fermi Calculator",
    page_icon="⚛️",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PHYSICS LOGIC ---
def calculate_fermi_energy(density, atomic_mass, valency):
    """Calculates Fermi Energy (Ef) using Free Electron Model."""
    # Constants
    H_BAR = 1.0545718e-34     # Planck constant
    M_E = 9.10938356e-31      # Electron mass
    N_A = 6.02214076e23       # Avogadro's number
    EV_CONV = 1.60218e-19     # J to eV conversion

    if density <= 0 or atomic_mass <= 0: return 0.0

    # Conversions
    rho = density * 1000      # g/cm^3 -> kg/m^3
    M = atomic_mass / 1000    # g/mol -> kg/mol

    # Math
    n = (valency * rho * N_A) / M
    Ef_joules = ((H_BAR**2) / (2*M_E)) * (3 * math.pi**2 * n)**(2/3)
    
    return Ef_joules / EV_CONV

# --- 3. SIDEBAR (User Inputs) ---
st.sidebar.header("⚙️ Metal Settings")
metal_name = st.sidebar.text_input("Metal Name", value="My Alloy")
density = st.sidebar.number_input("Density (g/cm³)", value=8.96, min_value=0.1)
atomic_mass = st.sidebar.number_input("Atomic Mass (g/mol)", value=63.55, min_value=1.0)
valency = st.sidebar.slider("Valency (Free Electrons)", 1, 5, 1)

# --- 4. MAIN APP LAYOUT ---
st.title("⚛️ Fermi Energy Simulator")
st.write("Calculates the quantum energy levels of metals based on electron density.")

# Calculation
ef = calculate_fermi_energy(density, atomic_mass, valency)

# Display Metric
col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="Calculated Fermi Energy", value=f"{ef:.4f} eV")
with col2:
    if ef > 10:
        st.info(f"🔥 High Energy! This is similar to **Aluminum**.")
    elif ef < 6:
        st.info(f"❄️ Low Energy! This is similar to **Gold**.")
    else:
        st.info(f"⚖️ Medium Energy! This is similar to **Copper**.")

st.divider()

# --- 5. GRAPH VISUALIZATION ---
st.subheader("📊 Comparison with Standard Metals")

# Data Setup
metals = ['Copper', 'Silver', 'Gold', 'Aluminum', metal_name]
values = [7.00, 5.49, 5.53, 11.70, ef]
colors = ['#bdc3c7', '#bdc3c7', '#bdc3c7', '#bdc3c7', '#2ecc71'] # Gray vs Green

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(metals, values, color=colors)
ax.set_ylabel("Fermi Energy (eV)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add numbers on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

st.pyplot(fig)
