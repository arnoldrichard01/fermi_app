import streamlit as st
import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np



# ==========================================
# 1. DATABASE: Atomic Mass (amu) & Density (g/cm³)
# ==========================================
# Includes Alkali, Alkaline, Transition, Post-Transition, Lanthanides, Actinides
METAL_DATA = {
    # Period 2
    "Li": {"m": 6.94, "d": 0.534}, "Be": {"m": 9.01, "d": 1.85},
    # Period 3
    "Na": {"m": 22.99, "d": 0.968}, "Mg": {"m": 24.31, "d": 1.738}, "Al": {"m": 26.98, "d": 2.70},
    # Period 4
    "K": {"m": 39.10, "d": 0.89}, "Ca": {"m": 40.08, "d": 1.55}, "Sc": {"m": 44.96, "d": 2.985},
    "Ti": {"m": 47.87, "d": 4.506}, "V": {"m": 50.94, "d": 6.11}, "Cr": {"m": 52.00, "d": 7.19},
    "Mn": {"m": 54.94, "d": 7.21}, "Fe": {"m": 55.85, "d": 7.874}, "Co": {"m": 58.93, "d": 8.90},
    "Ni": {"m": 58.69, "d": 8.908}, "Cu": {"m": 63.55, "d": 8.96}, "Zn": {"m": 65.38, "d": 7.14},
    "Ga": {"m": 69.72, "d": 5.91},
    # Period 5
    "Rb": {"m": 85.47, "d": 1.532}, "Sr": {"m": 87.62, "d": 2.64}, "Y": {"m": 88.91, "d": 4.47},
    "Zr": {"m": 91.22, "d": 6.52}, "Nb": {"m": 92.91, "d": 8.57}, "Mo": {"m": 95.95, "d": 10.28},
    "Tc": {"m": 98.00, "d": 11.0}, "Ru": {"m": 101.07, "d": 12.45}, "Rh": {"m": 102.91, "d": 12.41},
    "Pd": {"m": 106.42, "d": 12.02}, "Ag": {"m": 107.87, "d": 10.49}, "Cd": {"m": 112.41, "d": 8.65},
    "In": {"m": 114.82, "d": 7.31}, "Sn": {"m": 118.71, "d": 7.265},
    # Period 6
    "Cs": {"m": 132.91, "d": 1.93}, "Ba": {"m": 137.33, "d": 3.51},
    "Hf": {"m": 178.49, "d": 13.31}, "Ta": {"m": 180.95, "d": 16.69}, "W": {"m": 183.84, "d": 19.25},
    "Re": {"m": 186.21, "d": 21.02}, "Os": {"m": 190.23, "d": 22.59}, "Ir": {"m": 192.22, "d": 22.56},
    "Pt": {"m": 195.08, "d": 21.45}, "Au": {"m": 196.97, "d": 19.30}, "Hg": {"m": 200.59, "d": 13.53},
    "Tl": {"m": 204.38, "d": 11.85}, "Pb": {"m": 207.2, "d": 11.34}, "Bi": {"m": 208.98, "d": 9.78},
    # Lanthanides (Sample)
    "La": {"m": 138.91, "d": 6.162}, "Gd": {"m": 157.25, "d": 7.90},
    # Default Fallback
    "default": {"m": 50.0, "d": 5.0}
}


# ==========================================
# 2. PHYSICS ENGINE
# ==========================================
def calculate_fermi_energy(symbol, valency):
    data = METAL_DATA.get(symbol, METAL_DATA["default"])
    
    # Constants
    h = const.h        # Planck constant
    m_e = const.m_e    # Electron mass
    N_A = const.Avogadro
    
    # 1. Number density of electrons (n)
    # n = (Density * N_A * Valency) / Molar Mass
    # Unit conversion: Density is g/cm³, need kg/m³ for SI? 
    # Let's stay in SI:
    rho_si = data['d'] * 1000  # g/cm³ -> kg/m³
    M_si = data['m'] / 1000    # g/mol -> kg/mol
    
    n = (rho_si * N_A * valency) / M_si  # electrons per m³
    
    # 2. Fermi Energy Formula (at T=0K)
    # Ef = (h^2 / 8*m_e) * (3*n / pi)^(2/3)
    Ef_joules = ((h**2) / (8 * m_e)) * ((3 * n) / np.pi)**(2/3)
    
    # Convert to eV
    Ef_eV = Ef_joules / const.e
    
    return Ef_eV, n, data['d'], data['m']


# ==========================================
# NEW DATA: Experimental Values (eV)
# ==========================================
# Literature values for validation (Approximate standard Solid State Physics values)
EXPERIMENTAL_VALUES = {
    # Alkali
    "Li": 4.74, "Na": 3.24, "K": 2.12, "Rb": 1.85, "Cs": 1.59,
    # Alkaline Earth
    "Be": 14.3, "Mg": 7.08, "Ca": 4.69, "Sr": 3.93, "Ba": 3.64,
    # Transition / Noble
    "Cu": 7.00, "Ag": 5.49, "Au": 5.53, "Fe": 11.1, "Zn": 9.47,
    "Al": 11.7, "Sn": 10.2, "Pb": 9.47, "Ga": 10.4, "In": 8.63,
    # Common reference placeholders for others not strictly standard in texts
    "Sc": 9.0, "Ti": 10.5, "V": 11.0, "Cr": 10.8, "Mn": 10.9, "Co": 10.5, "Ni": 11.7
}

# Reference metals for the comparison graph
COMMON_METALS_REF = {
    "Cu": 7.00,
    "Al": 11.7,
    "Fe": 11.1,
    "Au": 5.53
}

# ==========================================
# 3. UI: THE CALCULATOR POPUP (Detail View)
# ==========================================
def show_calculator_popup(symbol):
    # CSS for the card layout
    st.markdown("""
        <style>
            .calc-card {
                background-color: #2b2b2b !important;
                padding: 2rem;
                border-radius: 1px;
                border: 10px solid #444;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                color: white;
            }
            div[data-testid="stExpander"] details summary p{
            font-size: 1rem !important;      /* Make it slightly bigger */
            color:#f12e2e !important;         /* Streamlit Red (or use hex like #00FFFF) */
            font-family: 'Courier New', monospace; /* Techy font style */
            font-weight: bold !important;
            /*height: 3rem !important;   */       /* Increase height for better click area */
            text-align: center !important;  /* Center align text */
            }
            /* Optional: Change color on hover */
            .streamlit-expanderHeader:hover {
                color: #FF8888 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 1. Back Button
    if st.button("← Back to Periodic Table"):
        st.query_params.clear()
        st.rerun()

    # 2. Main Logic
    with st.container():
        st.markdown(f"## ⚛️ Fermi Energy Calculator: **{symbol}**")
        
        # Split layout: Inputs on Left, Graphs on Right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info("Configuration")
            # Dynamic Input: Valency
            valency = st.slider("Select Valency (Free Electrons per atom):", 
                              min_value=1, max_value=6, value=1, step=1)
            
            calculate = st.button("⚡ Calculate Fermi Energy", type="primary", use_container_width=True)
            
            # Show static data
            data = METAL_DATA.get(symbol, METAL_DATA["default"])
            st.markdown("---")
            st.markdown(f"**Atomic Data:**")
            st.write(f"• Mass: `{data['m']} amu`")
            st.write(f"• Density: `{data['d']} g/cm³`")
            
            # Fetch experimental value if available
            exp_val = EXPERIMENTAL_VALUES.get(symbol, None)
            if exp_val:
                 st.markdown(f"• Literature $E_F$: `{exp_val} eV`")
            else:
                 st.markdown(f"• Literature $E_F$: `N/A`")

        with col2:
            if calculate:
                # 1. Perform Calculation
                Ef_calc, n_density, rho, mass = calculate_fermi_energy(symbol, valency)
                
                # 2. Display Metrics
                st.success("Calculation Complete")
                m1, m2 = st.columns(2)
                m1.metric("Calculated Fermi Energy", f"{Ef_calc:.4f} eV")
                m2.metric("Electron Density", f"{n_density:.2e} m⁻³")
                
                # Set Dark Theme for Plots
                plt.style.use('dark_background')
                
                # --- GRAPH 1: Comparison with Common Metals ---
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                
                # Data preparation
                comp_labels = [symbol] + list(COMMON_METALS_REF.keys())
                comp_values = [Ef_calc] + list(COMMON_METALS_REF.values())
                
                # Colors: Selected metal gets Cyan, others get Gray
                colors = ['#3d9df333'] + ["#585858"] * len(COMMON_METALS_REF)
                
                bars = ax1.bar(comp_labels, comp_values, color=colors)
                ax1.set_ylabel("Fermi Energy (eV)")
                ax1.set_title(f"Comparison: {symbol} vs Common Metals")
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', color='white', fontsize=8)
                
                # Remove top/right spines for cleanliness
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                
                st.pyplot(fig1)
                
                # --- GRAPH 2: Experimental vs Calculated (% Error) ---
                if exp_val:
                    # Calculate % Error
                    error_pct = abs((Ef_calc - exp_val) / exp_val) * 100
                    
                    fig2, ax2 = plt.subplots(figsize=(6, 2))
                    
                    # Data
                    labels = ['Experimental', 'Calculated']
                    values = [exp_val, Ef_calc]
                    
                    # Plot
                    bars2 = ax2.barh(labels, values, color=["#4DC373", '#3d9df333']) # Magenta vs Cyan
                    
                    ax2.set_xlabel("Energy (eV)")
                    ax2.set_title(f"Accuracy Check: {error_pct:.2f}% Error")
                    ax2.set_xlim(0, max(values) * 1.3) # Give room for text
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    
                    # Annotate bars
                    for i, v in enumerate(values):
                        ax2.text(v + 0.2, i, f"{v:.2f} eV", va='center', color='white', fontweight='bold')
                    
                    st.pyplot(fig2)
                    with st.expander("❓ Why is there a % Error?"):
                        st.markdown("""
                        **1. The "Integer Trap"**
                        You selected a standard integer valency (e.g., 2), but nature is messy. 
                        Real metals often act like they have "1.8" or "2.2" free electrons because of complex crystal structures. 
                        Since we force a simple integer, the math is naturally off by ~5-10% for complex metals like Cobalt.

                        **2. The Quantum Free Electron Model Assumption**
                        Our formula assumes electrons float freely like gas in a jar. 
                        * **Low Error (e.g., Fe, Li):** The electrons behave very much like "free" particles (or errors cancel out).
                        * **High Error (e.g., Co, Mn):** These metals have complex "d-orbitals" (weird shapes) where electrons get stuck. The simple formula doesn't capture this complexity.
                        
                        *The error isn't a mistake—it's a measurement of how "complex" the metal's internal structure is!*
                        """)
                
                else:
                    st.warning(f"Experimental data for {symbol} not found in database. Cannot generate Error Graph.")
                
            
            else:
                st.info("Adjust the valency slider and click Calculate to see the graphs.")
    # ... (After the code for Graph 2) ...

                # --- EDUCATIONAL EXPANDER: WHY THE ERROR? ---
    

# ==========================================
# 3. THE PERIOIC TABLE  (Main View)
# ==========================================


def show_periodic_table():
    # 2. Element Data Structure
    # Each element: (Symbol, Period, Group, Category, Is_Metal)
    elements = [
    # Period 1
    ("H", 1, 1, "Non-metal", False), ("He", 1, 18, "Noble", False),
    
    # Period 2
    ("Li", 2, 1, "Alkali", True), ("Be", 2, 2, "Alkaline", True),
    ("B", 2, 13, "Metalloid", False), ("C", 2, 14, "Non-metal", False), ("N", 2, 15, "Non-metal", False), ("O", 2, 16, "Non-metal", False), ("F", 2, 17, "Halogen", False), ("Ne", 2, 18, "Noble", False),
    
    # Period 3
    ("Na", 3, 1, "Alkali", True), ("Mg", 3, 2, "Alkaline", True),
    ("Al", 3, 13, "Post-trans", True), ("Si", 3, 14, "Metalloid", False), ("P", 3, 15, "Non-metal", False), ("S", 3, 16, "Non-metal", False), ("Cl", 3, 17, "Halogen", False), ("Ar", 3, 18, "Noble", False),
    
    # Period 4
    ("K", 4, 1, "Alkali", True), ("Ca", 4, 2, "Alkaline", True), 
    ("Sc", 4, 3, "Trans", True), ("Ti", 4, 4, "Trans", True), ("V", 4, 5, "Trans", True), ("Cr", 4, 6, "Trans", True), ("Mn", 4, 7, "Trans", True), ("Fe", 4, 8, "Trans", True), ("Co", 4, 9, "Trans", True), ("Ni", 4, 10, "Trans", True), ("Cu", 4, 11, "Trans", True), ("Zn", 4, 12, "Trans", True), 
    ("Ga", 4, 13, "Post-trans", True), ("Ge", 4, 14, "Metalloid", False), ("As", 4, 15, "Metalloid", False), ("Se", 4, 16, "Non-metal", False), ("Br", 4, 17, "Halogen", False), ("Kr", 4, 18, "Noble", False),
    
    # Period 5
    ("Rb", 5, 1, "Alkali", True), ("Sr", 5, 2, "Alkaline", True), 
    ("Y", 5, 3, "Trans", True), ("Zr", 5, 4, "Trans", True), ("Nb", 5, 5, "Trans", True), ("Mo", 5, 6, "Trans", True), ("Tc", 5, 7, "Trans", True), ("Ru", 5, 8, "Trans", True), ("Rh", 5, 9, "Trans", True), ("Pd", 5, 10, "Trans", True), ("Ag", 5, 11, "Trans", True), ("Cd", 5, 12, "Trans", True), 
    ("In", 5, 13, "Post-trans", True), ("Sn", 5, 14, "Post-trans", True), ("Sb", 5, 15, "Metalloid", False), ("Te", 5, 16, "Metalloid", False), ("I", 5, 17, "Halogen", False), ("Xe", 5, 18, "Noble", False),
    
    # Period 6
    ("Cs", 6, 1, "Alkali", True), ("Ba", 6, 2, "Alkaline", True), ("57-71", 6, 3, "Lanthanide", True), # Placeholder
    ("Hf", 6, 4, "Trans", True), ("Ta", 6, 5, "Trans", True), ("W", 6, 6, "Trans", True), ("Re", 6, 7, "Trans", True), ("Os", 6, 8, "Trans", True), ("Ir", 6, 9, "Trans", True), ("Pt", 6, 10, "Trans", True), ("Au", 6, 11, "Trans", True), ("Hg", 6, 12, "Trans", True), 
    ("Tl", 6, 13, "Post-trans", True), ("Pb", 6, 14, "Post-trans", True), ("Bi", 6, 15, "Post-trans", True), ("Po", 6, 16, "Post-trans", True), ("At", 6, 17, "Metalloid", False), ("Rn", 6, 18, "Noble", False),
    
    # Period 7
    ("Fr", 7, 1, "Alkali", True), ("Ra", 7, 2, "Alkaline", True), ("89-103", 7, 3, "Actinide", True), # Placeholder
    ("Rf", 7, 4, "Trans", True), ("Db", 7, 5, "Trans", True), ("Sg", 7, 6, "Trans", True), ("Bh", 7, 7, "Trans", True), ("Hs", 7, 8, "Trans", True), ("Mt", 7, 9, "Trans", True), ("Ds", 7, 10, "Trans", True), ("Rg", 7, 11, "Trans", True), ("Cn", 7, 12, "Trans", True), 
    ("Nh", 7, 13, "Post-trans", True), ("Fl", 7, 14, "Post-trans", True), ("Mc", 7, 15, "Post-trans", True), ("Lv", 7, 16, "Post-trans", True), ("Ts", 7, 17, "Halogen", False), ("Og", 7, 18, "Noble", False)
    ]

    # Separate F-Block lists for bottom rows
    lanthanides = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
    actinides = ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

    # Color Mapping
    COLORS = {
      "Alkali": "#FF7A73", "Alkaline": "#FFDFBA", "Trans": "#FFFFBA",
      "Post-trans": "#BAFFC9", "Metalloid": "#BAE1FF", "Non-metal": "#A0C4FF",
      "Halogen": "#E2F0CB", "Noble": "#F0F0F0",
      "Lanthanide": "#FFC6FF", "Actinide": "#BDB2FF"
    }

    # 4. CSS Styling
    st.markdown("""
    <style>
    /* Compact the top space */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    
    /* Main Layout container to center the table */
    .pt-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Row styling */
    .pt-row {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-bottom: 2px;
    }

    /* Element Box */
    .element {
        width: 4.8%; /* Fits 18 cols with small gaps */
        aspect-ratio: 1 / 1;
        margin: 0.1%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: Arial, sans-serif;
        font-weight: bold;
        font-size: 0.8vw; /* Responsive font size */
        border-radius: 4px;
        box-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        text-decoration: none !important;
        color: #111 !important;
        transition: transform 0.1s ease;
    }

    /* Clickable vs Non-Clickable styling */
    .clickable:hover {
        transform: scale(1.15);
        z-index: 100;
        border: 1px solid #333;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    
    .disabled {
        opacity: 0.6;
        cursor: default;
        filter: grayscale(10%);
        pointer-events: none; /* Prevents clicking */
    }
    .title{
            color: #ff3636;
    }

    /* Gap handling */
    .empty-cell {
        width: 4.8%;
        margin: 0.1%;
        visibility: hidden;
    }
    
    /* F-Block Indentation */
    .f-block-gap { width: 15%; } /* Shifts the row to align with Group 3 */

    </style>
    """, unsafe_allow_html=True)

    # 5. Header
    #st.title("Interactive Periodic Table")
    st.markdown("<h2 style='text-align: center; color: #dedede; margin-top: 7px;'>Interactive Periodic Table</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555; margin-top: -15px;'>Metals are Clickable • Click to calculate the Fermi Energy</h4>", unsafe_allow_html=True)

    # 6. HTML Generation
    html_content = '<div class="pt-wrapper">'

    # --- MAIN BLOCK (Rows 1-7) ---
    grid = {}
    for e in elements:
        grid[(e[1], e[2])] = e # Map (Period, Group) -> Data

    for period in range(1, 8):
        html_content += '<div class="pt-row">'
        for group in range(1, 19):
            if (period, group) in grid:
                sym, _, _, cat, is_metal = grid[(period, group)]
                bg = COLORS.get(cat, "#FFF")
            
                if is_metal:
                # Clickable: Anchor tag with reload
                  html_content += f'<a href="?element={sym}" target="_self" class="element clickable" style="background-color: {bg};">{sym}</a>'
                else:
                # Static Div
                  html_content += f'<div class="element disabled" style="background-color: {bg};">{sym}</div>'
            else:
                # Empty space
                html_content += '<div class="empty-cell"></div>'
        html_content += '</div>'

    # --- SPACER ---
    html_content += '<div style="height: 15px;"></div>'

    # --- F-BLOCK (Rows 8-9) ---
    # Helper to render F-block rows
    def render_f_row(element_list, category):
        row_html = '<div class="pt-row">'
        # Empty spacer to align with Group 3
        row_html += '<div class="empty-cell" style="width: 14.5%;"></div>' 
    
        for sym in element_list:
            bg = COLORS[category]
            # All F-block are metals -> Clickable
            row_html += f'<a href="?element={sym}" target="_self" class="element clickable" style="background-color: {bg};">{sym}</a>'
    
        # Fill remaining space
        row_html += '<div class="empty-cell" style="flex-grow: 1;"></div>'
        row_html += '</div>'
        return row_html

    html_content += render_f_row(lanthanides, "Lanthanide")
    html_content += render_f_row(actinides, "Actinide")

    html_content += '</div>' # End Wrapper

    # 7. Render
    st.markdown(html_content, unsafe_allow_html=True)

    # 8. Legend
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px; flex-wrap: wrap;">
        <span style="background:#FFB3BA; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Alkali</span>
        <span style="background:#FFDFBA; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Alkaline</span>
        <span style="background:#FFFFBA; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Transition</span>
        <span style="background:#BAFFC9; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Post-Trans</span>
        <span style="background:#BAE1FF; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Metalloid</span>
        <span style="background:#A0C4FF; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Non-metal</span>
        <span style="background:#E2F0CB; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Halogen</span>
        <span style="background:#F0F0F0; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Noble</span>
        <span style="background:#FFC6FF; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Lanthanide</span>
        <span style="background:#BDB2FF; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Actinide</span>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. MAIN APP CONTROLLER
# ==========================================
def main():
    st.set_page_config(page_title="Fermi Calculator", layout="wide", initial_sidebar_state="collapsed")
    
    # Check if a metal is selected via URL params
    params = st.query_params
    
    if "element" in params:
        # Show "Popup" / Detail View
        show_calculator_popup(params["element"])
    else:
        # Show Master Table
        show_periodic_table()

if __name__ == "__main__":
    main()
    