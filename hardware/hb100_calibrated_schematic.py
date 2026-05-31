"""Render the clean calibrated HB100 vital-sign radar analog frontend schematic.

The script is intentionally written against schemdraw 0.23, which does not provide
Blackbox/CapacitorElectrolytic aliases. Dedicated HB100 and ESP32 component blocks
are represented with elm.Ic blocks.
"""

from __future__ import annotations

from pathlib import Path

import schemdraw
import schemdraw.elements as elm

OUT = Path(__file__).with_suffix(".svg")


def hb100_block() -> elm.Ic:
    return elm.Ic(
        size=(2.0, 2.0),
        pins=[
            elm.IcPin(name="", side="L", pos=0.78, anchorname="vin"),
            elm.IcPin(name="", side="L", pos=0.22, anchorname="gnd"),
            elm.IcPin(name="", side="R", pos=0.50, anchorname="ifout"),
        ],
    ).label("HB100\nRadar Module", "center")


def esp32_block() -> elm.Ic:
    return elm.Ic(
        size=(2.2, 2.5),
        pins=[
            elm.IcPin(name="", side="L", pos=0.52, anchorname="adc"),
            elm.IcPin(name="", side="B", pos=0.50, anchorname="gnd"),
        ],
    ).label("ESP32\nModule", "center")


def mcp1700_block() -> elm.Ic:
    return elm.Ic(
        size=(2.2, 1.35),
        pins=[
            elm.IcPin(name="", side="L", pos=0.50, anchorname="vin"),
            elm.IcPin(name="", side="R", pos=0.50, anchorname="vout"),
            elm.IcPin(name="", side="B", pos=0.50, anchorname="gnd"),
        ],
    ).label("MCP1700\n3.3V LDO", "center")


# Keep a white background so the schematic is readable in dark IDE previews.
d = schemdraw.Drawing(show=False, transparent=False)
d.config(unit=2.0, fontsize=8, bgcolor="white", margin=0.35)

# ====================================================
# 1. HB100 RADAR MODULE BLOCK
# ====================================================
hb100 = d.add(hb100_block())

# Left side supply indicators.
d += elm.Line().left().at(hb100.vin).length(0.45).label("5V (VIN)", "top")
d += elm.Line().down().at(hb100.gnd).length(0.45)
d += elm.Ground()

# Pull signal right from IF output terminal.
d += elm.Line().right().at(hb100.ifout).length(0.75).label("IF OUT", "top")
c1 = d.add(elm.Capacitor().right().label('C1\n1µF ("105")', "top"))

# ====================================================
# 2. MCP1700 LDO FOR CLEAN ANALOG POWER
# ====================================================
# Move far above the signal path; this prevents the power/virtual-ground rails
# from colliding with op-amp feedback labels and capacitors.
d.move_from(c1.end, dx=-6.80, dy=7.70)

vin_5v = d.add(elm.Dot().label("ESP32 5V\n(VBUS)", "left"))
d += elm.Capacitor().down().at(vin_5v.center)
d += elm.Ground()
d += elm.Label().at((vin_5v.center[0] - 0.75, vin_5v.center[1] - 1.20)).label('C_in\n1µF ("105")', "left")

# Regulate noisy USB/ESP32 5 V down to a local analog 3.3 V rail.
d += elm.Line().right().at(vin_5v.center).length(0.85)
ldo = d.add(mcp1700_block().right().anchor("vin"))
d += elm.Label().at((ldo.vin[0] + 0.15, ldo.vin[1] + 0.32)).label("VIN", "top")
d += elm.Label().at((ldo.vout[0] - 0.15, ldo.vout[1] + 0.32)).label("VOUT", "top")
d += elm.Line().down().at(ldo.gnd).length(0.35)
d += elm.Ground()

d += elm.Line().right().at(ldo.vout).length(0.80)
ldo_out = d.add(elm.Dot())
d += elm.Capacitor().down().at(ldo_out.center)
d += elm.Ground()
d += elm.Label().at((ldo_out.center[0] - 0.45, ldo_out.center[1] - 1.25)).label('C_out\n1µF ("105")', "left")

# ====================================================
# 3. SEPARATED VIRTUAL GROUND RAIL (1.65 V)
# ====================================================
d += elm.Line().right().at(ldo_out.center).length(2.20).label("3.3V_ANA", "top")
v_ana = d.add(elm.Dot())
d += elm.Resistor().down()
d += elm.Label().at((v_ana.center[0] + 0.70, v_ana.center[1] - 0.60)).label("R_bias1\n10kΩ", "right")
v_vg_node = d.add(elm.Dot())
d += elm.Resistor().down()
d += elm.Label().at((v_vg_node.center[0] + 0.70, v_vg_node.center[1] - 0.95)).label("R_bias2\n10kΩ", "right")
d += elm.Ground()

# Stabilizing capacitor for virtual ground.
d += elm.Line().left().at(v_vg_node.center).length(1.35)
d += elm.Capacitor().down()
d += elm.Ground()
d += elm.Label().at((v_vg_node.center[0] - 2.35, v_vg_node.center[1] - 1.25)).label("C_vg\n100µF", "left")

# Clean horizontal V_VG bus across the top.
d += elm.Line().right().at(v_vg_node.center).length(16.8).dot("end").label("V_VG Bus (1.65V)", "top")

# ====================================================
# 4. STAGE 1 AMPLIFIER (MCP6002A)
# ====================================================
# Return to the main input line path.
d.move_from(c1.end)
d += elm.Line().right().length(1.00)
sig1_in = d.add(elm.Dot())

# Pull 1M bias up to the overhead V_VG bus.
d += elm.Resistor().up().at(sig1_in.center).toy(v_vg_node.center)
d += elm.Label().at((sig1_in.center[0] + 1.15, (sig1_in.center[1] + v_vg_node.center[1]) / 2)).label("R_vbias1\n1MΩ", "center")

# Connect signal straight to non-inverting input (+). Extra length separates
# the input-bias resistor from the inverting gain network.
d += elm.Line().right().at(sig1_in.center).length(1.35)
op1 = d.add(elm.Opamp().right().anchor("in2"))
d += elm.Label().at((op1.center[0] - 0.12, op1.center[1] - 0.08)).label("MCP6002A\nStage 1", "center")

# Route inverting input (-) configuration safely downward.
d += elm.Line().left().at(op1.in1).length(0.55)
inv1_node = d.add(elm.Dot().label("Pin 2 (-)", "left"))
d += elm.Resistor().down().at(inv1_node.center)
d += elm.Capacitor().down()
d += elm.Ground()
d += elm.Label().at((inv1_node.center[0] - 0.95, inv1_node.center[1] - 2.05)).label("Rg1 10kΩ\nC_g1 100µF\n(+ up)", "left")

# Clear Stage 1 feedback overhead tracking loop.
d += elm.Line().up().at(op1.in1).length(1.10)
fb1_start = d.add(elm.Dot())
d += elm.Resistor().right().length(2.35).label("Rf1\n100kΩ", "bottom")
fb1_end = d.add(elm.Dot())
d += elm.Line().down().toy(op1.out)
d += elm.Line().left().tox(op1.out)
out1_node = d.add(elm.Dot().label("Pin 1", "bottom"))

# Parallel low-pass cap over feedback path.
d += elm.Line().up().at(fb1_start.center).length(0.62)
d += elm.Capacitor().right().length(2.35).label('C_lpf1\n22nF ("223")', "top")
d += elm.Line().down().toy(fb1_end.center)

# ====================================================
# 5. INTER-STAGE COUPLING
# ====================================================
d += elm.Line().right().at(out1_node.center).length(0.75)
c2 = d.add(elm.Capacitor().right().label('C2\n1µF ("105")', "top"))

# ====================================================
# 6. STAGE 2 AMPLIFIER (MCP6002B)
# ====================================================
d += elm.Line().right().at(c2.end).length(1.00)
sig2_in = d.add(elm.Dot())

# Pull second 1M bias resistor up to the overhead V_VG bus.
d += elm.Resistor().up().at(sig2_in.center).toy(v_vg_node.center)
d += elm.Label().at((sig2_in.center[0] - 0.95, v_vg_node.center[1] - 1.0)).label("R_vbias2\n1MΩ", "left")

# Route straight into non-inverting input (+).
d += elm.Line().right().at(sig2_in.center).length(1.35)
op2 = d.add(elm.Opamp().right().anchor("in2"))
d += elm.Label().at((op2.center[0] - 0.12, op2.center[1] - 0.08)).label("MCP6002B\nStage 2", "center")

# Route inverting input (-) configuration down.
d += elm.Line().left().at(op2.in1).length(0.55)
inv2_node = d.add(elm.Dot().label("Pin 6 (-)", "left"))
d += elm.Resistor().down().at(inv2_node.center)
d += elm.Capacitor().down()
d += elm.Ground()
d += elm.Label().at((inv2_node.center[0] - 0.70, inv2_node.center[1] - 2.05)).label("Rg2 10kΩ\nC_g2 100µF\n(+ up)", "left")

# Clear Stage 2 feedback overhead tracking loop.
d += elm.Line().up().at(op2.in1).length(1.10)
fb2_start = d.add(elm.Dot())
d += elm.Resistor().right().length(2.35).label("Rf2\n100kΩ", "bottom")
fb2_end = d.add(elm.Dot())
d += elm.Line().down().toy(op2.out)
d += elm.Line().left().tox(op2.out)
out2_node = d.add(elm.Dot().label("Pin 7", "bottom"))

# Parallel low-pass cap over feedback path.
d += elm.Line().up().at(fb2_start.center).length(0.62)
d += elm.Capacitor().right().length(2.35).label('C_lpf2\n22nF ("223")', "top")
d += elm.Line().down().toy(fb2_end.center)

# ====================================================
# 7. OUTPUT & ESP32 MCU BLOCK
# ====================================================
d += elm.Line().right().at(out2_node.center).length(0.75)
d += elm.Resistor().right().label("R_out\n1kΩ", "top")

esp32 = d.add(esp32_block().right().anchor("adc"))
d += elm.Dot().at(esp32.adc)
d += elm.Label().at((esp32.adc[0] - 0.55, esp32.adc[1] - 0.55)).label("GPIO 33\n(ADC1_CH5)", "bottom")
d += elm.Line().down().at(esp32.gnd).length(0.42)
d += elm.Ground()

# Keep the package supply note out of the signal/feedback paths.
d += elm.Label().at((op2.out[0] + 1.2, op2.out[1] + 2.4)).label(
    "MCP6002 supply:\nPin 8 → 3.3V_ANA\nPin 4 → GND", "right"
)

# Save cleanly spaced output file.
d.save(OUT)
print(f"Clean schematic with MCP1700 analog LDO generated: {OUT}")
