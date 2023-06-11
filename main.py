# Project: PETE332 - Computer Assignment
# Authors: Eren Tuna Açıkbaş, Tuğberk Ataş
# Date created: 2020-11-20
# Date last modified: 2020-11-20
# Python Version: 3.10
import numpy as np
import matplotlib.pyplot as plt
import time

# Configurations
topSegments = 80
bottomSegments = 20

# Eren Tuna Açıkbaş's Student Number is selected
# 2441657
a = 2
b = 4
c = 4
d = 1
e = 6
f = 5
g = 7

# Given variables and constants
# Reservoir Depth
ResDepth = 3000 + (g * 100)  # ft
# Reservoir Pressure
Pres = ResDepth * 0.38  # psi
# Productivity Index
J = 1.20 + (f * 0.05)  # std/day/psi
# Gravity
Gravity = 20 + (e * 2)  # API
# Water Cut
WC = (10 + (f * 5)) * 0.01  # %
# Pump Height
PumpH = 300 + (g * 20)  # ft
# Flow Rate
Q = 200 + (g * 45)  # stb/day
# Gas Liquid Ratio
GLR = 100 + (d * 10)  # scf/stb

# Additional Data is provided as
# Water Formation Volume Factor
Bw = 1.03  # rb/stb
# Water Specific Gravity
SGWater = 1.05
# Gas Specific Gravity
GasSG = 0.68
# Reservoir Temperature
ResTemp = 180  # F
# Temperature Gradient
TempGrad = 2 / 100  # F/100 ft
# Oil Specific Gravity
OilSG = 141.5 / (Gravity + 131.5)

# Calculations from provided reservoir data
# Bubble Point Pressure can be obtained from Standing's Correlation
Pb = 18.2 * (((GLR / GasSG) ** 0.83) * 10 ** (0.00091 * ResTemp - 0.0125 * Gravity) - 1.4)

# Solution Gas Oil Ratio
GOR = GLR / (1 - WC)
WOR = WC / (1 - WC)
AirDensity = 0.0765  # lb/ft3
Qoil = Q * (1 - WC)  # stb/day


def calculatePbhf(Pres: float, Pb: float):
    """
    This function decides whether the flow in the reservoir is
    partial two phase,single phase, or two phase and calculate the
    pressure at the bottom hole accordingly.
    :param Pres: Pressure at the reservoir
    :param Pb: Bubble Point Pressure
    :rtype: float
    """

    # Partial Two Phase Flow maximum flow rate
    Qmax = J * Pres / 1.8
    if Pres > Pb:
        Qb = J * (Pres - Pb)
        if Qb <= Q:  # Partial Two Phase Flow
            Pbhf = 0
            qcheck = Qb + Qmax * (1 - 0.2 * (Pbhf / Pb) - 0.8 * (Pbhf / Pb) ** 2)
            while qcheck - Q >= 0.01:
                qcheck = Qb + Qmax * (1 - 0.2 * (Pbhf / Pb) - 0.8 * (Pbhf / Pb) ** 2)
                Pbhf += 0.01
        else:
            Pbhf: float = Pres - Q / J
        return Pbhf
    elif Pres <= Pb:  # Single Phase Flow
        Pbhf: float = 0.125 * Pres * ((81 - 80 * (Q / Qmax)) ** 0.5 - 1)
        return Pbhf
    else:
        raise Exception("Flow Behaviour is not defined")


def calculateZ(P: float, T: float, GasSG: float):
    """
    This function calculates the Z factor of the gas.
    :param P: Pressure
    :param T: Temperature
    :param GasSG: Gas Specific Gravity
    :rtype: float
    """
    # Pseudo Critical Temperature
    Tpc = 168 + 325 * GasSG - 12.5 * GasSG ** 2
    # Pseudo Critical Pressure
    Ppc = 677 + 15 * GasSG - 37.5 * GasSG ** 2
    # Reduced Temperature
    Tr = T / Tpc
    # Reduced Pressure
    Pr = P / Ppc
    # Omega
    Omega = 0.27 - 0.0009 * Gravity
    # Universal Gas Constant
    R = 10.732

    m = 0.37464 + 1.54226 * Omega - 0.26992 * Omega ** 2

    # Peng-Robinson equation
    a = (0.45724 * R ** 2 * Tpc ** 2 * (1 + m * (1 - Tr ** 0.5)) ** 2) / Ppc
    b = 0.07780 * R * Tpc / Ppc

    A = a * P / (R * T) ** 2
    B = b * P / (R * T)

    # Cubic Equation Coefficients
    Coefficients = [1, - (1 - B), A - 3 * B ** 2 - 2 * B, - (A * B - B ** 2 - B ** 3)]
    # Gas Z Factor (Real Roots of Cubic Equation)
    Z = max(np.real(np.roots(Coefficients)))
    return Z


def calculateRs(P: float, T: float, Gravity: float, SG: float):
    """
    This function calculates the solution gas oil ratio.
    :param Gravity:
    :param P: Pressure
    :param T: Temperature (Fahrenheit)
    :param SG: Specific Gravity
    :rtype: float
    """
    Pb = 18.2 * ((GOR / GasSG) ** 0.83 * 10 ** (0.00091 * T - 0.0125 * Gravity) - 1.4)
    if P <= Pb:
        Rs = ((P / 18.2 + 1.4) / (10 ** (0.00091 * T - 0.0125 * Gravity))) ** (1 / 0.83) * SG
    elif P > Pb:
        Rs = GOR
    else:
        raise Exception("P:{}, Pb:{} This behaviour is not expected".format(P, Pb))
    return Rs


def calculateBo(P: float, Rs: float, SG: float, OilSG: float, T: float):
    """
    This function calculates the oil formation volume factor.
    :param P: Pressure
    :param Rs: Oil Formation Volume Factor
    :param SG: Specific Gravity
    :param OilSG: Oil Specific Gravity
    :param T: Temperature (Fahrenheit)
    :rtype: float
    """
    Pb = 18.2 * ((GOR / GasSG) ** 0.83 * 10 ** (0.00091 * T - 0.0125 * Gravity) - 1.4)
    if P <= Pb:
        Cb = Rs * (SG / OilSG) ** 0.5 + 1.25 * T
        Bo = 0.9759 + 0.00012 * Cb * 1.2
    else:
        Cb = GOR * (SG / OilSG) ** 0.5 + 1.25 * T
        Bo = 0.9759 + 0.00012 * Cb * 1.2
    return Bo


# Poetmann-Carpentar Method
# 𝑀=350.17(𝛾_𝑜+𝑊𝑂𝑅𝛾_𝑤)+𝐺𝑂𝑅𝛾_𝑔 𝜌_𝑎𝑖𝑟
M = 350.17 * (OilSG + WOR * SGWater) + GOR * GasSG * AirDensity


def calculateProperties(P0: float, Density0: float, Rs0: float, T0: float, Depth: float, numOfSegments: int):
    """
    This function calculates the pressure at the bottom hole.
    :param P0: Initial pressure assumption
    :param T0: InitialTemperature (Fahrenheit)
    :param Rs0: Initial solution gas oil ratio
    :param Density0: Initial density
    :param Depth: Given Depth
    :param numOfSegments: Number of segments
    """
    # Opening some lists to store the data calculated in each segment
    arrayOfPressures = [P0]  # Starts from P0 to store the first pressure
    arrayOfDensities = [Density0]  # starts from density_0 to store the first data
    arrayOfRs = [Rs0]  # Starts from Rs_0 to store every Rs_0
    arrayOfTemperature = [T0 + 460]  # starts from T0 to store every temperature in Rankine

    lengthOfSegments = Depth / numOfSegments

    Pi = P0 - 50
    Ti = (T0 - lengthOfSegments * TempGrad) + 460  # Rankine

    givenData = [["P0", "Density0", "Rs0", "T0", "Depth", "Number of Segments"],
                 [P0, Density0, Rs0, T0, Depth, numOfSegments]]
    print("\n\033[93mWarning:\033[0m The below provided data is calculated using the Poetmann-Carpentar Method\n")
    printTable(givenData)
    count = 0
    Densityi: float = 0
    while count < numOfSegments:
        tolerance = 0.5

        Rsi = calculateRs(Pi, Ti - 460, Gravity, GasSG)
        Zi = calculateZ(Pi, Ti, GasSG)
        Boi = calculateBo(Pi, Rsi, GasSG, OilSG, Ti - 460)
        Vmi = 5.615 * (Boi + (WOR * Bw)) + (GOR - Rsi) * (14.7 / Pi) * (Ti / 520) * (Zi / 1)
        Densityi = M / Vmi
        DensityAvg = (Density0 + Densityi) / 2
        Drv = (1.4737 * 10 ** -5 * M * Qoil) / (4.67 / 12)
        f = 10 ** (1.444 - 2.5 * np.log10(Drv))
        k = (f * (Qoil ** 2) * (M ** 2)) / (7.4137 * 10 ** 10 * DensityAvg * (4.67 / 12) ** 5)
        DeltaP = (DensityAvg + (k / DensityAvg)) * (lengthOfSegments / 144)
        Pcheck = Pi + DeltaP
        Pnew = P0 - DeltaP

        if np.abs(Pcheck - P0) <= tolerance:
            count += 1
            arrayOfPressures.append(Pnew)
            arrayOfDensities.append(DensityAvg)
            arrayOfRs.append(Rsi)
            arrayOfTemperature.append(Ti)
            P0 = Pi
            Pi = P0 - 50
            T0 = Ti
            T0Fahrenheit = T0 - 460
            Ti = (T0Fahrenheit - TempGrad * lengthOfSegments) + 460

        elif np.abs(Pcheck - P0) >= tolerance:
            Pi += 0.1

    results = [["Pressure", "Density", "Rs", "Temperature"]]
    for i in range(len(arrayOfPressures)):
        list = [arrayOfPressures[i], arrayOfDensities[i], arrayOfRs[i], arrayOfTemperature[i]]
        results.append(list)
    print("\n\033[92mSuccess:\033[0m Calculations has been finished. Results can be observed below.\n")
    printTable(results)
    return arrayOfPressures, arrayOfDensities, arrayOfRs, arrayOfTemperature, Densityi


def calculateDepths(numOfBottomSegments: int, numOfTopSegments: int):
    """
    This function calculates the depths of the reservoir.
    :type numOfBottomSegments: int
    :type numOfTopSegments: int
    :param numOfBottomSegments: Number of segments below the pump
    :param numOfTopSegments: Number of segments above the pump
    :return: [float]
    """
    DepthsBelowPump = [ResDepth]
    DepthsAbovePump = []
    lastElement: float = 0
    lengthOfSegmentsBelowPump = PumpH / numOfBottomSegments
    lengthOfSegmentsAbovePump = (ResDepth - PumpH) / numOfTopSegments
    for numOfSegment in range(numOfBottomSegments):
        DepthsBelowPump.append(ResDepth - (numOfSegment + 1) * lengthOfSegmentsBelowPump)
        lastElement = DepthsBelowPump[-1]
    for numOfSegment in range(numOfTopSegments + 1):
        DepthsAbovePump.append(lastElement - numOfSegment * lengthOfSegmentsAbovePump)
    return DepthsBelowPump, DepthsAbovePump


def drawMultiplePlot(x: [[float]], y: [float], title: [str], xlabel: [str], ylabel: [str], filename: str) -> object:
    """
    This function draws the given x and y values.
    :param x: x values
    :param y: y values
    :param title: title of the plot
    :param xlabel: x label
    :param ylabel: y label
    :param filename: filename of the plot
    :return:
    """
    q1 = input("Do you want to display all graph in one figure? (y/n): ")

    num_graphs = len(x)
    figures = []

    if q1 == "y":
        fig, axs = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 4))
        for i in range(num_graphs):
            axs[i].plot(x[i], y)
            axs[i].set_title(title[i], fontsize=14, fontweight="bold")
            axs[i].set_xlabel(xlabel[i])
            axs[i].set_ylabel(ylabel[i])
            axs[i].grid(True)
            axs[i].invert_yaxis()
        figures.append(fig)
        fig.tight_layout()
        plt.show()  # Display the figure inline using plt.show()
        return figures
    if q1 == "n":
        for i in range(num_graphs):
            fig, ax = plt.subplots()
            ax.plot(x[i], y)
            ax.set_title(title[i], fontsize=14, fontweight="bold")
            ax.set_xlabel(xlabel[i])
            ax.set_ylabel(ylabel[i])
            ax.grid(True)
            ax.invert_yaxis()
            fig.tight_layout()
            figures.append(fig)
            plt.show()  # Display the figure inline using plt.show()
        return figures
    else:
        print("\033[91mError:\033[0m Invalid input")
        exit(1)


def printTable(data):
    # Determine the maximum width of each column
    column_widths = [max(len(str(item)) for item in column) for column in zip(*data)]

    # Print the table headers
    for header, width in zip(data[0], column_widths):
        print(f"{header:{width}}", end=" | ")
    print()

    # Print the table separator
    separator = "-+-".join("-" * width for width in column_widths)
    print(separator)

    # Print the table rows
    for row in data[1:]:
        for item, width in zip(row, column_widths):
            print(f"{item:{width}}", end=" | ")
        print()


def main():
    # Properties Below the Pump
    Pbhf = calculatePbhf(Pres, Pb)
    P0 = Pbhf
    T0Fahrenheit = ResTemp
    T0 = T0Fahrenheit + 460  # Rankin
    Rs0 = calculateRs(P0, T0Fahrenheit, Gravity, GasSG)
    Z0 = calculateZ(P0, T0, GasSG)
    Bo0 = calculateBo(P0, Rs0, GasSG, OilSG, T0Fahrenheit)
    Vm = 5.615 * (Bo0 + WOR * Bw) + (GOR - Rs0) * (14.7 / P0) * (T0 / 520) * (Z0 / 1)  # Reservoir Volume Factor
    Density0 = M / Vm  # lb/ft3

    print("\n\033[94mInfo:\033[0m Calculating properties below the pump")
    time.sleep(5)

    PressuresBelowPump, DensitiesBelowPump, RsBelowPump, TemperaturesBelowPump, Densityi = calculateProperties(
        P0, Density0, Rs0, T0Fahrenheit, PumpH, bottomSegments)

    # Pump Properties
    HeadPerStage = -7.90 * (10 ** -8) * (Q ** 3) + 2.47 * (10 ** -5) * (Q ** 2) - 1.16 * (10 ** -2) * Q + 3.24 * (
            10 ** 1)
    NumberOfStages = 100
    TotalHead = HeadPerStage * NumberOfStages
    PumpPressure = (TotalHead * Densityi) / 144

    # Properties above the pump
    P1 = PressuresBelowPump[-1] + PumpPressure
    T1 = TemperaturesBelowPump[-1]
    T1Fahrenheit = T1 - 460
    Rs1 = calculateRs(P1, T1Fahrenheit, Gravity, GasSG)
    Z1 = calculateZ(P1, T1, GasSG)
    Bo1 = calculateBo(P1, Rs1, GasSG, OilSG, T1Fahrenheit)
    Vm1 = 5.615 * (Bo1 + WOR * Bw) + (GOR - Rs1) * (14.7 / P1) * (T1 / 520) * (Z1 / 1)
    Density1 = M / Vm1
    DepthAbovePump = ResDepth - PumpH

    print(
        "\n\033[94mInfo:\033[0m Calculating properties above the pump by taking the pump pressure effect into account")
    time.sleep(5)
    PressuresAbovePump, DensitiesAbovePump, RsAbovePump, TemperaturesAbovePump, Densityi = calculateProperties(
        P1, Density1, Rs1, T1Fahrenheit, DepthAbovePump, topSegments)

    # Combining the results
    properties = [PressuresBelowPump + PressuresAbovePump, DensitiesBelowPump + DensitiesAbovePump,
                  RsBelowPump + RsAbovePump]
    DepthsBelowPump, DepthsAbovePump = calculateDepths(bottomSegments, topSegments)

    # Plotting the results
    depths = DepthsBelowPump + DepthsAbovePump
    titles = ["Pressure vs Depth", "Density vs Depth", "Rs vs Depth"]
    xlabels = ["Pressure (psi)", "Density (lb/ft3)", "Rs (scf/STB)"]
    ylabels = ["Depth (ft)", "Depth (ft)", "Depth (ft)"]

    # Plotting the results in one figure
    drawMultiplePlot(properties, depths, titles, xlabels, ylabels, "results")


if __name__ == '__main__':
    main()
