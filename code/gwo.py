import numpy as np
import random
import matplotlib.pyplot as plt

# PARAMETER GWO -----------------------------------------------------

# Jumlah serigala (search agents)
numberOfWolves = 100

# Jumlah iterasi maksimum
maxIteration = 50

# Tampilkan fitness tiap iterasi
displayIteration = True

# Tampilkan plot konvergensi di akhir
plotConvergence = True


# CLASS GWO -----------------------------------------------------

class GreyWolfOptimizer:

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------
    def __init__(self, objectiveFunction, lowerBound, upperBound, dimension):

        # Fungsi objektif
        self.objectiveFunction = objectiveFunction

        # Batas bawah dan atas ruang pencarian
        self.lowerBound = np.array(lowerBound)
        self.upperBound = np.array(upperBound)

        # Jumlah dimensi
        self.dimension = dimension

        # Alpha wolf (solusi terbaik)
        self.alphaPosition = np.zeros(dimension)
        self.alphaScore = float("inf")

        # Beta wolf (solusi terbaik kedua)
        self.betaPosition = np.zeros(dimension)
        self.betaScore = float("inf")

        # Delta wolf (solusi terbaik ketiga)
        self.deltaPosition = np.zeros(dimension)
        self.deltaScore = float("inf")

        # Inisialisasi posisi semua serigala secara acak
        self.wolvesPosition = np.random.uniform(
            self.lowerBound,
            self.upperBound,
            (numberOfWolves, dimension)
        )

        # Menyimpan nilai fitness terbaik tiap iterasi
        self.convergenceCurve = []

    # --------------------------------------------------------
    # Proses Optimasi
    # --------------------------------------------------------
    def optimize(self):

        # Loop utama iterasi
        for iteration in range(maxIteration):

            a = 2 - (2 * iteration / maxIteration)

            # Iterasi Serigala ke-1 hingga akhir
            for wolfIndex in range(numberOfWolves):

                # Pastikan posisi serigala dalam batas pencarian
                self.wolvesPosition[wolfIndex] = np.clip(
                    self.wolvesPosition[wolfIndex],
                    self.lowerBound,
                    self.upperBound
                )

                # Hitung fitness
                fitnessValue = self.objectiveFunction(
                    self.wolvesPosition[wolfIndex]
                )

                # update Alpha
                if fitnessValue < self.alphaScore:
                    self.deltaScore = self.betaScore
                    self.deltaPosition = self.betaPosition.copy()

                    self.betaScore = self.alphaScore
                    self.betaPosition = self.alphaPosition.copy()

                    self.alphaScore = fitnessValue
                    self.alphaPosition = self.wolvesPosition[wolfIndex].copy()

                # update Beta
                elif fitnessValue < self.betaScore:
                    self.deltaScore = self.betaScore
                    self.deltaPosition = self.betaPosition.copy()

                    self.betaScore = fitnessValue
                    self.betaPosition = self.wolvesPosition[wolfIndex].copy()

                # update Delta
                elif fitnessValue < self.deltaScore:
                    self.deltaScore = fitnessValue
                    self.deltaPosition = self.wolvesPosition[wolfIndex].copy()

            # Update Posisi
            for wolfIndex in range(numberOfWolves):
                for dimensionIndex in range(self.dimension):

                    # -----------------------------
                    # Pengaruh Alpha
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(
                        C1 * self.alphaPosition[dimensionIndex]
                        - self.wolvesPosition[wolfIndex][dimensionIndex]
                    )
                    X1 = self.alphaPosition[dimensionIndex] - A1 * D_alpha

                    # -----------------------------
                    # Pengaruh Beta
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(
                        C2 * self.betaPosition[dimensionIndex]
                        - self.wolvesPosition[wolfIndex][dimensionIndex]
                    )
                    X2 = self.betaPosition[dimensionIndex] - A2 * D_beta

                    # -----------------------------
                    # Pengaruh Delta
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(
                        C3 * self.deltaPosition[dimensionIndex]
                        - self.wolvesPosition[wolfIndex][dimensionIndex]
                    )
                    X3 = self.deltaPosition[dimensionIndex] - A3 * D_delta

                    # Update posisi akhir
                    self.wolvesPosition[wolfIndex][dimensionIndex] = (
                        X1 + X2 + X3
                    ) / 3

            # Output tiap iterasi
            self.convergenceCurve.append(self.alphaScore)

            if displayIteration:
                print(
                    f"Iterasi {iteration + 1}/{maxIteration} | "
                    f"Best Fitness: {self.alphaScore:.6f}"
                )

        # Pplot
        if plotConvergence:
            plt.figure(figsize=(8, 5))
            plt.plot(self.convergenceCurve, linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness")
            plt.title("GWO Convergence")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # Alpha
        return {
            "bestScore": self.alphaScore,
            "bestPosition": self.alphaPosition,
            "convergenceCurve": self.convergenceCurve
        }
