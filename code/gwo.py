import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

# Jumlah serigala (search agents)
numberOfWolves = 100

# Jumlah iterasi maksimum
maxIteration = 50

# Tampilkan fitness tiap iterasi
displayIteration = True

# Tampilkan plot konvergensi di akhir
plotConvergence = True

# Batas stagnasi
stagnationPatience = 5

# proporsi wolf di reset
resetRatio = 0.5

# toleransi perubahan
epsilon = 1e-6  

# ============================================================
# GREY WOLF OPTIMIZER CLASS
# ============================================================

class GreyWolfOptimizer:

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------
    def __init__(self, objectiveFunction, lowerBound, upperBound, dimension):

        # Fungsi objektif (fitness function)
        self.objectiveFunction = objectiveFunction

        # Batas bawah dan atas ruang pencarian
        self.lowerBound = np.array(lowerBound)
        self.upperBound = np.array(upperBound)

        # Jumlah dimensi (jumlah parameter yang dioptimasi)
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

        # Variabel pengecekan stagnasi atau tidak
        self.noImprovementCounter = 0
        self.previousBestScore = float("inf")

    # Menghitung Standart Deviasi populasi
    def calculateDiversity(self):
        return np.mean(np.std(self.wolvesPosition, axis=0))

    # --------------------------------------------------------
    # Proses Optimasi
    # --------------------------------------------------------
    def optimize(self):

        # Loop utama iterasi
        for iteration in range(maxIteration):

            # Parameter kontrol eksplorasi–eksploitasi
            diversity = self.calculateDiversity()
            a = 2 * np.exp(-iteration / maxIteration) * (1+ diversity)

            # =================================================
            # Evaluasi Fitness & Update Alpha, Beta, Delta
            # =================================================
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

            # =================================================
            # Update Posisi Semua Serigala
            # =================================================
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

            # Stagnasi hanlder
            if abs(self.alphaScore - self.previousBestScore) < epsilon:
                self.noImprovementCounter += 1
            else:
                self.noImprovementCounter = 0
            self.previousBestScore = self.alphaScore

            # kondisi Stagnasi
            if self.noImprovementCounter >= stagnationPatience:
                resetCount = int(resetRatio * numberOfWolves)
                resetIndex = np.random.choice(
                    numberOfWolves, resetCount, replace=False
                )

                self.wolvesPosition[resetIndex] = np.random.uniform(
                    self.lowerBound,
                    self.upperBound,
                    (resetCount, self.dimension)
                )
                self.noImprovementCounter = 0
            # =================================================
            # Simpan & Tampilkan Fitness Iterasi
            # =================================================
            self.convergenceCurve.append(self.alphaScore)

            if displayIteration:
                print(
                    f"Iterasi {iteration + 1}/{maxIteration} | "
                    f"Best Fitness: {self.alphaScore:.6f}"
                )

        # =====================================================
        # Plot Konvergensi
        # =====================================================
        if plotConvergence:
            plt.figure(figsize=(8, 5))
            plt.plot(self.convergenceCurve, linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness")
            plt.title("GWO Convergence")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # =====================================================
        # Return hasil optimasi
        # =====================================================
        return {
            "bestScore": self.alphaScore,
            "bestPosition": self.alphaPosition,
            "convergenceCurve": self.convergenceCurve
        }
