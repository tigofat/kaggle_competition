import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from typing import List

@dataclass(init=True)
class Plotter:

    colors: List

    def relationship(self, df, x, y, color=None, alpha=1):
        color_ = np.random.choice(self.colors) if not color else color
        plt.plot(df[x], df[y], 'o', label=f'{x} - {y}', color=color_, alpha=alpha)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.show()
