import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('losses.csv')

    df.plot(y='loss')
    plt.show()
