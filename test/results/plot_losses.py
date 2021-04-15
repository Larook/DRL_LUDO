import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('losses_100_epochs_100_batch_l1_loss.csv')

    df.plot(y='loss')
    plt.show()
