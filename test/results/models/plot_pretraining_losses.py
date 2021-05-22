import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('pretrained_losses_epochs200_lr0.1__21_15_37.csv', index_col=False)
    # df = df.reset_index(drop=True, inplace=True)
    # df = df.drop(df.columns[0], axis=1, inplace=True)
    # plt.title('Pretraining of the MLP, epochs = 200, batch_size = 50, learning_rate = 0.1')
    # plt.xlabel('observations')
    # plt.ylabel('MSE')
    df.columns = [0, 'pretraining loss']

    df.plot(y=df.columns[1], figsize=(6, 2), xlim=[0, len(df)], lw=0.03)


    # plt.savefig("20_04.jpg")
    plt.show()
