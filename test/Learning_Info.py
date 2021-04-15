import pandas as pd


class Learning_Info():
    def __init__(self):
        self.whole_list = []

    def append(self, epoch_no, epochs_won, action_no, ai_player_i, begin_state, action, new_state, reward, loss):
        self.whole_list.append({'epoch_no': epoch_no, 'epochs_won': epochs_won, 'action_no': action_no,
                                'ai_player_i': ai_player_i, 'begin_state': begin_state, 'action': action,
                                'new_state': new_state, 'reward': reward, 'loss': loss})


    def save_to_csv(self, path):
        data_df = pd.DataFrame(self.whole_list)
        data_df.to_csv(path)

