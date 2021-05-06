last_turn_state_new = []
rewards_detected = {'piece_release': 0, 'defend_vulnerable': 0, 'knock_opponent': 0,
                    'move_closest_goal': 0, 'move_closest_safe': 0, 'forming_blockade': 0,
                    'getting_piece_knocked_next_turn': 0, 'ai_agent_won': 0, 'ai_agent_lost': 0}
epsilon_now = 0
