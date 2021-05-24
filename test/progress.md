## Saving the state
somehow save the information about the state in a string maybe?
    54-xxxExxPxxxxxx|12-xxxxxxPxxxxxx|0|0

<> Notes on the way of saving state:

sometimes one enemy pawn is seen by our 2 pawns:
found enemy[2] pawn = 1 near player_0[0] = 4
found enemy[2] pawn = 1 near player_0[1] = 1
state now =  4-1,|1-1,|0|0|

I cant see them dying - ok field 1 is safe state
but why
found enemy[3] pawn = 6 near player_0[0] = 4
found enemy[2] pawn = 18 near player_0[1] = 18
state now =  4-6,|18-18,|0|0|

Do other players start from the same global coordinates of saved tiles? Or is it player specific?
I think it is player-specific because max index of each pawn is 59
we have to somehow unify this, so the states of other players are mapped to the player0 - there is 13 tiles difference between each base

enemy positions cant exceed 52 -> if >52 then next one is 1 WORKS
    
    round = 6 	state = 0|0|30-|0|
    round = 7 	state = 0|0|30-|0|
    found enemy[1] pawn = 41 near player_0[2] = 38
    found enemy[2] pawn = 40 near player_0[2] = 38
    round = 7 	state = 0|0|38-41,40,|0|
    found enemy[1] pawn = 42 near player_0[2] = 38
    found enemy[2] pawn = 40 near player_0[2] = 38
    round = 7 	state = 0|0|38-42,40,|0|
    found enemy[1] pawn = 42 near player_0[2] = 38
    found enemy[2] pawn = 41 near player_0[2] = 38
    round = 7 	state = 0|0|38-42,41,|0|
    found enemy[1] pawn = 42 near player_0[2] = 38
    found enemy[2] pawn = 41 near player_0[2] = 38
    round = 8 	state = 0|0|38-42,41,|0|                <- here enemy at 41
    found enemy[1] pawn = 42 near player_0[2] = 41
    round = 8 	state = 0|0|41-42,|0|
    found enemy[1] pawn = 42 near player_0[2] = 41
    round = 8 	state = 0|0|41-42,|0|                   <- here enemy dead
    round = 8 	state = 0|0|41-|0|
    round = 8 	state = 0|0|41-|0|
    round = 9 	state = 0|0|41-|0|

####In general it works but cant print when somebody was struck out!

## Choosing actions based on state

first try playing with choosing the furthest player to move
then maybe add choosing an action that can eliminate enemy

But for all of that we will need to decode the state - from string again to arrays - maybe it can be made on the go

but definiately function will be needed

#### OK the furthest one is working!

## Try 4 x 59 + 4 state representation from papers
The first 59 variables represent the current player’s board positions.
Each variable is a number between 0  and 1 (inclusive) that represents the percentage of the player’s pieces that are
in that position.
For example, if the current player has 3 of the 4 pieces in position 5, the variable corresponding to this position 
has a value of 0.75.

• The next 59 variables represent the next player’s board
positions, and so on to account for all 4 players.

• The first variable for each of the 4 players corresponds
to the HOME position. The initial value of this variable is
1 for all players. The last variable for each player corresponds to the GOAL position.
A player wins when this variable becomes 1.

OK this state is working
Try to do the "furthest moves" strategy - WORKS

Now try to do actual Q-learning. And maybe clean up the code.

## saving actions
"Every action is represented as a tuple (x0/58, xf/58) where x0 is initial position and xf is the final position.
Dividing by 58 is necessary to obtain a value between 0 and 1."

Is it necessary though?

    action = (x0, xf). 

We should see all the possible actions and choose the one with the highest value.
After selecting an action, the id of pawn should be returned.

How to return the id of pawn after choosing an action saved as that?

    state_before = [player0, player1, player2, player3]
    state_after_a0, value_0 = get_Q(state_before, action=pawn_id0)  
    state_after_a1, value_1 = get_Q(state_before, action=pawn_id1)  
    state_after_a2, value_2 = get_Q(state_before, action=pawn_id2)  
    state_after_a3, value_3 = get_Q(state_before, action=pawn_id3)

then select the best action based on values - why then save actions as tuple?

## saving reward
get_reward(state)

    • 1.0 for winning a game.
    • 0.25 for releasing a piece from JAIL.
    • 0.2 for defending a vulnerable piece.
    • 0.15 for knocking an opponent’s piece.
    • 0.1 for moving the piece that is closest to home.
    • 0.05 for forming a blockade.
    • -0.25 for getting a piece knocked in the next turn.
    • -1.0 for losing a game.
    Rewards can be accumulated

## state, action, reward in transition (input to ANN)

Just to be able to save it in format applicable by the neural network. We need transition

    Transition - named tuple representing a single transition on environment. Maps (state, action) pairs to their
     (next_state, reward) result, with the staet being the screen difference image
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

## ANN
select_action(state):
    uses trained model of NN
    
    input: 242 inputs
    240 for state s, 2 for actions
    
    layer of 20 hidden neurons with sigmoid activation

    1 output Q(s,a) - value of what our return would be, if we were to take an action in a given state

Where to add the rewards?
    

## tutorial 
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
The key idea of experience replay5 is that we store these transitions in our memory and during each learning step, sample a random batch and perform a gradient descend on it. This way we solve both issues.
    
The Brain class encapsulates the neural network. Our problem is simple enough so we will use only one hidden layer of 64 neurons, with ReLU activation function. The final layer will consist of only two neurons, one for each available action. Their activation function will be linear. Remember that we are trying to approximate the Q function, which in essence can be of any real value.

Initialization with random agent
~"We can assume, that a random policy would do the same job in choosing actions, but at much higher rate. Therefore we can simply fill the memory of the agent before the start of the learning itself with a random policy."
    
Ok it slightly is ok - at least it runs. But need to inspect more this target and saving the final states


In tutorial he uses states and states_
but I am already saving the differences between the states as an input to the network

    input: 242 inputs
        240 for state s, 2 for actions

What can we see from the analysis of losses?
It seems unstable, but need to check for many more epochs -> more epochs still not winning

    TODO:
    - clean the code
    - plot the win rate ~ epochs
    - maybe play also with history and try to see if its improving - bad idea

    what to save to csv?
    every decision:
    - epoch_no, epochs_won, action_no, begin_state, action, new_state, reward, loss
    - action = -1 means there were no movable objects
    - add time DONE
    - accumulated reward??? little sense yet

    - change the epsilon - it should be totally random for the first n=5 plays -> it can be changed only with epsilon
    - epsilon threshold should be still around 0.7-0.8 after 10 round -> 3380 action_no

    - does it run, even when an enemy player won? Should it learn those states? Maybe not
    - doesnt even get reward -1, nor 1 DONE
    - add reward also for the moving the furthest pawn ? seems sketchy but at least could outperform random players? DONE
- what about recording few plays by myself and then just pretrain the network? - so it actually has some winning data
- probably still then there will be a need for tweaking the parameters
    - add average reward of epoch to plot DONE
  
- why the first rounds of a game are super fast? - doesn't it have access to the full batch of the net? - Should have
- read the implementations again!

should add target network?
should train after the reached end state with reward +-1?

1 - plot the random walk rewards etc
2 - look at the code from internet! check similarities -> how is their immediate reward calculated?
3 - try to teach just to choose the furthest piece

    in pytorch there was no training dqn if batch isn't the full size -> fixed

Add saving the model of ann!

Case of the batch_size: https://www.samyzaf.com/ML/rl/qmaze.html
"The training of N will be done after each game move by injecting a random selection of the most recent training samples to N. Assuming that our game skill will get better in time, we will use only a small number of the most recent training samples. We will forget old samples (which are probably bad) and will delete them from memory."

    select action:
        use Q from NN
    get reward and next state (state, action) - it is immediate reward!
    store experience
    train NN - this guy uses only one network - not really terminal difference, and the pipeline looks ok

when epoch==40 -> avg time epoch == 224sec = almost 4 min
avg time left to have 10k -> approx 60 hours!!!
to see if we can do better need to make it faster

see what is executing every time - whats the longest function
## timing of functions
1. optimize_model() = 7.0066 - could try to move to collabs
2. get_state_after_action() = 0.0035 but executed a lot! - how can we strip it? Need to constantly check if random_walk still sees some wins
3. action_selection()
4. get_game_state()
5. get_reward()

    on laptop it seems to run ludo faster without the GPU -> move to collab and see there.
    After comparing timing of the first fully DQN-predicted game the fastest approach would be solving the problem locally without gpu

After that try TD approach with 2 networks, but still 1 is not enough...
Then try just to teach the approach to move the furthest pawn - save the model and then try to replay the game using model

    print current dice to csv
    if won record the video

    can the game be won with the random moves? Yes it should be 25% chance to win. Fixed and working!
    add winrate to csv

    using the DQN shows behaviour of selecting the furthest pawn - that's okish but moving new one from home should be more important

Add continuing to learn from the selected model!


## using 2 networks - q_net and target_net  ->  https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435
We executed our optimization step to bring the prediction close to ground truth but at the same time we are changing the weights of the network which gave us the ground truth. 
This causes instability in training.

The solution is to have another network called Target Network which is an exact copy of the Main Network. 
This target network is used to generate target values or ground truth. 
The weights of this network are held fixed for a fixed number of training steps after which these are updated with the weight of Main Network. 
In this way, the distribution of our target return is also held fixed for some fixed iterations which increase training stability.

    def train(self, batch_size ):
            s, a, rn, sn = self.sample_from_experience( sample_size = batch_size)
            if(self.network_sync_counter == self.network_sync_freq):
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.network_sync_counter = 0
            
            # predict expected return of current state using main network
            qp = self.q_net(s.cuda())
            pred_return, _ = torch.max(qp, axis=1)
            
            # get target return using target network
            q_next = self.get_q_next(sn.cuda())
            target_return = rn.cuda() + self.gamma * q_next
            
            loss = self.loss_fn(pred_return, target_return)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            self.network_sync_counter += 1       
            return loss.item()

From what is visible from 50 epochs the model trained to select the furthest pawn - because of frequent reward.
What I can do more:
- record playing the game by myself - expert data
    - add rewards for other behaviours like blockade - DONE
    - count the rewards got and plot the number of kills and blockades and stuff like this (print also won and lost games in rewards plotting)
- maybe reward for reaching the star (teleport)

- when stuff is running - check the correctness of networks once again - check the no_grad and how it should be done - check also the learning
- when that is done - compare results with different batch_size

after 134 epochs:
    still chooses the furthest pawn

    need to change rewards - when choosing furthest which is in the safe zone give less reward - done
    or give negative reward when breaking the blockade and more points for getting out a fresh piece from home ---
    added reward for losing pieces

maybe change learning rate / batch size
add more readable plots - seems like the counter for the rewards taken is too high - only on live plot
seems like knocking out pieces is wrong! - fixed

maybe value forming a blockade more than selecting furhtest piece.

after 222 epochs winrate is still around 28% with random players - which seem random. Things to change would be probably the batch size or again reading more about DQN. 
    
    change x axis to number of epochs
    plot the decision making epsilon and maybe add learning rate?
    
read more on the stuff!

TODO:
    add rewards for going to GLOBES and STARS and then record the state of the games and feed the neural net

Ok it turned out that data acquired is not enough, will need to run again the human recordings
Before that double check if everything needed is saved
especially:

get_max_reward_from_state(game, s_, available_actions)
    get_state_after_action(game, action)

    add plot counter info about new rewards
    tune the epsilon to settle down after 50 games

    
add -0.1 reward for moving the furthest piece when in safe zone near goal - encourage to go out of home with new pieces!
How to encourage it instead of adding human-played games?

test the model on bigger number of epochs
    epoch = 99 | round = 112 <avg_time_left = 3.56 avg_time_epoch = 1.78 | avg_time_turn = 0.01> | won_counter = 28 | steps_done = 36540 

maybe also check training with semi-smart agents   
    epoch = 199 | round = 93 <avg_time_left = 3.31 avg_time_epoch = 1.65 | avg_time_turn = 0.01> | won_counter = 49 | steps_done = 72400 



testing the 1200 batch size model:
    epoch = 199 | round = 78 <avg_time_left = 3.31 avg_time_epoch = 1.66 | avg_time_turn = 0.01> | won_counter = 51 | steps_done = 73090 |


    TODO: 

Still hasnt learnt anything - need to try to pretrain the network!

sampling list of dict!  -      batch = data.sample(batch_size)
get_max_reward_from_state

save more to csv:
get_max_reward_from_state(pieces_player_begin, dice, state, possible_actions)
    pieces_player_begin
    possible_actions


    rewards - when the furthest away is in the safe zone - dont add +0.1 reward!!!

check if shuffling of the memory works!
record more games
Piece release might be wrong!!!

fix rewards problem!!! Loosing pieces - doesnt work 100% but cant find where is the error


evaluating after 2 human games -> still 22% winrate against random player which is really bad
But this is the raw 2 games played so far - play another game

getting_human_data:
    make it faster by accessing the recent np.history in RAM - done. Need testing 

evaluating only HUMAN after 3 games -> still 21% winrate in 100 games

now try to train on the pretrained model!
- after 3 games the training winrate was again 21%
Maybe because the first game recorded was lost?
  Try to evaluate the 296 trained epochs with 3 games - performance 19%
  
and add more preplayed games



fixed the ann input representation
pretrained again the human expert data
model with wrong shuffling het the winrate around 10/57

now the correct pretraining - actually using the whole dataset - the winrate now is 14/57 which is much better!
actually it is 20% winrate with pretrained net
epoch = 100 | round = 89 <avg_time_left = 113.80 avg_time_epoch = 1.13 | avg_time_turn = 0.00> | won_counter = 20 | steps_done = 38218 | action = 1 | avg_reward = 0.078642, loss_avg = 0.000000 | epsilon = 0.051282
Hopefully trained MLP now will give better results

trained mlp on 282 epochs - need to fix plots

Evaluate model playing with random player
epoch = 199 | round = 73 <avg_time_left = 2.18 avg_time_epoch = 1.09 | avg_time_turn = 0.00> | won_counter = 36 | steps_done = 73636 | action = 0 | avg_reward = 0.078152, loss_avg = 0.000000 | epsilon = 0.050058
winrate = 18%

then evalueate model playing with enemy players controlled by pretrained neural network


    fixed the networks - now loads the correct state

Pretrained again the initial human data network
    # human network pretrain
    epochs_pretrain = 200
    pretrain_batch_size = 50
    learning_rate_pretrain = 0.1  # big one

    saved the plots from training
    
and evaluated the winrate after 200 games -> it won 45 thus the winrate was 22.5%

now after fixing the previous bugs the model can be trained again with correctly pretrained network

it had 16/71 = 0.225352 winrate but I realized that avg loss isnt plotted

the training model was 31/170= 18.2% which was still decreasing as unlearning

changed platou of epsilon to be after 200 games
changed network_sync_freq from 100 to 500 now
changed learning rate from 5e-3 to 1e-2


when training - 52/249 = 0.208 winrate
evaluating: 49/199 -> 0.2412
another evaluating 73/299
which gives 122/498 -> 0.2449 but when 122/500 then winrate=0.244


playing vs Nathan's Q learning -> his agent has 54% against 3 random players
so all of the other random players have 15.3%
for now winrate is 22/169=13%

tbh showing the correct winrate doesnt work...

epoch = 315 | round = 63 <avg_time_left = 375.42 avg_time_epoch = 2.02 | avg_time_turn = 0.16> | won_counter = 28 | steps_done = 114818 | action = 2 | avg_reward = 0.074843, loss_avg = 0.000000 | epsilon = 0.051176
players_win_counter =  {0: 172, 1: 61, 2: 231, 3: 63}     Nathan_won_counter 231

28/315 = 8.88 %   - DRL

epoch = 393 | round = 55 <avg_time_left = 129.98 avg_time_epoch = 1.20 | avg_time_turn = 0.08> | won_counter = 55 | steps_done = 140105 | action = 2 | avg_reward = 0.085973, loss_avg = 0.000000 | epsilon = 0.050258
players_win_counter =  {0: 68, 1: 49, 2: 231, 3: 44}     Nathan_won_counter 231

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

epoch = 344 | round = 90 <avg_time_left = 294.68 avg_time_epoch = 1.88 | avg_time_turn = 0.12> | won_counter = 52 | steps_done = 124023 | action = 1 | avg_reward = 0.084659, loss_avg = 0.000000 | epsilon = 0.050677
players_win_counter =  {0: 65, 1: 56, 2: 226, 3: 58}     Nathan_won_counter 226

DRL - 65/405 -0.16049382716049382
random1 - 56/405 - 0.1382716049382716
Q-learning - 226/405 - 0.5580246913580247
random2 = 58/405 - 0.14320987654320988




