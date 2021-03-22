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

Now try to do actual Q-learning. And maybe clean up the code