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