from core.environment import CardGameEnv
from core.actions import legal_actions_draw, legal_actions_discard, legal_actions_flip
from core.cards import card_str
from core.actions import view_drawn_cards
import GUI.screen as Screen_class
import GUI.card_display as card_graphics
import pygame
import sys

def print_scores(env: CardGameEnv):
    print("\nScores after round", env.game_state.round_number)
    for i, player in enumerate(env.player_state):
        print(f"Player {i}: Individual scores: {player.individual_scores} | Total score: {player.score}")

def print_cards(player,game_display):
    for i, c in enumerate(player.cards):
        visible = player.is_face_up(i) or player.is_privately_seen(i)
        #display = card_str(c) if visible else "🂠" 
       # flipped_mark = " (flipped)" if player.is_face_up(i) else ""
        #print(f"{i}: {display}{flipped_mark}") #Here I need to draw the card
        #c is the card tuple
        #I need to put each of the cards in different spots.
        #TODO: Make this so that it can pick where the cards go and be centered no matter how many cards
        #TODO: Make it so this is dependent on player number 
        x_pos_list = (600,840,600,840)
        y_pos_list = (650,650,420,420)
        print("current card is"+ str(c))
        #print(visible)
        #if visible==True:
        if player.is_face_up(i)==True:
            card_graphics.draw_card(game_display,x_pos_list[i],y_pos_list[i],c[0],c[1])
            pygame.draw.circle(game_display,(255,0,0),(x_pos_list[i]-20,y_pos_list[i]),10)
            print("block 1 success")
        elif player.is_privately_seen(i)==True:
            card_graphics.draw_card(game_display,x_pos_list[i],y_pos_list[i],c[0],c[1])
            print("block 2 success")
        #there's something here that's preventing all of the cards from getting drawn, even though I can see that it runs 4 times
        #this set gets three cards drawn, but not the 4th. So it must not meet any of thse conditions somehow
        elif visible==False:
            print("block 3 success")
            pygame.draw.rect(game_display,(20,25,112),pygame.Rect(x_pos_list[i],y_pos_list[i],120,180))
        pygame.display.flip()

#literally just prints the available actions
def print_legal_actions(actions):
    for idx, act in enumerate(actions):
        desc = act.type.name
        if act.data is not None:
            desc += f" ({act.data})"
        print(f"{idx}: {desc}")

#so here is where the player interaction occurs
def choose_action(actions):
    #I need to alter how "choice" gets set.
    #TODO: set each of the cards to be an object that I can use with collidepoint to make this logic easier
    #Or maybe I can add coordinates to the card data structure. I would need to make it a list
    choice = None
    while choice is None:
      #  for event in pygame.event.get():
        #    if event==pygame.MOUSEBUTTONDOWN:
         #       x_selected,y_selected = pygame.mouse.get_pos()
                #I need to cast choice (AKA the action index) to specific action sequences

        try:
            idx = int(input("Choose action index: "))
            if 0 <= idx < len(actions):
                choice = actions[idx]
            else:
               print("Invalid index")
        except ValueError:
            print("Enter a number")
    return choice

def play_turn(env: CardGameEnv,game_display):
    game_state = env.game_state
    round_state = env.round_state
    player = env.player_state[game_state.current_player]

    # Skip player turn if all cards are flipped
    if player.is_face_up_all():
        print(f"\nPlayer {game_state.current_player} | Phase: all cards flipped")
        env.advance_turn()
        return

    # Play hand -------------------------------------------------------
    print(f"\nPlayer {player.player_number}")

    # Display discard pile
    num_discard = min(len(round_state.discard_pile), player.num_cards)
    assert num_discard >= 1
    top_discards = round_state.discard_pile[-num_discard:]
    #Here I need to make it display the card instead of printing it
    print("\nTop discard pile:", " ".join(card_str(c) for c in top_discards))
    #TODO: make this work with multiple discards, not just the very top one. Or make it so that you can click through the discards
    #Or click to bring up a window to view them
    #(690,360) is centered
    card_graphics.draw_card(game_display,690,160,top_discards[-1][0],top_discards[-1][1])
    pygame.display.flip()

    # Display player's cards
    print("Your cards:")
    print_cards(player,game_display)

    # Draw. 
    # I don't need to print any actions any more. The player just needs to be able to do each action
    #So I think I need to go rewrite the logic in the actions code
    print("\nDraw options:")
    #So it specifies the possible draw actions with legal_actions_draw(). These are either drawing from the deck or picking up a discard
    actions = legal_actions_draw(env)
    #then it prints them out
    print_legal_actions(actions)
    #and then it gets the action the player chooses with choose action
    #So choose action does the actual action logic
    draw_action = choose_action(actions)
    #then prints those out
    drawn_cards = view_drawn_cards(env, draw_action)
    print("\nDrawn card(s):", " ".join(card_str(c) for c in drawn_cards))

    #Each turn has a draw, discard, and a flip
    # Discard
    print("\nDiscard options:")
    actions = legal_actions_discard(env, draw_action)
    print_legal_actions(actions)
    discard_action = choose_action(actions)

    # Flip
    print("\nFlip options:")
    actions = legal_actions_flip(env, draw_action, discard_action)
    print_legal_actions(actions)
    flip_action = choose_action(actions)


    # Execute the composite action via the environment step (pass previewed drawn_cards)
    obs, reward, done, info = env.step((draw_action, discard_action, flip_action))

    # Display hand state from env for the acting player
    print("Hand at end of turn:")
    print_cards(player,game_display)

def run_terminal_game(env: CardGameEnv):

    print(80*"=")
    print("Game begins!")
    print(80*"=")
    #I would like this to just be an instance of the screen class that I can create in main_terminal and pass to this function
   # pygame.init()
    #SCREEN_SIZE = 900
    #I might want to modify this later
   # SCREEN = pygame.display.set_mode((1500, SCREEN_SIZE))
   # CLOCK = pygame.time.Clock()

    window = Screen_class.Screen(pygame.display.set_mode((1500, 900)))

    
    while not env.game_state.complete:

        #make game close on x-button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.game_state.complete = True
                return

        #print(f'\nRound {env.game_state.round_number+1} ===============================')

        window.get_screen.fill((14,97,36))
        #so step one is initializing the round. I have code to draw the cards when I need it. I'll just need to pass the window
        #to wherever needs it
        env.initialize_round()

     
        pygame.display.flip()

       # env.initialize_round()



        while not env.round_state.complete:
            play_turn(env, window.get_screen)
            pygame.display.flip()

        # Advance round
       # env.advance_round()
       # print_scores(env)

    pygame.quit()
    sys.exit()

    print(80*"=")
    print("\nGame over! All rounds completed.")
    print(80*"=")