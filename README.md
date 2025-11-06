## Diplomacy Agent
This was a poject for my Intellgent Agents unit. We had to make a agent to play the game Diplomacy. 
In this project we were tasked to research, design, develop, and evaluate an agent for playing the game Diplomacy.
This was assessed by the performance of the agent we develop and on our written report of the techniques you investigated and developed.

### What is diploamcy
Diplomacy is a strategic board game with seven players competing and cooperating to capture supply centres on a map of Europe.
All players play simultanously and at every turn you make a move for each of your units. 

#### Game engine
- Game Engine: https://github.com/diplomacy/diplomacy
- Documentation: https://diplomacy.readthedocs.io/en/stable/

## How my agent works
I chose MCTS because it scales better to Diplomacy’s simultaneous, multi-
player setting and tight time budget (we had a 1 second time limit per turn), whereas minimax cannot reliably evaluate even
shallow depths under these conditions. 

The first implementation of MCTS didn’t work very well at all as it played essentially
random moves as it wasn’t really able to fit in the 1-time budget yet. I heavily modify it
later on to work a lot better. The main problems were that it was given primarily holding
moves and did almost 0 support orders which is needed to kick enemies of SCs.

#### Heavily modified MCTS to work for Diplomacy constraints: 
- Root-only UCT: Apply UCT at the root only (no deep rollouts) to stay within the 1-second budget.
- Capped joint actions: get_mcts_root_actions(cap) builds a small fixed set of root children by letting each unit choose only {hold, BFS-next-hop toward target SCs},
  then rewrites collisions into legal supports to reduce bounces.
- One-step rollout: simulate_value(...) runs a single adjudication (not to terminal).
- Light opponent model: opponent_orders_random_lite(...) tends to hold on owned SCs, otherwise picks random legal moves.
- Shaped heuristic: Evaluate the post-adjudication state with features that push away from passivity and useless attacks:
  - W_DIFF (our SCs − best rival SCs)
  - W_FOREIGN_SC (reward taking neutral/enemy SCs)
  - HOLD_OFF_SC, HOLD_ON_SC (penalize needless holds on/off SCs)
  - BOUNCE (penalize single-unit attacks that just bounce)
- Effect: improved win/survive rates; behavior shifted from holds/bounces to supported advances and more SC captures, improving win/survive rates in the given
### (if you want a more in depth explanantion read project report or just ask me)

