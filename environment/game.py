from typing import List, Dict, Tuple, Optional
# import torch as t
# import numpy as np
# import pandas as pd
from dataclasses import dataclass
import random
from colorama import Fore, Style
from collections import Counter
from functools import cmp_to_key

"""
Cards mapping goes 
0-12:  2♣3♣4♣5♣6♣7♣8♣9♣10♣J♣Q♣K♣A♣
13-25: 2♦3♦4♦5♦6♦7♦8♦9♦10♦J♦Q♦K♦A♦
26-38: 2♥3♥4♥5♥6♥7♥8♥9♥10♥J♥Q♥K♥A♥
39-51: 2♠3♠4♠5♠6♠7♠8♠9♠10♠J♠Q♠K♠A♠
"""
SUIT_LETTERS = ['c', 'd', 'h', 's']
SUIT_SYMBOLS = [
    '\u2663', # clubs
    '\u2666', # diamonds
    '\u2665', # hearts
    '\u2660', # spades
]
SUIT_COLORS = [
    Fore.GREEN, # clubs
    Fore.BLUE,  # diamonds
    Fore.RED,   # hearts
    Fore.WHITE, # spades
]
RANK_MAP = {
    0: '2',
    1: '3',
    2: '4',
    3: '5',
    4: '6',
    5: '7',
    6: '8',
    7: '9',
    8: '10',
    9: 'J',
    10: 'Q',
    11: 'K',
    12: 'A'
}

ROUND_NAMES = [
    'Pre-Flop',
    'Flop',
    'Turn',
    'River'
]

HAND_ORDERING = [
    'straight-flush',
    'four-of-a-kind',
    'full-house',
    'flush',
    'straight',
    'three-of-a-kind',
    'two-pair',
    'pair',
    'high-card'
]

def get_rank(card: int) -> int:
    return card % 13

def get_suit(card: int) -> int:
    return card // 13

def humanize_card(card: int) -> str:
    rank = get_rank(card)
    suit = get_suit(card)
    return f"{SUIT_COLORS[suit]}{RANK_MAP[rank]}{SUIT_SYMBOLS[suit]}{Style.RESET_ALL}"

def humanize_cards(cards: List[int]) -> str:
    return ' '.join([humanize_card(card) for card in cards])

def make_card(card_str: str) -> int: # *** NOT CORRECT
    rank_str, suit_str = card_str[:-1], card_str[-1]
    assert rank_str in RANK_MAP.values(), f"{rank_str} not a valid rank"
    assert suit_str in SUIT_LETTERS or suit_str in SUIT_SYMBOLS, f"{suit_str} not a valid suit"
    suit_num = SUIT_LETTERS.index(suit_str) if suit_str in SUIT_LETTERS else SUIT_SYMBOLS.index(suit_str)
    rank_num = list(RANK_MAP.keys())[list(RANK_MAP.values()).index(rank_str)]
    return 13 * suit_num + rank_num
    

def get_suit_of_flush(cards: List[int]) -> bool:
    # Count 5 of a suit
    suits = Counter([get_suit(card) for card in cards])
    for s, count in suits.items():
        if count >= 5:
            return s
    return None

# Takes a sorted array of cards
def get_straight(cards: List[int]) -> bool:
    append_ace = [cards[0]] if get_rank(cards[0]) == 12 else []
    
    straight = [cards[0]]
    pivot = get_rank(cards[0])
    for card in cards + append_ace:
        curr_rank = get_rank(card)

        if curr_rank == pivot:
            # skip if same
            continue

        elif (curr_rank + 1) % 13 == pivot:
            # append card if connected
            straight.append(card)
            if len(straight) == 5:
                # return first valid straight (guaranteed to be largest)
                return straight
        else:
            # otherwise, continue with new pivot
            straight = [card]
        pivot = curr_rank

    # return None on failure
    return None

def filter_cards(cards: List[int], filter_value: int, criteria: str = 'rank') -> List[int]:
    match criteria:
        case 'rank':
            get_criteria = get_rank
        case 'suit':
            get_criteria = get_suit

    return [card for card in cards if get_criteria(card) == filter_value]

def get_top_k_cards_excluding(cards: List[int], k: int, excluding_ranks: List[int]) -> List[int]:
    ret = []
    for card in cards:
        if get_rank(card) not in excluding_ranks:
            ret.append(card)
            if len(ret) == k:
                return ret
    return ret


@dataclass
class GameState:
    """
    Represents the state of the session.
    """
    num_wins: int
    num_losses: int
    num_players: int

    """
    Represents the state of the game.
    """
    current_player: int         # Player whose turn it is
    current_round: int          # 0: Pre-Flop, ..., 3: River
    deck: List[int]             # Deck of remaining cards
    board: List[int]            # Community cards shown
    hands: List[List[int]]      # Players' hands, at given index
    tokens: List[List[int]]     # Ownership of tokens, by round (eg, tokens[3][0] is Player 0's token on River)
    # token 0 (best hand) > 1 > 2 > 3 > ... (worst hand)
    num_passes: int             # Number of passes used in game


class Game:
    def __init__(self, num_players: int):
        deck = list(range(52))
        random.shuffle(deck)

        self.num_players = num_players
        self.game_state = GameState(
            num_players=num_players,
            num_wins=0,
            num_losses=0,

            current_player=0,
            current_round=0,
            deck=deck,
            board=[],
            hands=[[] for _ in range(num_players)],
            tokens=[[-1 for _ in range(num_players)] for _ in range(4)],
            # tokens=[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]],
            num_passes=0
        )

    def __str__(self):
        return f"Game (num_players={self.num_players})"
    
    def reset_game(self):
        deck = list(range(52))
        random.shuffle(deck)
        self.game_state.deck = deck
        self.game_state.current_player = 0
        self.game_state.current_round = 0
        self.game_state.board = []
        self.game_state.hands = [[] for _ in range(self.num_players)]
        self.game_state.tokens = [[-1 for _ in range(num_players)] for _ in range(4)]
        self.game_state.num_passes = 0

    def next_round(self):
        curr_round = self.game_state.current_round
        if curr_round == 3:
            self.handle_end_of_game()
            return
        if len(self.get_hand(0)) == 0:
            print("Cannot proceed to next round until all players have cards in their hand.")
            return
        if -1 in self.get_tokens(curr_round):
            print("Cannot proceed to next round until all players have taken a token.")
            return
        self.game_state.current_round += 1
        self.game_state.current_player = self.game_state.current_round  # Increment starting player each round
        self.deal()
        return

    def next_player(self):
        self.game_state.current_player += 1
        self.game_state.current_player %= self.num_players

    def print_game_state(self):
        for attr, value in self.game_state.__dict__.items():
            print(f"{attr}: {value}")

    def print_board(self):
        print(f"Board: {humanize_cards(self.game_state.board)}")

    def print_player_hands(self):
        for i, hand in enumerate(self.game_state.hands):
            poker_hand_cards, poker_hand_type = self.get_poker_hand(i)
            print(f"P{i}: {humanize_cards(hand)}\t({humanize_cards(poker_hand_cards)})\t[{poker_hand_type}]")
        # print(f"Ranking: {', '.join(self.order_hands())}")

    def print_round_tokens(self, round_idx=None, with_player_idx=False):
        round_idx = round_idx if round_idx != None else self.game_state.current_round
        for p in range(self.num_players):
            value = self.get_tokens(round_idx)[p]
            humanized_value = value if value >= 0 else '_'
            if with_player_idx:
                print(f"P{p}: {humanized_value}", end="\t")
            else:
                print(f"{humanized_value}", end="\t")
        print()

    def print_tokens(self):
        for p in range(self.num_players):
            print(f"\tP{p}", end="")
        print()
        for r in range(self.game_state.current_round + 1):
            print(f"R{r}", end="\t")
            self.print_round_tokens(round_idx=r)

    def deal(self):
        if self.game_state.current_round == 0:
            self.deal_hands()
        else:
            self.deal_board()
        

    def deal_hands(self):
        for i in range(self.num_players):
            self.game_state.hands[i] = [self.game_state.deck.pop(0), self.game_state.deck.pop(0)]

    def get_hand(self, player_idx: int) -> List[int]:
        assert 0 <= player_idx < self.num_players, "Must request a valid player index"
        return self.game_state.hands[player_idx]

    def get_tokens(self, round_idx: int) -> List[int]:
        assert 0 <= round_idx < 4, "Must request a valid round index"
        return self.game_state.tokens[round_idx]

    def print_current_state(self):
        print("BOARD:")
        self.print_board()
        self.print_player_hands()
        print()
        print("TOKENS:")
        self.print_tokens()
        print()
        print(f"ROUND: {self.game_state.current_round}")
        print(f"It is currently Player {self.game_state.current_player}'s turn.")
        print()

    def deal_board(self):
        cards_to_deal = 0
        match len(self.game_state.board):
            case 0: cards_to_deal = 3
            case 3: cards_to_deal = 1
            case 4: cards_to_deal = 1

        for _ in range(cards_to_deal):
            card = self.game_state.deck.pop(0)
            self.game_state.board.append(card)

    def take_token(self, token_num: int):
        player_idx = self.game_state.current_player
        round_tokens = self.get_tokens(self.game_state.current_round)

        assert round_tokens[player_idx] != token_num, f"Player {player_idx} already has token {token_num}"

        if (token_num in round_tokens):
            round_tokens[round_tokens.index(token_num)] = -1
        
        round_tokens[player_idx] = token_num

        self.next_player()

    def player_passes(self):
        self.game_state.num_passes += 1
        self.next_player()

    def get_poker_hand(self, player_idx: int) -> tuple[List[int], str]:
        cards = sorted(
            self.game_state.hands[player_idx] + self.game_state.board, 
            key=lambda x: get_rank(x), 
            reverse=True
        )

        if len(cards) == 0:
            return (cards, 'N/A')

        flush_suit = get_suit_of_flush(cards)
        quads, trips, pairs = [], [], []
        for rank, count in Counter([get_rank(card) for card in cards]).items():
            if count == 4: quads.append(rank)
            if count == 3: trips.append(rank)
            if count == 2: pairs.append(rank)
        
        # Check for straight-flush
        if (flush_suit is not None) and (straight_flush := get_straight(filter_cards(cards, flush_suit, 'suit'))):
            return (straight_flush, 'straight-flush')

        # Check for four-of-a-kind
        if len(quads): 
            hand = filter_cards(cards, quads[0]) + get_top_k_cards_excluding(cards, 1, [quads[0]])
            return (hand, 'four-of-a-kind')

        # Check for full-house
        if len(trips) and len(pairs):
            hand = filter_cards(cards, trips[0]) + filter_cards(cards, pairs[0])
            return (hand, 'full-house')
        
        # Check for flush
        if flush_suit is not None: 
            hand = filter_cards(cards, flush_suit, 'suit')[:5]
            return (hand, 'flush')

        # Check for straight
        if straight := get_straight(cards):
            return (straight, 'straight')

        # Check for three-of-a-kind
        if len(trips): 
            hand = filter_cards(cards, trips[0]) + get_top_k_cards_excluding(cards, 2, [trips[0]])
            return (hand, 'three-of-a-kind')

        # Check for two-pair
        if len(pairs) >= 2:
            hand = filter_cards(cards, pairs[0]) + filter_cards(cards, pairs[1]) + get_top_k_cards_excluding(cards, 1, pairs[:2])
            return (hand, 'two-pair')

        # Check for pair
        if len(pairs): 
            hand = filter_cards(cards, pairs[0]) + get_top_k_cards_excluding(cards, 3, [pairs[0]])
            return (hand, 'pair')

        # Otherwise, return high-card
        return (cards[:5], 'high-card')

    """
    returns False if hand_b is better than hand_a, otherwise True
    """
    def compare_hands(self, player_a: int, player_b: int) -> bool:
        hand_a, type_a = self.get_poker_hand(player_a)
        hand_b, type_b = self.get_poker_hand(player_b)

        # Return immediately if hand types differ
        if type_a != type_b:
            return HAND_ORDERING.index(type_a) < HAND_ORDERING.index(type_b)

        # If hand types are the same, check for point of rank divergence
        better = [get_rank(a) > get_rank(b) for a, b in zip(hand_a, hand_b)]
        worse = [get_rank(a) < get_rank(b) for a, b in zip(hand_a, hand_b)]

        for b, w in zip(better, worse):
            if b: return True
            if w: return False
        
        # If hands are equal, return True by default
        return True
            
    def order_hands(self) -> List[int]:
        hands = [(i, self.get_poker_hand(i)) for i in range(self.num_players)]
        
        def compare_hand_tuples(a: Tuple[int, Tuple[List[int], str]], b: Tuple[int, Tuple[List[int], str]]) -> int:
            # Return -1 if a is better than b, 1 if b is better than a, 0 if equal
            return -1 if self.compare_hands(a[0], b[0]) else 1
        
        # Sort using the comparison function converted to a key function
        sorted_hands = sorted(hands, key=cmp_to_key(compare_hand_tuples))
        
        # Extract just the player indices from the sorted tuples
        return [player_idx for player_idx, _ in sorted_hands]

    def check_hands_in_order(self, tokens: List[int]) -> bool:
        players_ordered = [None for _ in range(self.num_players)]
        for i, token in enumerate(tokens):
            players_ordered[token] = i

        for idx in range(self.num_players - 1):
            player_a, player_b = tokens[idx], tokens[idx+1]
            if not self.compare_hands(player_a, player_b):
                return False
        return True

    def handle_end_of_game(self):
        guessed_order = self.get_tokens(3)
        win = self.check_hands_in_order(guessed_order)
        if win:
            print("You win!")
            self.game_state.num_wins += 1
        else:
            print("You lose!")
            print(f"Actual order: {self.order_hands()}")
            self.game_state.num_losses += 1
        self.reset_game()
    
    def get_action(self):
        if len(self.get_hand(0)) == 0:
            input("Press Enter to begin the game...")
            self.deal()
        player_idx = self.game_state.current_player
        player_hand = humanize_cards(self.game_state.hands[player_idx])
        print(f"Round: {self.game_state.current_round}")
        self.print_board()
        print(f"Tokens: ", end="")
        self.print_round_tokens(with_player_idx=True)
        response = input(f"Player {player_idx} [{player_hand}]: What would you like to do? ")
        args = response.strip().split(" ")
        match args[0]:
            case 'take':
                assert len(args) == 2, f"'take' must be followed by a single number from 0-{self.num_players-1}."
                token_num = int(args[1])
                assert token_num in range(self.num_players), f"You can only take a token between 0 and {self.num_players-1}."
                self.take_token(int(args[1]))
            case 'pass':
                assert len(args) == 1, "'pass' does not take any arguments."
                self.player_passes()
            case 'next':
                assert len(args) == 1, "'next' does not take any arguments."
                self.next_round()
            case 'reset':
                assert len(args) == 1, "'reset' does not take any arguments."
                self.reset_game()
            case 'print':
                assert len(args) == 1, "'print' does not take any arguments."
                self.print_current_state()

    




num_players = 4
game = Game(num_players)
while True:
    game.get_action()

# game.print_current_state()

# # Pre-Flop
# game.deal()
# game.take_token(0)
# game.take_token(1)
# game.take_token(2)
# game.take_token(3)
# game.print_current_state()
# game.next_round()

# # Flop
# game.take_token(0)
# game.take_token(1)
# game.take_token(2)
# game.take_token(3)
# game.print_current_state()
# game.next_round()

# # Turn
# game.take_token(0)
# game.take_token(1)
# game.take_token(2)
# game.take_token(3)
# game.print_current_state()
# game.next_round()

# # River
# game.take_token(0)
# game.take_token(1)
# game.take_token(2)
# game.take_token(3)
# game.print_current_state()


# game.print_current_state()
# game.handle_end_of_game()
