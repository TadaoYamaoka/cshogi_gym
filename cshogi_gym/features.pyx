from cshogi import *
from cshogi.gym_shogi.envs import ShogiEnv

import numpy as np
cimport numpy as np

# 入力特徴量：駒の種類×手番 + 持ち駒の種類×手番 + 繰り返し数
FEATURES_NUM = 14 * 2 + 7 * 2 + 1

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 指し手を表すラベルの数
MAX_MOVE_LABEL_NUM = len(MOVE_DIRECTION) + 7 # 7はhand piece

def make_position_features(env, np.ndarray features):
    cdef int c
    cdef int hp
    cdef int num
    cdef float max_hp_num

    board = env.board
    # piece
    board.piece_planes_rotate(features)
    pieces_in_hand = board.pieces_in_hand
    if board.turn == WHITE:
        # 白の場合、反転
        pieces_in_hand = (pieces_in_hand[1], pieces_in_hand[0])
    for c, hands in enumerate(pieces_in_hand):
        for hp, num in enumerate(hands):
            if hp == HPAWN:
                max_hp_num = 8
            elif hp == HBISHOP or hp == HROOK:
                max_hp_num = 2
            else:
                max_hp_num = 4

            features[28 + c * 7 + hp].fill(num / max_hp_num)
    # 繰り返し数
    cdef float repetitions = env.repetition_hash[board.zobrist_hash()]
    features[42] = (repetitions - 1) / 3

def make_output_label(int move, int turn):
    cdef int dir
    cdef int to_sq
    cdef int from_sq
    cdef int to_file, to_rank
    cdef int from_file, from_rank

    if turn == WHITE:
        move = move_rotate(move)

    # 移動先座標
    to_sq = move_to(move)
    # 移動方向((8方向+桂馬2方向)×成2+持ち駒7種類)
    if not move_is_drop(move):
        from_sq = move_from(move)

        to_file, to_rank = divmod(to_sq, 9)
        from_file, from_rank = divmod(from_sq, 9)

        if to_file == from_file:
            if to_rank > from_rank:
                dir = UP
            else:
                dir = DOWN
        elif to_rank == from_rank:
            if to_file > from_file:
                dir = LEFT
            else:
                dir = RIGHT
        elif to_rank - from_rank == 2 and abs(to_file - to_file):
            if to_file > from_file:
                dir = UP2_LEFT
            else:
                dir = UP2_RIGHT
        elif to_file > from_file:
            if to_rank > from_rank:
                dir = UP_LEFT
            else:
                dir = DOWN_LEFT
        else:
            if to_rank > from_rank:
                dir = UP_RIGHT
            else:
                dir = DOWN_RIGHT

        if move_is_promotion(move):
            dir += 10
    else:
        dir = 20 + move_drop_hand_piece(move)

    return 9 * 9 * dir + to_sq

def get_legal_moves_labels(board):
    cdef int move

    cdef list legal_labels = []
    cdef list legal_moves = []
    for move in board.legal_moves:
        legal_labels.append(make_output_label(move, board.turn))
        legal_moves.append(move)

    return legal_moves, legal_labels

def get_legal_labels(board):
    cdef int move

    cdef list legal_labels = []
    for move in board.legal_moves:
        legal_labels.append(make_output_label(move, board.turn))

    return legal_labels

def make_position_features_vec(list envs, np.ndarray features_vec):
    for env, features in zip(envs, features_vec):
        make_position_features(env, features)
