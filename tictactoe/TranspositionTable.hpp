#pragma once
#include <unordered_map>
#include <vector>
#include <random>

#include "./enums.hpp"

struct TranspositionTableEntry {
    float score;
    int depth;
    NodeType type;
};

template<int BOARD_SIDE_LEN, int MAX_ELEMENTS = 65536>
struct TranspositionTable {
    long long x_to_move;
    std::unordered_map<long long, TranspositionTableEntry> map;
    std::vector<long long> rand_numbers;

    TranspositionTable() {
        rand_numbers.reserve(BOARD_SIDE_LEN * BOARD_SIDE_LEN * 2);
        std::mt19937 mt_generator(std::random_device{}());
        for (int i = 0; i < BOARD_SIDE_LEN * BOARD_SIDE_LEN * 2; i++) {
            rand_numbers.push_back(mt_generator());
        }

        x_to_move = mt_generator();

        map.reserve(MAX_ELEMENTS);
    }

    long long hash(const std::array<BoardSquare, BOARD_SIDE_LEN * BOARD_SIDE_LEN> &board, BoardSquare player_turn) {
        long long h = 0;
        if (player_turn == BoardSquare::X) {
            h ^= x_to_move;
        }

        for (int i = 0; i < BOARD_SIDE_LEN * BOARD_SIDE_LEN; i++) {
            if (board[i] == BoardSquare::EMPTY) continue;
            long long j = board[i];
            h ^= rand_numbers[i * 2 + j];
        }

        return h;
    }

    void insert(long long key, const TranspositionTableEntry& entry) {
        if (map.size() >= MAX_ELEMENTS) {
            map.erase(map.begin());
        }

        map[key] = entry;
    }

    bool contains(long long key) {
        return map.contains(key);
    }

    TranspositionTableEntry get(long long key) {
        return map.find(key)->second;
    }
};
