#pragma once
#include "../connect4.hpp"
#include "../enums.hpp"

#include <iostream>
#include <limits>
#include <string>

template<int WIDTH, int HEIGHT>
struct C4HumanAgent {
    std::string name;

    C4HumanAgent() : name("Unnamed C4HumanAgent") {}
    C4HumanAgent(std::string name) : name(std::move(name)) {}

    float get_eval(Connect4<WIDTH, HEIGHT>& game) {
        float user_eval;
        do {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "If -1 is winning for YELLOW and 1 is winning for RED, how would you evaluate the game?" << std::endl;
        } while (!(std::cin >> user_eval));

        return user_eval;
    }

    int get_move(Connect4<WIDTH, HEIGHT>& game) {
        game.print_board();
        std::cout << "You are " << disc_to_char(game.next_player()) << std::endl;

        int col;
        while (true) {
            std::cout << "Column: ";
            while (!(std::cin >> col)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid number. Column: ";
            }

            if (game.can_play(col)) break;
            std::cout << "Invalid move" << std::endl;
        }

        return col;
    }

    std::string& get_name() {
        return name;
    }
};
