#include <iostream>
#include <string>

// this file was taken from gnu_plot

void wait_for_key (std::string s) {
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)  // every keypress registered, also arrow keys
        cout << endl << "Press any key to continue..." << endl;

        FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
        _getch();
    #elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        std::cout << std::endl << s << std::endl;

        std::cin.clear();
        std::cin.ignore(std::cin.rdbuf()->in_avail());
        std::cin.get();
    #endif
        return;
}

void wait_for_key () {
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)  // every keypress registered, also arrow keys
        cout << endl << "Press any key to continue..." << endl;

        FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
        _getch();
    #elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
        std::cout << std::endl << "Press the ENTER key to continue..." << std::endl;

        std::cin.clear();
        std::cin.ignore(std::cin.rdbuf()->in_avail());
        std::cin.get();
    #endif
        return;
}