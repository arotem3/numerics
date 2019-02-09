#include "gnuplot_i.hpp"

// wrappers for gnuplot that behave like matlab

inline std::string ls(char c) {
    if (c == '-') return "lt 1 ";       // solid line
    else if (c == '=') return "lt 2 ";  // dashed
    else if (c == ':') return "lt 4 ";  // dotted
    else if (c == '!') return "lt 5 ";  // dot-dashed
    
    else if (c == 'b') return "lc rgb '#0060ad' ";  // blue
    else if (c == 'r') return "lc rgb '#dd181f' ";  // red
    else if (c == 'k') return "lc rgb '#000000' ";  // black
    else if (c == 'p') return "lc rgb '#990042' ";  // purple
    else if (c == 'g') return "lc rgb '#31f120' ";  // green

    else if (c == 'o') return "pt 6 ";  // circle
    else if (c == 's') return "pt 5 ";  // square
    else if (c == 't') return "pt 9 ";  // triangle
    else if (c == '*') return "pt 3 ";  // star
    else if (c == 'd') return "pt 12 "; // diamond
    else if (c == '.') return "pt 7 ";  // dot

    else return "";
}

std::string to_cmd(const std::string& s) {
    std::string cmd;
    bool lines = false, points = false, color = false;
    std::string Lt = "-=:!", Pt = "ost*d.", Ct = "brkpg";
    for (char c : s) { // consider only the first command and skip anything that wants to overwrite
        if (Lt.find(c) != std::string::npos) {
            if (lines) continue;
            else lines = true;
        } else if (Pt.find(c) != std::string::npos) {
            if (points) continue;
            else points = true;
        } else if (Ct.find(c) != std::string::npos) {
            if (color) continue;
            else color = true;
        }

        cmd += ls(c);
    }

    if (!color) {
        cmd += "lc rgb '#000000' ";
    }

    if (lines && points) return "linespoints " + cmd + "lw 2 ps 1.5";
    else if (points) return "points " + cmd + "ps 1.5";
    else return "lines " + cmd + "lw 2";
}

template<class T, class S>
void plot(Gnuplot& fig, const T& x, const S& y, const std::string& label = "", const std::string& linespec = "-k") {
    fig.set_style( to_cmd(linespec) );
    fig.plot_xy(x, y, label);
}

template<class T, class S>
inline void plot(const T& x, const S& y, const std::string& label = "", const std::string& linespec = "-k") {
    Gnuplot fig;
    plot(fig, x, y, label, linespec);
}

template<class T, class S>
inline void lines(Gnuplot& fig, const T& x, const S& y, const std::string& label = "", char clr = 'k') {
    plot(fig, x, y, label, std::string("-") + clr);
}

template<class T, class S>
inline void lines(const T& x, const S& y, const std::string& label = "", char clr = 'k') {
    plot(x, y, label, std::string("-") + clr);
}

template<class T, class S>
inline void scatter(Gnuplot& fig, const T& x, const S& y, const std::string& label = "", char clr = 'k') {
    plot(fig, x, y, label, std::string("o") + clr);
}

template<class T, class S>
inline void scatter(const T& x, const S& y, const std::string& label = "", char clr = 'k') {
    plot(x, y, label, std::string("o") + clr);
}

template<class A,class B, class C>
void plot3d(Gnuplot& p, const A& x, const B& y, const C& z) {
    p << "set palette defined (0  0.0 0.0 0.5, 1  0.0 0.0 1.0, 2  0.0 0.5 1.0, 3  0.0 1.0 1.0, 4  0.5 1.0 0.5, 5  1.0 1.0 0.0, 6  1.0 0.5 0.0, 7  1.0 0.0 0.0,8 0.5 0.0 0.0)";
    p << "set dgrid3d 30,30 splines";
    p.set_style("pm3d");
    p.plot_xyz(x,y,z);
}