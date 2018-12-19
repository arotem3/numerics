#include <vector>
#include <iostream>


namespace vector_operators {
    //--- element wise printing ---//
    template<class T>
    std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
        for (T t : v) {
            out << t << " ";
        }
        return out;
    }

    //--- element wise addition ---//
    template<class S, class T>
    std::vector<S> operator+(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator+() error: cannot add vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<S> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) + v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<T> operator+(std::vector<T>& v, T c) {
        std::vector<T> V;
        for (T t : v) {
            V.push_back(  t + c  );
        }
        return V;
    }

    template<class S, class T>
    std::vector<S>& operator+=(std::vector<T>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator+() error: cannot add vectors of different lengths" << std::endl;
            return {};
        }

        for (int i(0); i < v1.size(); ++i) {
            v1.at(i) += v2.at(i);
        }
        return v1;
    }

    template<class T>
    std::vector<T>& operator+=(std::vector<T>& v, T c) {
        for (T& t : v) {
            t += c;
        }
        return v;
    }

    template<class T>
    std::vector<T>& operator++(std::vector<T>& V) {
        for (T &t: V) {
            t++;
        }
        return V;
    }

    //--- element wise subtraction ---//
    template<class S, class T>
    std::vector<S> operator-(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator-() error: cannot subtract vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<S> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) - v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<T> operator-(std::vector<T>& v, T c) {
        std::vector<T> V;
        for (T t : v) {
            V.push_back(  t - c  );
        }
        return V;
    }

    template<class S, class T>
    std::vector<S>& operator-=(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator-() error: cannot subtract vectors of different lengths" << std::endl;
            return {};
        }

        for (int i(0); i < v1.size(); ++i) {
            v1.at(i) -=  v2.at(i);
        }
        return v1;
    }

    template<class T>
    std::vector<T>& operator-=(std::vector<T>& v, T c) {
        for (T& t : v) {
            t -= c;
        }
        return v;
    }

    template<class T>
    std::vector<T>& operator--(std::vector<T>& V) {
        for (T &t: V) {
            t--;
        }
        return V;
    }

    //--- element wise multiplication ---//
    template<class S, class T>
    std::vector<S> operator*(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator*() error: cannot multiply vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<S> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) * v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<T> operator*(std::vector<T>& v, T c) {
        std::vector<T> V;
        for (T t : v) {
            V.push_back(  t * c  );
        }
        return V;
    }

    template<class S, class T>
    std::vector<S>& operator*=(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator*() error: cannot multiply vectors of different lengths" << std::endl;
            return {};
        }

        for (int i(0); i < v1.size(); ++i) {
            v1.at(i) *=  v2.at(i);
        }
        return v1;
    }

    template<class T>
    std::vector<T>& operator*=(std::vector<T>& v, T c) {
        for (T& t : v) {
            t *= c;
        }
        return v;
    }

    //--- element wise division ---//
    template<class S, class T>
    std::vector<S> operator/(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator/() error: cannot divide vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<S> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) / v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<T> operator/(std::vector<T>& v, T c) {
        std::vector<T> V;
        for (T t : v) {
            V.push_back(  t / c  );
        }
        return V;
    }

    template<class S, class T>
    std::vector<S>& operator/=(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator/() error: cannot divide vectors of different lengths" << std::endl;
            return {};
        }

        for (int i(0); i < v1.size(); ++i) {
            v1.at(i) /= v2.at(i);
        }
        return v1;
    }

    template<class T>
    std::vector<T>& operator/=(std::vector<T>& v, T c) {
        for (T& t : v) {
            t /= c;
        }
        return v;
    }

    //--- element wise equality ---//
    template<class S, class T>
    std::vector<bool> operator==(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator==() error: cannot compare vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) == v2.at(i)  );
        }
        return V;
    }

    //--- element wise inequality ---//
    template<class S, class T>
    std::vector<bool> operator!=(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator!=() error: cannot compare vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) != v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<bool> operator!=(std::vector<T>& v1, T c) {
        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) != c  );
        }
        return V;
    }

    //--- element wise less than ---//
    template<class S, class T>
    std::vector<bool> operator<(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator<() error: cannot compare vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) < v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<bool> operator<(std::vector<T>& v1, T c) {
        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) < c  );
        }
        return V;
    }

    template<class S, class T>
    std::vector<bool> operator<=(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator<=() error: cannot compare vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) <= v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<bool> operator<=(std::vector<T>& v1, T c) {
        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) <= c  );
        }
        return V;
    }

    //--- element wise greater than ---//
    template<class S, class T>
    std::vector<bool> operator>(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator>() error: cannot compare vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) > v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<bool> operator>(std::vector<T>& v1, T c) {
        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) > c  );
        }
        return V;
    }

    template<class S, class T>
    std::vector<bool> operator>=(std::vector<S>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator>=() error: cannot compare vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) >= v2.at(i)  );
        }
        return V;
    }

    template<class T>
    std::vector<bool> operator>=(std::vector<T>& v1, T c) {
        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) >= c  );
        }
        return V;
    }

    //--- element wise and/or/not ---//
    std::vector<bool> operator&&(std::vector<bool>& v1, std::vector<bool>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator&() error: cannot perform logic operation on vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) && v2.at(i)  );
        }
        return V;
    }

    std::vector<bool> operator||(std::vector<bool>& v1, std::vector<bool>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "operator|() error: cannot perform logic operation on vectors of different lengths" << std::endl;
            return {};
        }

        std::vector<bool> V;
        for (int i(0); i < v1.size(); ++i) {
            V.push_back(  v1.at(i) || v2.at(i)  );
        }
        return V;
    }

    //--- any/or functions ---//
    bool any(std::vector<bool> V) {
        for (bool t : V) {
            if (t) return true;
        }
        return false;
    }

    bool all(std::vector<bool> V) {
        for (bool t: V) {
            if (!t) return false;
        }
        return true;
    }

    //---element wise cmath++ ---//
    template<class T>
    std::vector<T> sin(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::sin(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> cos(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::cos(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> tan(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::tan(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> asin(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::asin(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> acos(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::acos(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> atan(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::atan(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> atan2(std::vector<T>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "atan2() error: vectors must be the same length. currently " << v1.size() << " and " << v2.size() << " element vectors." << std::endl;
            return {};
        }
        std::vector<T> vout;
        for (size_t i(0); i < v1.size(); ++i) {
            vout.push_back(atan2(v1.at(i), v2.at(i)));
        }
        return vout;
    }
    
    template<class T>
    std::vector<T> cosh(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::cosh(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> sinh(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::sinh(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> tanh(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::tanh(t));
        }
        return vout;
    }
    
    template<class T>
    std::vector<T> acosh(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::acosh(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> asinh(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::asinh(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> atanh(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::atanh(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> exp(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::exp(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> log(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::log(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> log10(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::log10(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> log2(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::log2(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> pow(std::vector<T>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "pow() error: vectors must be the same length. currently " << v1.size() << " and " << v2.size() << " element vectors." << std::endl;
            return {};
        }
        std::vector<T> vout;
        for (size_t i(0); i < v1.size(); ++i) {
            vout.push_back(std::pow(v1.at(i), v2.at(i)));
        }
        return vout;
    }

    template<class T>
    std::vector<T> pow(std::vector<T>& v, double p) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::pow(t, p));
        }
        return vout;
    }

    template<class T>
    std::vector<T> pow(double b, std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::pow(b, t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> sqrt(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::sqrt(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> cbrt(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::cbrt(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> erf(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::erf(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> erfc(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::erfc(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> tgamma(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::tgamma(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> lgamma(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::lgamma(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> ceil(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::ceil(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> floor(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::floor(t));
        }
        return vout;
    }

    template<class T>
    std::vector<T> fmod(std::vector<T>& v1, std::vector<T>& v2) {
        if (v1.size() != v2.size()) {
            std::cerr << "fmod() error: vectors must be the same length. currently " << v1.size() << " and " << v2.size() << " element vectors." << std::endl;
            return {};
        }
        std::vector<T> vout;
        for (size_t i(0); i < v1.size(); ++i) {
            vout.push_back(std::fmod(v1.at(i), v2.at(i)));
        }
        return vout;
    }

    template<class T>
    std::vector<T> abs(std::vector<T>& v) {
        std::vector<T> vout;
        for (T t : v) {
            vout.push_back(std::abs(t));
        }
        return vout;
    }
    
    template<class T>
    double sum(std::vector<T>& v) {
        double s = 0;
        for (T t : v) {
            s += t;
        }
        return s;
    }

    template<class T>
    double norm(std::vector<T> v, double p = 2.0) {
        v = abs(v);
        v = pow(v,p);
        return sum(v);
    }

    template<class T>
    double norm(std::vector<T> v, std::string p) {
        for (char& c : p) {
            c = toupper(c);
        }
        if (p != "INF" || p != "-INF") {
            std::cerr << "norm() error: invalid p input. accepting 'inf' or '-inf' only. (case insensitive)" << std::endl;
            return -1;
        }
        v = abs(v);
        double n=0;
        if (p == "INF") n = *max_element(v.begin(), v.end());
        else if (p == "-INF") n = *min_element(v.begin(), v.end());
        return n;
    }
}