#include "statistics.hpp"

std::ostream& statistics::operator<<(std::ostream &out, const mean_test &x) {
    std::string title;
    char var_score;
    std::string hh;
    bool df_check;
    if (x.T == z1) {
        title = "One Sample z-test";
        var_score = 'z';
        hh = "mean";
        df_check = false;
    } else if (x.T == z2) {
        title = "Unpaired Two Sample Difference of Means z-test";
        var_score = 'z';
        hh = "difference of means";
        df_check = false;
    } else if (x.T == z2_paired) {
        title = "Paired Two Sample Difference of Means z-test";
        var_score = 'z';
        hh = "difference of means";
        df_check = false;
    } else if (x.T == t1) {
        title = "One Sample Student's t-test";
        var_score = 't';
        hh = "mean";
        df_check = true;
    } else if (x.T == t2) {
        title = "Unpaired Two Sample Difference of Means t-test";
        var_score = 't';
        hh = "difference of means";
        df_check = true;
    } else if (x.T == t2_paired) {
        title = "Paired Two Sample Difference of Means t-test";
        var_score = 't';
        hh = "difference of means";
        df_check = true;
    }
    out << "----------------------------------------------------" << std::endl;
    out << "\t" << title << std::endl << std::setprecision(5);
    out << var_score << "-score = " << x.score;
    if (df_check) out << ",\t" << "df = " << x.df << std::endl;
    else out << std::endl;
    out << "p-value = " << x.p << std::endl << std::setprecision(3);

    std::string hyp;
    if (x.H1 == NEQ) hyp = " is not equal to ";
    else if (x.H1 == GREATER) hyp = " is greater than to ";
    else if(x.H1 == LESS) hyp = " is less than to ";

    out << "H_1 : the " << hh << hyp << x.test_mu << std::endl;
    out << x.conf_level*100 << " percent confidence interval is: ["  << x.conf_interval.first << " , " << x.conf_interval.second << "]" << std::endl;

    out << "mean = " << x.S.x_bar << "\t\tstd dev = " << x.S.x_sd << std::endl;

    out << "----------------------------------------------------" << std::endl;
    return out;
}

std::ostream& statistics::operator<<(std::ostream &out, const prop_test &x) {
    out << "----------------------------------------------------" << std::endl;
    out << "\tOne Sample Test for Population Proportion" << std::endl << std::setprecision(5);
    out << "t-score = " << x.score << ",\tdf = " << x.df << std::endl;
    out << "p-value = " << x.p << std::endl << std::setprecision(3);
    
    std::string hyp;
    if (x.H1 == NEQ) hyp = " is not equal to ";
    else if (x.H1 == GREATER) hyp = " is greater than to ";
    else if(x.H1 == LESS) hyp = " is less than to ";

    out << "H_1 : the true population proportion" << hyp << x.test_p0 << std::endl;
    out << x.conf_level*100.0 << " percent confidence interval is: ["  << x.conf_interval.first << " , " << x.conf_interval.second << "]" << std::endl;
    out << "prop = " << x.p1 << "\t\tprop sd = " << x.p1_sd << std::endl;
    out << "----------------------------------------------------" << std::endl;
    return out;
}

std::ostream& statistics::operator<<(std::ostream &out, const category_test &x) {
    std::string test_type;
    std::string hyp;
    if (x.H1 == hypothesis::GOF) {
        test_type = "Goodness of Fit";
        hyp = "the observed data is not well represented by the expected distribution.";
    } else if (x.H1 == hypothesis::HOMOGENEITY) {
        test_type = "Homogeneity";
        hyp = "the observed data is not from a homogeneous population.";
    } else if (x.H1 == hypothesis::INDEPENDENCE) {
        test_type = "Independence";
        hyp = "the observed data is not from independent samples.";
    }

    out << "----------------------------------------------------" << std::endl;
    out << "\tChi Squared Test for " << test_type << std::endl << std::fixed << std::setprecision(5);
    out << "X^2 = " << x.X2 << ",\tdf = " << x.df << std::endl;
    out << "p-value = " << x.p << std::endl << std::setprecision(3);
    out << "H_1 : " << hyp << std::endl;
    out << "----------------------------------------------------" << std::endl;
    return out;
}

/* LATEX_PRINT : prints test results to output stream formatted for LaTeX.
 * --- out : output/file stream to print to.
 * --- x : test result. */
void statistics::LaTeX_print(std::ostream &out, const mean_test &x) {
    std::string title;
    char var_score;
    std::string hh;
    bool df_check;
    if (x.T == z1) {
        title = "One Sample z-test";
        var_score = 'z';
        hh = "true mean";
        df_check = false;
    } else if (x.T == z2) {
        title = "Unpaired Two Sample Difference of Means z-test";
        var_score = 'z';
        hh = "true difference of means";
        df_check = false;
    } else if (x.T == z2_paired) {
        title = "Paired Two Sample Difference of Means z-test";
        var_score = 'z';
        hh = "true difference of means";
        df_check = false;
    } else if (x.T == t1) {
        title = "One Sample Student's t-test";
        var_score = 't';
        hh = "true difference of means";
        df_check = true;
    } else if (x.T == t2) {
        title = "Unpaired Two Sample Difference of Means t-test";
        var_score = 't';
        hh = "true difference of means";
        df_check = true;
    } else if (x.T == t2_paired) {
        title = "Paired Two Sample Difference of Means t-test";
        var_score = 't';
        hh = "true difference of means";
        df_check = true;
    }
    out << "\\hline\n\\vspace{2mm}\n\n\\begin{center}\n\\large \\textbf{"<< title << "}\n\\end{center}\n\n" << std::setprecision(5);
    out << "$" << var_score << "$-score = " << x.score;
    if (df_check) out << "\\hspace{2cm}$df$ = " << x.df << "\n\n";
    else out << "\n\n";
    out << "$p$-value = " << x.p << "\n\n" << std::setprecision(2);

    std::string hyp;
    if (x.H1 == NEQ) hyp = " is not equal to ";
    else if (x.H1 == LESS) hyp = " is greater than or equal to ";
    else if(x.H1 == GREATER) hyp = " is less than or equal to ";

    out << "$H_1$ : the " << hh << hyp << x.test_mu << ".\n\n";
    out << "The " << x.conf_level*100 << "\\% confidence interval is: ["  << x.conf_interval.first << " , " << x.conf_interval.second << "]" << "\n\n";

    out << "$\\bar{X} = $" << x.S.x_bar << "\\hspace{2cm}$s_X = $" << x.S.x_sd << "\n\n";

    out << "\\vspace{2mm}\n\\hline\n\n" << std::endl;
}

/* LATEX_PRINT : prints test results to output stream formatted for LaTeX.
 * --- out : output/file stream to print to.
 * --- x : test result. */
void statistics::LaTeX_print(std::ostream &out, const prop_test &x) {
    out << "\\hline\n\\vspace{2mm}\n\n\\begin{center}\n\\large \\textbf{One Sample Test for Population Proportion}\n\\end{center}\n\n" << std::setprecision(5);
    out << "$t$-score = " << x.score;
    out << "\\hspace{2cm}$df$ = " << x.df << "\n\n";
    out << "$p$-value = " << x.p << "\n\n" << std::setprecision(2);

    std::string hyp;
    if (x.H1 == NEQ) hyp = " is not equal to ";
    else if (x.H1 == LESS) hyp = " is greater than or equal to ";
    else if(x.H1 == GREATER) hyp = " is less than or equal to ";

    out << "$H_1$ : the true population proportion" << hyp << x.test_p0 << ".\n\n";
    out << "The " << x.conf_level*100.0 << "\\% confidence interval is: ["  << x.conf_interval.first << " , " << x.conf_interval.second << "]" << "\n\n";

    out << "$\\tilde{p} = $" << x.p1 << "\\hspace{2cm}$s_{\\tilde{p}} = $" << x.p1_sd << "\n\n";

    out << "\\vspace{2mm}\n\\hline\n\n" << std::endl;
}

/* LATEX_PRINT : prints test results to output stream formatted for LaTeX.
 * --- out : output/file stream to print to.
 * --- x : test result. */
void statistics::LaTeX_print(std::ostream &out, const category_test &x) {
    std::string test_type;
    std::string hyp;
    if (x.H1 == hypothesis::GOF) {
        test_type = "Goodness of Fit";
        hyp = "the observed data is not well represented by the expected distribution.";
    } else if (x.H1 == hypothesis::HOMOGENEITY) {
        test_type = "Homogeneity";
        hyp = "the observed data is not from a homogeneous population.";
    } else if (x.H1 == hypothesis::INDEPENDENCE) {
        test_type = "Independence";
        hyp = "the observed data is not from independent samples.";
    }

    out << "\\hline\n\\vspace{2mm}\n\n\\begin{center}\n\\large \\textbf{$\\chi^2$ test for "<< test_type << "}\n\\end{center}\n\n" << std::fixed << std::setprecision(5);
    out << "$\\chi^2$ = " << x.X2 << "\\hspace{2cm}$df$ = " << x.df << "\n\n";
    out << "$p$-value = " << x.p << "\n\n" << std::setprecision(2);
    out << "\\vspace{2mm}\n\\hline\n\n" << std::endl;
}