auto pi = TMath::Pi()

// function code in C
double single(double *x, double *par) {
    return pow(sin(pi*par[0]*x[0])/(pi*par[0]*x[0]),2);
}

double nslit0(double *x, double *par) {
    
}