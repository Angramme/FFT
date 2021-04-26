
#define SAMPLES_N (2048)
#define PLOT_OUTPUT 1
#define SAME_EPS 0.001

// #pragma GCC optimize("Ofast")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,fma")
// #pragma GCC optimize("unroll-loops")

#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <assert.h>
 
using namespace std;
 
typedef long long ll;
typedef long double ld;
typedef pair<int,int> p32;
typedef pair<ll,ll> p64;
typedef pair<double,double> pdd;
typedef vector<ll> v64;
typedef vector<int> v32;
typedef vector<vector<int> > vv32;
typedef vector<vector<ll> > vv64;
typedef vector<vector<p64> > vvp64;
typedef vector<p64> vp64;
typedef vector<p32> vp32;
ll MOD = 998244353;
double eps = 1e-12;
#define forn(i,e) for(ll i = 0; i < e; i++)
#define forsn(i,s,e) for(ll i = s; i < e; i++)
#define rforn(i,s) for(ll i = s; i >= 0; i--)
#define rforsn(i,s,e) for(ll i = s; i >= e; i--)
#define ln "\n"
#define dbg(x) cout<<#x<<" = "<<x<<ln
#define mp make_pair
#define pb push_back
#define fi first
#define se second
#define INF 2e18
#define fast_cin() ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)
#define all(x) (x).begin(), (x).end()
#define sz(x) ((ll)(x).size())
 
#define PI 3.14159265359

#if PLOT_OUTPUT
#include "matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif


class CPX{
public:
    double r;
    double i;

    CPX operator*(const CPX& other) const {
        return {r*other.r-i*other.i, r*other.i+i*other.r};
    }
    void operator*=(const CPX& other){
        const auto old_r = r;
        r = r*other.r-i*other.i;
        i = old_r*other.i+i*other.r;
    }
    CPX operator-(const CPX& other) const {
        return {r-other.r, i-other.i};
    }
    CPX operator+(const CPX& other) const {
        return {r+other.r, i+other.i};
    }
    void operator+=(const CPX& other){
        r += other.r;
        i += other.i;
    }
    void operator=(const CPX& other){
        r = other.r;
        i = other.i;
    }
};

bool operator!=(const CPX& a, const CPX& b){
    return abs(a.r - b.r) > SAME_EPS || abs(a.i - b.i)  > SAME_EPS;
}

// // potential O(N log N) and O(1) memory if a tranformation at the beginning can be found
// vector<CPX> iterFFT2(vector<CPX> c){
//     vector<CPX> in(c);
//     vector<CPX> out(c.size());

//     for(int n=2; n<c.size(); n *= 2){
//         CPX wn = {cos(-2*PI/N), sin(-2*PI/N)};
//         for(int s = 0; s<c.size(); s+= n){
//             CPX wnk = {1, 0};
//             for(int k=0; k<n/2; k++){
//                 out[k] = in[k] + wnk*in[k+n/2];
//                 out[k+n/2] = in[k] - wnk*in[k+n/2];
//                 wnk *= wn;
//             }
//         }
//         std::swap(in, out);
//     }

//     return in;
// }

// O(N log N) and O(N) memory
vector<CPX> iterFFT(const vector<CPX>& c){
    const size_t N = c.size();

    vector<CPX> in(c);
    vector<CPX> out(N);

    for(size_t n=2; n<=N; n *= 2){
        const size_t alpha = N/n;
        const CPX w = {cos(-2*PI/n), sin(-2*PI/n)};

        for(size_t s=0; s<alpha; s++){
            CPX wi = {1, 0};
            for(size_t i=0; i < n/2; i++){
                out[s+i*alpha] = in[s+2*i*alpha] + wi * in[s+(2*i+1)*alpha];
                out[s+(i+n/2)*alpha] = in[s+2*i*alpha] - wi * in[s+(2*i+1)*alpha];
                wi *= w;
            }
        }
        in.swap(out);
    }

    return in;
}

// O(N log N) with useless heap allocations and stuff
vector<CPX> recurFFT(const vector<CPX>& c){
    const size_t n = c.size();
    if(n == 1) return c;

    vector<CPX> P;
    vector<CPX> O;
    {
        vector<CPX> evens(n/2);
        vector<CPX> odds(n/2);
        for(size_t i = 0; i<n; i++){
            if(i%2==0) evens[i/2] = c[i];
            else odds[(i-1)/2] = c[i];
        }
        P = recurFFT(evens);
        O = recurFFT(odds);
    }

    vector<CPX> out(n);

    const CPX w = {cos(-2*PI/n), sin(-2*PI/n)};
    CPX wk = {1, 0};

    for(size_t k=0; k<n/2; k++){
        out[k] = P[k] + wk*O[k];
        out[k + n/2] = P[k] - wk*O[k];

        wk *= w; 
    }

    return out;
}

// O(n^2)
vector<CPX> bruteFFT(const vector<CPX>& c){
    const size_t n = c.size();
    vector<CPX> out(n);
    for(size_t i=0; i<n; i++){
        // const CPX w = {cos(-2*PI*i/n), sin(-2*PI*i/n)};
        // CPX wk = {1, 0};
        for(size_t k=0; k<n; k++){
            const CPX wk = {cos(-2*PI*k*i/n), sin(-2*PI*k*i/n)};
            out[i] += wk * c[k];
            // wk *= w;
        }
    }

    return out;
}

// O(n^2)
vector<CPX> bruteIFFT(const vector<CPX>& c){
    const size_t n = c.size();
    const CPX oneovern = {(double)1.0/n, 0};
    vector<CPX> out(n);
    for(size_t i=0; i<n; i++){
        const CPX w = {cos(2*PI*i/n), sin(2*PI*i/n)};
        CPX wk = {1, 0};
        for(size_t k=0; k<n; k++){
            out[i] += oneovern * wk * c[k];
            wk *= w;
        }
    }

    return out;
}

template<typename T>
bool same(const vector<T>& a, const vector<T>& b){
    if(a.size() != b.size()) return false;
    for(size_t i=0; i<a.size(); i++) if(a[i] != b[i]) return false;
    return true;
}


#define BENCH(out, func, args...) {\
        const auto m_start = chrono::high_resolution_clock::now(); \
        out = func(args); \
        const auto m_end = chrono::high_resolution_clock::now(); \
        cout << #func << " took " << \
        chrono::duration_cast<chrono::microseconds>(m_end - m_start).count() \
        << " microseconds..." << endl; \
    }

#if PLOT_OUTPUT
void plot_complex(const vector<CPX>& in, const char* label){
    constexpr int wid = 3;
    constexpr int hei = 2;
    static int count = 1;
    vector<float> real(in.size());
    vector<float> imag(in.size());
    for(size_t i=0; i<in.size(); i++){
        real[i] = in[i].r;
        imag[i] = in[i].i;
    }
    assert(count <= wid*hei);
    plt::subplot(hei, wid, count);
    count++;
    plt::title(label);
    plt::named_plot("real component", real, "r");
    plt::named_plot("imaginary component", imag, "b");
    plt::legend();
}
#endif


int main()
{
    //fast_cin();

    cout << "generating input..." << endl;

    vector<CPX> in(SAMPLES_N);

    for(size_t i=0; i<in.size(); i++){
        auto& x = in[i];
        const double a = (double)1*i/in.size()*2*PI;
        const double a2 = (double)5*i/in.size()*2*PI;
        x.r = cos(a) + cos(a2);
        x.i = sin(a) + sin(a2);
        // x.i = 0;
    }

    cout << "running benchmarks..." << endl;

    vector<CPX> brute;
    BENCH(brute, bruteFFT, in)
    // {
    //     // Bench("bruteFFT");
    //     brute = bruteFFT(in);
    // }

    vector<CPX> outr;
    BENCH(outr, recurFFT, in)
    // {
    //     // Bench("recurFFT");
    //     outr = recurFFT(in);
    // }

    vector<CPX> outi;
    BENCH(outi, iterFFT, in)
    // {
    //     Bench("iterFFT");
    //     outi = iterFFT(in);
    // }

    vector<CPX> ifft_outr = bruteIFFT(outr);
    vector<CPX> ifft_outi = bruteIFFT(outi);

    #if PLOT_OUTPUT 
    cout << "plotting output..." << endl;
    
    plot_complex(in, "input");
    plot_complex(brute, "brute force");
    plot_complex(outr, "recursive approach");
    plot_complex(outi, "iterative approach");
    plot_complex(ifft_outr, "brute ifft(of recursive)");
    plot_complex(ifft_outi, "brute ifft(of iterative)");

    plt::show();
    #endif

    cout << "running tests: " << endl;
    assert(same(brute, outr));
    assert(same(brute ,outi));

    assert(same(in, ifft_outi));
    assert(same(in, ifft_outr));

    cout << "all tests were successfull..." << endl;
    
    return 0;
}