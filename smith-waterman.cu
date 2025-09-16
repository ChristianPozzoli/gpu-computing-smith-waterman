#include <iostream>
#include <string.h>

using namespace std;

inline int INDEX(int rows, int cols, int stride) { return rows * stride + cols; }

#define MATCH 3
#define MISMATCH -3
#define GAP -2

#define MAIN_SEQUENCE "-GGTTGACTA"
#define SEQUENCE "-TGTTACGG"

enum dir { STOP, DIAG, UP, LEFT };

int main(int argc, char **argv) {
    string main_sequence = MAIN_SEQUENCE;
    string sequence = SEQUENCE;

    int main_len = main_sequence.length();
    int seq_len = sequence.length();
    int size = main_len * seq_len;
    int rows = main_len;
    int cols = seq_len;

    int* h = (int*) malloc(size * sizeof(int));
    dir* m = (dir*) malloc(size * sizeof(dir));

    for(int i = 0; i < size; ++i) {
        h[i] = 0;
        m[i] = STOP;
    }

    int max_i = 0, max_j = 0, nmax = 0;

    for(int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            bool match = main_sequence[i] == sequence[j];
            int d = h[INDEX(i - 1, j - 1, cols)] + (match ? MATCH : MISMATCH);
            int u = h[INDEX(i - 1, j, cols)] + GAP;
            int l = h[INDEX(i, j - 1, cols)] + GAP;
            int result = max({0, d, u, l});
            
            if(result > nmax) {
                nmax = result;
                max_i = i;
                max_j = j;
            }
            cout << result << " ";

            h[INDEX(i, j, cols)] = result;

            if (result == 0)    m[INDEX(i, j, cols)] = STOP;
            else if (result == d)   m[INDEX(i, j, cols)] = DIAG;
            else if (result == u)   m[INDEX(i, j, cols)] = UP;
            else if (result == l)   m[INDEX(i, j, cols)] = LEFT;
        }
        cout << endl;
    }

    cout << endl;

    int next = h[INDEX(max_i, max_j, cols)];
    int i = max_i, j = max_j;
    string seq_1 = "";//string(1, main_sequence[i]);
    string seq_2 = "";//string(1, sequence[j]);

    while (next != 0) {
        int d = h[INDEX(i - 1, j - 1, cols)];
        int u = h[INDEX(i - 1, j, cols)];
        int l = h[INDEX(i, j - 1, cols)];

        if(m[INDEX(i, j, cols)] == DIAG) {
            next = d;
            seq_1 += main_sequence[i];
            seq_2 += sequence[j];
            i -= 1;
            j -= 1;
        }
        else if (m[INDEX(i, j, cols)] == UP) {
            next = u;
            seq_1 += main_sequence[i];
            seq_2 += '-';
            i -= 1;
        }
        else if (m[INDEX(i, j, cols)] == LEFT) {
            next = l;
            seq_1 += '-';
            seq_2 += sequence[j];
            j -= 1;
        }
        else {
            next = 0;
        }
    }

    int rsize = seq_1.length();
    for (int k = 0; k < rsize / 2; ++k) {
        swap(seq_1[k], seq_1[rsize - 1 - k]);
        swap(seq_2[k], seq_2[rsize - 1 - k]);
    }

    cout << seq_1 << endl << seq_2 << endl;
    
	return(0);
}