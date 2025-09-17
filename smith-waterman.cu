#include <iostream>
#include <string.h>

using namespace std;

inline int INDEX(int row, int col, int stride) { return row * stride + col; }

#define MATCH 3
#define MISMATCH -3
#define GAP -2

#define MAIN_SEQUENCE "-CACTATGC"
#define SEQUENCE "-AGCCTCA"

enum dir { STOP, DIAG, UP, LEFT };
struct sw_mat {
    int value;
    dir dir;
};

void cpu_alignment(sw_mat* h, string seq_1, string seq_2, int seq_1_length, int seq_2_length) {
    for(int i = 1; i < seq_1_length; ++i) {
        for (int j = 1; j < seq_2_length; ++j) {
            bool match = seq_1[i] == seq_2[j];
            int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match ? MATCH : MISMATCH);
            int u = h[INDEX(i - 1, j, seq_2_length)].value + GAP;
            int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
            int result = max({0, d, u, l});
            
            //cout << result << " ";

            sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
            h_current->value = result;

            if (result == 0)    h_current->dir = STOP;
            else if (result == d)   h_current->dir = DIAG;
            else if (result == u)   h_current->dir = UP;
            else if (result == l)   h_current->dir = LEFT;
        }
        //cout << endl;
    }
}

int main(int argc, char **argv) {
    string main_sequence = MAIN_SEQUENCE;
    string sequence = SEQUENCE;

    int main_len = main_sequence.length();
    int seq_len = sequence.length();
    int size = main_len * seq_len;
    int rows = main_len;
    int cols = seq_len;

    sw_mat* h = (sw_mat*) malloc(size * sizeof(sw_mat));

    for(int i = 0; i < size; ++i) {
        h[i].value = 0;
        h[i].dir = STOP;
    }

    cpu_alignment(h, main_sequence, sequence, main_len, seq_len);
    
    int max_i = 0, max_j = 0, nmax = 0;
    for(int i = 0; i < size; ++i) {
        int n = h[i].value;

        if(n > nmax) {
            nmax = n;
            max_i = i / cols;
            max_j = i % cols;
        }
    }

    int next = h[INDEX(max_i, max_j, cols)].value;
    int i = max_i, j = max_j, seq_1_gap = 0, seq_2_gap = 0;
    string seq_1 = "";
    string seq_2 = "";

    while (next != 0) {
        if(h[INDEX(i, j, cols)].dir == DIAG) {
            next = h[INDEX(i - 1, j - 1, cols)].value;
            seq_1 += main_sequence[i--];
            seq_2 += sequence[j--];
        }
        else if (h[INDEX(i, j, cols)].dir == UP) {
            next = h[INDEX(i - 1, j, cols)].value;
            seq_1 += main_sequence[i--];
            seq_2 += '-';
            ++seq_2_gap;
        }
        else if (h[INDEX(i, j, cols)].dir == LEFT) {
            next = h[INDEX(i, j - 1, cols)].value;
            seq_1 += '-';
            seq_2 += sequence[j--];
            ++seq_1_gap;
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

    cout << seq_1 << endl << seq_2 << endl << endl;

    int start = abs(i - j);

    if(i < j) cout << string(start, ' ');
    cout << main_sequence.substr(0, i + 1) << "|" << seq_1 << "|" << main_sequence.substr(i + 1 + seq_1.length() - seq_1_gap, main_len) << endl;
    if(i > j) cout << string(start, ' ');
    cout << sequence.substr(0, j + 1) << "|" << seq_2 << "|" << sequence.substr(j + 1 + seq_2.length() - seq_2_gap, seq_len) << endl;
    
	return(0);
}