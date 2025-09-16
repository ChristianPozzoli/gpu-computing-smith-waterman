#include <iostream>
#include <string.h>

using namespace std;

inline int INDEX(int rows, int cols, int stride) { return rows * stride + cols; }

#define MATCH 2
#define MISMATCH -1
#define GAP -2

#define MAIN_SEQUENCE "-GGTTGACTA"
#define SEQUENCE "-TGTTACGG"

int main(int argc, char **argv) {
    string main_sequence = MAIN_SEQUENCE;
    string sequence = SEQUENCE;

    int main_len = main_sequence.length();
    int seq_len = sequence.length();
    int size = main_len * seq_len;
    int row = main_len;
    int col = seq_len;

    int* h = (int*) malloc(size * sizeof(int));

    for(int i = 0; i < size; ++i) {
        h[i] = 0;
    }

    int max_i = 0, max_j = 0, nmax = 0;

    for(int i = 1; i < row; ++i) {
        for (int j = 1; j < col; ++j) {
            bool match = main_sequence[i] == sequence[j];
            int d = h[INDEX(i - 1, j - 1, seq_len)] + (match ? MATCH : MISMATCH);
            int l = h[INDEX(i - 1, j, seq_len)] + GAP;
            int r = h[INDEX(i, j - 1, seq_len)] + GAP;
            int result = max(max(max(0, d), l), r);
            
            if(result > nmax) {
                nmax = result;
                max_i = i;
                max_j = j;
            }
            cout << result << " ";

            h[INDEX(i, j, seq_len)] = result;
        }
        cout << endl;
    }
    
	return(0);
}