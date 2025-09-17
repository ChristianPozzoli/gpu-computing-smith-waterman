#include <iostream>
#include <string.h>

using namespace std;

#define INDEX(row, col, stride) ( (row) * (stride) + (col) )

#define MATCH 3
#define MISMATCH -3
#define GAP -2

#define MAIN_SEQUENCE "+GACT"
#define SEQUENCE "-GCT"

enum dir { STOP, DIAG, UP, LEFT };
struct sw_mat {
    int value;
    dir dir;
};

void traceback(sw_mat* h, int size, int cols, string seq_1, string seq_2) {
    int seq_1_len = seq_1.length();
    int seq_2_len = seq_2.length();

    for(int i = 1; i < seq_1_len; ++i) {
        for (int j = 1; j < seq_2_len; ++j) {
            cout << h[INDEX(i, j, seq_2_len)].value << " ";
        }
        cout << endl;
    }
    cout << endl;
    
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
    string res_seq_1 = "";
    string res_seq_2 = "";

    while (next != 0) {
        if(h[INDEX(i, j, cols)].dir == DIAG) {
            next = h[INDEX(i - 1, j - 1, cols)].value;
            res_seq_1 += seq_1[i--];
            res_seq_2 += seq_2[j--];
        }
        else if (h[INDEX(i, j, cols)].dir == UP) {
            next = h[INDEX(i - 1, j, cols)].value;
            res_seq_1 += seq_1[i--];
            res_seq_2 += '-';
            ++seq_2_gap;
        }
        else if (h[INDEX(i, j, cols)].dir == LEFT) {
            next = h[INDEX(i, j - 1, cols)].value;
            res_seq_1 += '-';
            res_seq_2 += seq_2[j--];
            ++seq_1_gap;
        }
        else {
            next = 0;
        }
    }

    int rsize = res_seq_1.length();
    for (int k = 0; k < rsize / 2; ++k) {
        swap(res_seq_1[k], res_seq_1[rsize - 1 - k]);
        swap(res_seq_2[k], res_seq_2[rsize - 1 - k]);
    }

    cout << res_seq_1 << endl << res_seq_2 << endl << endl;

    int start = abs(i - j);

    if(i < j) cout << string(start, ' ');
    cout << seq_1.substr(0, i + 1) << "|" << res_seq_1 << "|" << seq_1.substr(i + 1 + res_seq_1.length() - seq_1_gap, seq_1_len) << endl;
    if(i > j) cout << string(start, ' ');
    cout << seq_2.substr(0, j + 1) << "|" << res_seq_2 << "|" << seq_2.substr(j + 1 + res_seq_2.length() - seq_2_gap, seq_2_len) << endl;
    cout << endl;
}

void cpu_alignment(sw_mat* h, string seq_1, string seq_2, int seq_1_length, int seq_2_length) {
    for(int i = 1; i < seq_1_length; ++i) {
        for (int j = 1; j < seq_2_length; ++j) {
            bool match = seq_1[i] == seq_2[j];
            int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match ? MATCH : MISMATCH);
            int u = h[INDEX(i - 1, j, seq_2_length)].value + GAP;
            int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
            int result = max({0, d, u, l});
            
            sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
            h_current->value = result;

            if (result == 0)    h_current->dir = STOP;
            else if (result == d)   h_current->dir = DIAG;
            else if (result == u)   h_current->dir = UP;
            else if (result == l)   h_current->dir = LEFT;
        }
    }
}

void cpu_column_alignment(sw_mat* h, const char* seq_1, const char* seq_2, int seq_1_length, int seq_2_length) {
    for (int j = 1; j < seq_2_length; ++j) {
        for(int i = 1; i < seq_1_length; ++i) {
            bool match = seq_1[i] == seq_2[j];
            int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match ? MATCH : MISMATCH);
            int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
            int u = 0;
            
            bool stop = false;
            for(int k = i - 1; k > 0 && !stop; --k) {
                if(seq_1[k] == seq_2[j]) {
                    u = h[INDEX(k - 1, j - 1, seq_2_length)].value + (i - k) * GAP + MATCH;
                    stop = true;
                }
            }
            
            int result = max({0, d, u, l});
            
            sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
            h_current->value = result;
            
            if (result == 0)    h_current->dir = STOP;
            else if (result == d)   h_current->dir = DIAG;
            else if (result == u)   h_current->dir = UP;
            else if (result == l)   h_current->dir = LEFT;
        }
    }
    cout << endl;
}

__global__ void gpu_alignment(sw_mat* h, char* seq_1, char* seq_2, int seq_1_length, int seq_2_length) {
    int i = threadIdx.x;
    for (int j = 1; j < seq_2_length; ++j) {
        bool match = seq_1[i] == seq_2[j];
        int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match ? MATCH : MISMATCH);
        int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
        int u = 0;

        bool stop = false;
        for(int k = i - 1; k > 0 && !stop; --k) {
            if(seq_1[k] == seq_2[j]) {
                u = h[INDEX(k - 1, j - 1, seq_2_length)].value + (i - k) * GAP + MATCH;
                stop = true;
            }
        }
        
        int result = max(0, d);
        result = max(result, u);
        result = max(result, l);
        
        sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
        h_current->value = result;

        if (result == 0)    h_current->dir = STOP;
        else if (result == d)   h_current->dir = DIAG;
        else if (result == u)   h_current->dir = UP;
        else if (result == l)   h_current->dir = LEFT;

        __syncthreads();
    }
}

int main(int argc, char **argv) {
    string main_sequence = MAIN_SEQUENCE;
    string sequence = SEQUENCE;

    int main_len = main_sequence.length();
    int seq_len = sequence.length();
    int size = main_len * seq_len;
    int nBytes = size * sizeof(sw_mat);
    int rows = main_len;
    int cols = seq_len;

    sw_mat* h = (sw_mat*) malloc(nBytes);
    sw_mat* hc = (sw_mat*) malloc(nBytes);
    sw_mat* h1 = (sw_mat*) malloc(nBytes);
    
    sw_mat* d_h;
    char *d_main_seq, *d_seq;
    
    for(int i = 0; i < size; ++i) {
        h[i].value = 0;
        h[i].dir = STOP;
        hc[i].value = 0;
        hc[i].dir = STOP;
    }
    
    cudaMalloc((void** )&d_h, nBytes);
    cudaMalloc((void** )&d_main_seq, main_len * sizeof(char));
    cudaMalloc((void** )&d_seq, seq_len * sizeof(char));
    
    cudaMemcpy(d_h, h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_main_seq, main_sequence.c_str(), main_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq, sequence.c_str(), seq_len * sizeof(char), cudaMemcpyHostToDevice);

    // CPU
    cpu_alignment(h, main_sequence, sequence, main_len, seq_len);
    
    cpu_column_alignment(hc, main_sequence.c_str(), sequence.c_str(), main_len, seq_len);
                
    // GPU
    
    dim3 grid(1, 1, 1);
    dim3 block(main_len, 1, 1);
    
    gpu_alignment<<<grid, block>>>(d_h, d_main_seq, d_seq, main_len, seq_len);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h1, d_h, nBytes, cudaMemcpyDeviceToHost);

    // Traceback
    traceback(h, size, seq_len, main_sequence, sequence);
    traceback(hc, size, seq_len, main_sequence, sequence);
    traceback(h1, size, seq_len, main_sequence, sequence);
    
	return(0);
}