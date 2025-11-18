#include <iostream>
#include <string.h>
#include <random>
#include "utils/common.h"
#include <cuda_runtime.h>

using namespace std;

#define INDEX(row, col, stride) ( (row) * (stride) + (col) )
#define INDEX_W(row, col, stride) ( (col) * (stride) + (row) )

#define THREADS 512
#define THREADS_W 64

#define MATCH 3
#define MISMATCH -3
#define GAP -2

enum dir : unsigned short { STOP, DIAG, UP, LEFT };
struct sw_mat {
    unsigned short value;
    dir dir;
};

std::string random_dna(size_t length, double gc_content = 0.5) {
    std::string seq;
    seq.reserve(length);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::discrete_distribution<> base_type({(1.0 - gc_content), gc_content});
    std::uniform_int_distribution<> coin(0, 1);

    for (size_t i = 0; i < length; ++i) {
        if (base_type(gen) == 0) {
            seq += (coin(gen) ? 'A' : 'T');
        } else {
            seq += (coin(gen) ? 'G' : 'C');
        }
    }
    return seq;
}

std::string mutate_dna(const std::string &original, double mutation_rate = 0.05) {
    std::string mutated = original;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::uniform_int_distribution<> pick(0, 3);
    const char bases[4] = {'A', 'C', 'G', 'T'};

    for (size_t i = 0; i < original.size(); ++i) {
        if (prob(gen) < mutation_rate) {
            char new_base;
            do {
                new_base = bases[pick(gen)];
            } while (new_base == original[i]);
            mutated[i] = new_base;
        }
    }
    return mutated;
}

void traceback(sw_mat* h, int size, string seq_1, string seq_2, int rows, int cols) {
    /* 
    for(int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            cout << h[INDEX(i, j, cols)].value << " ";
        }
        cout << endl;
    }
    cout << endl;
    */

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

    //cout << res_seq_1 << endl << res_seq_2 << endl << endl;

    int start = abs(i - j);
    
    if(i < j) cout << string(start, ' ');
    cout << seq_1.substr(0, i + 1) << "|" << res_seq_1 << "|" << seq_1.substr(i + 1 + res_seq_1.length() - seq_1_gap, rows) << endl;
    if(i > j) cout << string(start, ' ');
    cout << seq_2.substr(0, j + 1) << "|" << res_seq_2 << "|" << seq_2.substr(j + 1 + res_seq_2.length() - seq_2_gap, cols) << endl;
    cout << endl;
}

void traceback_warp(sw_mat* h, int size, string seq_1, string seq_2, int rows, int cols) {
    int max_i = 0, max_j = 0, nmax = 0;
    for(int i = 0; i < size; ++i) {
        int n = h[i].value;
        if(n > nmax) {
            nmax = n;
            max_i = i % rows;
            max_j = i / rows;
        }
    }

    int next = h[INDEX_W(max_i, max_j, rows)].value;
    int i = max_i, j = max_j, seq_1_gap = 0, seq_2_gap = 0;
    string res_seq_1 = "";
    string res_seq_2 = "";

    while (next != 0) {
        if(h[INDEX_W(i, j, rows)].dir == DIAG) {
            next = h[INDEX_W(i - 1, j - 1, rows)].value;
            res_seq_1 += seq_1[i--];
            res_seq_2 += seq_2[j--];
        }
        else if (h[INDEX_W(i, j, rows)].dir == UP) {
            next = h[INDEX_W(i - 1, j, rows)].value;
            res_seq_1 += seq_1[i--];
            res_seq_2 += '-';
            ++seq_2_gap;
        }
        else if (h[INDEX_W(i, j, rows)].dir == LEFT) {
            next = h[INDEX_W(i, j - 1, rows)].value;
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

    //cout << res_seq_1 << endl << res_seq_2 << endl << endl;

    /*
    int start = abs(i - j);
    
    if(i < j) cout << string(start, ' ');
    cout << seq_1.substr(0, i + 1) << "|" << res_seq_1 << "|" << seq_1.substr(i + 1 + res_seq_1.length() - seq_1_gap, rows) << endl;
    if(i > j) cout << string(start, ' ');
    cout << seq_2.substr(0, j + 1) << "|" << res_seq_2 << "|" << seq_2.substr(j + 1 + res_seq_2.length() - seq_2_gap, cols) << endl;
    cout << endl;
    */
}

void cpu_alignment(sw_mat* h, string seq_1, string seq_2, int seq_1_length, int seq_2_length) {
    for(int i = 1; i < seq_1_length; ++i) {
        for (int j = 1; j < seq_2_length; ++j) {
            bool match = seq_1[i] == seq_2[j];
            int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match * MATCH + (1 - match) * MISMATCH);
            int u = h[INDEX(i - 1, j, seq_2_length)].value + GAP;
            int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
            int result = max(max(0, d), max(u, l));
            
            sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
            h_current->value = result;

            if (result == 0)    h_current->dir = STOP;
            else if (result == d)   h_current->dir = DIAG;
            else if (result == u)   h_current->dir = UP;
            else if (result == l)   h_current->dir = LEFT;
        }
    }
}

__global__ void gpu_alignment(sw_mat* h, char* seq_1, char seq_2_char, int seq_1_length, int seq_2_length, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i == 0 || i >= seq_1_length) {
        return;
    }
    bool match = seq_1[i] == seq_2_char;
    int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match * MATCH + (1 - match) * MISMATCH);
    int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
    int u = 0;

    bool stop = false;
    for(int k = i - 1; k > 0 && !stop; --k) {
        if(seq_1[k] == seq_2_char) {
            u = h[INDEX(k - 1, j - 1, seq_2_length)].value + (i - k) * GAP + MATCH;
            stop = true;
        }
    }
    
    int result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
    h_current->value = result;

    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_alignment_smem(sw_mat* h, char* seq_1, char seq_2_char, int seq_1_length, int seq_2_length, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x + 1;
    if(i >= seq_1_length) {
        return;
    }

    __shared__ int smem[THREADS + 1];
    if(threadIdx.x == 0) {
        smem[ti - 1] = h[INDEX(i - 1, j - 1, seq_2_length)].value;
    }
    smem[ti] = h[INDEX(i, j - 1, seq_2_length)].value;

    __syncthreads();
    
    if(i == 0) {
        return;
    }
    
    bool match = seq_1[i] == seq_2_char;
    int d = smem[ti - 1] + (match * MATCH + (1 - match) * MISMATCH);
    int l = smem[ti] + GAP;
    int u = 0;
    
    bool stop = false;
    for(int k = i - 1; k > 0 && !stop; --k) {
        if(seq_1[k] == seq_2_char) {
            if(i - k > ti - 1) {
                u = h[INDEX(k - 1, j - 1, seq_2_length)].value + (i - k) * GAP + MATCH;
                //atomicAdd(&h[0].value, 1);
            } else {
                u = smem[ti - (i - k) - 1] + (i - k) * GAP + MATCH;
                //atomicAdd(&h[1].value, 1);
            }
            stop = true;
        }
    }
    
    int result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
    h_current->value = result;

    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_alignment_warp(sw_mat* h, char* seq_1, char seq_2_char, int seq_1_length, int seq_2_length, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i == 0 || i >= seq_1_length) {
        return;
    }

    int left_value = h[INDEX_W(i, j - 1, seq_1_length)].value;
    int diag_value = __shfl_up_sync(0xFFFFFFFF, left_value, 1);
    if (threadIdx.x % 32 == 0) diag_value = h[INDEX_W(i - 1, j - 1, seq_1_length)].value;

    bool match = seq_1[i] == seq_2_char;
    int d = diag_value + (match * MATCH + (1 - match) * MISMATCH);
    int l = left_value + GAP;
    int u = 0;
    
    bool stop = false;
    for(int k = i - 1; k > 0 && !stop; --k) {
        if(seq_1[k] == seq_2_char) {
            u = h[INDEX_W(k - 1, j - 1, seq_1_length)].value + (i - k) * GAP + MATCH;
            stop = true;
        }
    }
    
    int result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX_W(i, j, seq_1_length)];
    h_current->value = result;
    
    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_alignment_warp_smem(sw_mat* h, char* seq_1, char seq_2_char, int seq_1_length, int seq_2_length, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x + 1;
    if(i >= seq_1_length) {
        return;
    }

    __shared__ int smem[THREADS + 1];
    if(threadIdx.x == 0) {
        smem[0] = h[INDEX_W(i - 1, j - 1, seq_1_length)].value;
    }
    smem[ti] = h[INDEX_W(i, j - 1, seq_1_length)].value;

    __syncthreads();
    
    if(i == 0) {
        return;
    }
    
    bool match = seq_1[i] == seq_2_char;
    int d = smem[ti - 1] + (match * MATCH + (1 - match) * MISMATCH);
    int l = smem[ti] + GAP;
    int u = 0;
    
    bool stop = false;
    for(int k = i - 1; k > 0 && !stop; --k) {
        if(seq_1[k] == seq_2_char) {
            if(i - k > ti - 1) {
                u = h[INDEX_W(k - 1, j - 1, seq_1_length)].value + (i - k) * GAP + MATCH;
            } else {
                u = smem[ti - (i - k) - 1] + (i - k) * GAP + MATCH;
            }
            stop = true;
        }
    }
    
    int result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX_W(i, j, seq_1_length)];
    h_current->value = result;

    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

void gpu_alignment_loop(sw_mat* h, char* seq_1, string seq_2, int seq_1_length, int seq_2_length, int grid, int block) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment<<<grid, block>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
    }
}

void gpu_alignment_smem_loop(sw_mat* h, char* seq_1, string seq_2, int seq_1_length, int seq_2_length, int grid, int block) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment_smem<<<grid, block>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
    }
}

void gpu_alignment_warp_loop(sw_mat* h, char* seq_1, string seq_2, int seq_1_length, int seq_2_length, int grid, int block) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment_warp<<<grid, block>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
    }
}

void gpu_alignment_warp_smem_loop(sw_mat* h, char* seq_1, string seq_2, int seq_1_length, int seq_2_length, int grid, int block) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment_warp_smem<<<grid, block>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
    }
}

int main(int argc, char **argv) {
    string main_sequence = random_dna(10000);
    string sequence = mutate_dna(main_sequence);

    int rows = main_sequence.length();
    int cols = sequence.length();
    int size = rows * cols;
    int nBytes = size * sizeof(sw_mat);
    
    cout << "Size of sw_mat: " << sizeof(sw_mat) << endl;

    // Memory
    sw_mat* h = (sw_mat*) malloc(nBytes); // Result CPU
    sw_mat* h1 = (sw_mat*) malloc(nBytes); // Result GPU
    
    sw_mat *d_h1; // Device result
    char *d_main_seq, *d_seq; // Device sequence
    
    for(int i = 0; i < size; ++i) {
        h[i].value = 0;
        h[i].dir = STOP;
    }
    
    CHECK(cudaMalloc((void** )&d_main_seq, rows * sizeof(char)));
    CHECK(cudaMemcpy(d_main_seq, main_sequence.c_str(), rows * sizeof(char), cudaMemcpyHostToDevice));
    
    CHECK(cudaMalloc((void** )&d_seq, cols * sizeof(char)));
    CHECK(cudaMemcpy(d_seq, sequence.c_str(), cols * sizeof(char), cudaMemcpyHostToDevice));
    
    CHECK(cudaMalloc((void** )&d_h1, nBytes));
    CHECK(cudaMemcpy(d_h1, h, nBytes, cudaMemcpyHostToDevice));
    
    // CPU ====================================================================
    
    double start = seconds();
    cpu_alignment(h, main_sequence, sequence, rows, cols);
    double stopCPU = seconds() - start;
    
    // GPU ====================================================================
    
    int blocks_n = (rows + THREADS_W - 1) / THREADS_W;
    
    dim3 grid(blocks_n, 1, 1);
    dim3 block(THREADS_W, 1, 1);
    
    start = seconds();
    gpu_alignment_loop(d_h1, d_main_seq, sequence, rows, cols, grid.x, block.x);
    CHECK(cudaDeviceSynchronize());
    double stopGPU_noSMEM = seconds() - start;
    
    CHECK(cudaMemcpy(h1, d_h1, nBytes, cudaMemcpyDeviceToHost));
    bool err = false;
    for(int i = 0; i < size && !err; ++i) {
        err = h[i].value != h1[i].value || h[i].dir != h1[i].dir;
    }
    if(err) {
        cout << "ERROR: results do not match! [PARALLEL]" << endl;
        return(1);
    }
    
    // SMEM ===================================================================
    
    start = seconds();
    gpu_alignment_smem_loop(d_h1, d_main_seq, sequence, rows, cols, grid.x, block.x);
    CHECK(cudaDeviceSynchronize());
    double stopGPU_SMEM = seconds() - start;
    
    CHECK(cudaMemcpy(h1, d_h1, nBytes, cudaMemcpyDeviceToHost));
    err = false;
    for(int i = 0; i < size && !err; ++i) {
        err = h[i].value != h1[i].value || h[i].dir != h1[i].dir;
    }
    if(err) {
        cout << "ERROR: results do not match! [SMEM]" << endl;
        return(1);
    }
    
    // WARP ===================================================================
    
    {
        sw_mat* temp_h = (sw_mat*) malloc(nBytes);
        
        for(int i = 0; i < size; ++i) {
            temp_h[i].value = 0;
            temp_h[i].dir = STOP;
        }
        CHECK(cudaMemcpy(d_h1, temp_h, nBytes, cudaMemcpyHostToDevice));
        free(temp_h);
    }
    
    start = seconds();
    gpu_alignment_warp_loop(d_h1, d_main_seq, sequence, rows, cols, grid.x, block.x);
    CHECK(cudaDeviceSynchronize());
    double stopGPU_warp = seconds() - start;
    
    CHECK(cudaMemcpy(h1, d_h1, nBytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i < rows && !err; ++i) {
        for(int j = 0; j < cols && !err; ++j) {
            err = h[INDEX(i, j, cols)].value != h1[INDEX_W(i, j, rows)].value;
            err |= h[INDEX(i, j, cols)].dir != h1[INDEX_W(i, j, rows)].dir;
        }
    }
    if(err) {
        cout << "ERROR: results do not match! [WARP LOOP]" << endl;
        return(1);
    }
    
    // WARP SMEM ==============================================================
    
    start = seconds();
    gpu_alignment_warp_smem_loop(d_h1, d_main_seq, sequence, rows, cols, grid.x, block.x);
    CHECK(cudaDeviceSynchronize());
    double stopGPU_warp_smem = seconds() - start;
    
    CHECK(cudaMemcpy(h1, d_h1, nBytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i < rows && !err; ++i) {
        for(int j = 0; j < cols && !err; ++j) {
            err = h[INDEX(i, j, cols)].value != h1[INDEX_W(i, j, rows)].value;
            err |= h[INDEX(i, j, cols)].dir != h1[INDEX_W(i, j, rows)].dir;
        }
    }
    if(err) {
        cout << "ERROR: results do not match! [WARP SMEM LOOP]" << endl;
        return(1);
    }

    // Results ================================================================
    
    cout << "|=================RESULTS=================" << endl;
    cout << "| CPU time:\t\t\t" << stopCPU << "s" << endl;
    cout << "| GPU time (parallel):\t\t" << stopGPU_noSMEM << "s" << endl;
    cout << "| GPU time (SMEM):\t\t" << stopGPU_SMEM << "s" << endl;
    cout << "| GPU time (warp):\t\t" << stopGPU_warp << "s" << endl;
    cout << "| GPU time (warp SMEM):\t\t" << stopGPU_warp_smem << "s" << endl;
    cout << "|================SPEED UP=================" << endl;
    cout << "| Over CPU (parallel):\t\t" << stopCPU / stopGPU_noSMEM << endl;
    cout << "| Over CPU (SMEM):\t\t" << stopCPU / stopGPU_SMEM << endl;
    cout << "| Over CPU (warp):\t\t" << stopCPU / stopGPU_warp << endl;
    cout << "| Over CPU (warp SMEM):\t\t" << stopCPU / stopGPU_warp_smem << endl;
    cout << "|=========================================" << endl;
    cout << "| Over parallel (SMEM):\t\t" << stopGPU_noSMEM / stopGPU_SMEM << endl;
    cout << "| Over parallel (warp):\t\t" << stopGPU_noSMEM / stopGPU_warp << endl;
    cout << "| Over parallel (warp SMEM):\t" << stopGPU_noSMEM / stopGPU_warp_smem << endl;
    cout << "|=========================================" << endl;
    cout << "| Over SMEM (warp):\t\t" << stopGPU_SMEM / stopGPU_warp << endl;
    cout << "| Over SMEM (warp SMEM):\t" << stopGPU_SMEM / stopGPU_warp_smem << endl;
    cout << "|=========================================" << endl;
    cout << "| Over warp (warp SMEM):\t" << stopGPU_warp / stopGPU_warp_smem << endl;
    cout << "|=========================================" << endl;
    
	return(0);
}