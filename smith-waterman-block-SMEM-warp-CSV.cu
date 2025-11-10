#include <iostream>
#include <string.h>
#include <random>
#include "utils/common.h"
#include <cuda_runtime.h>

using namespace std;

#define INDEX(row, col, stride) ( (size_t)(row) * (size_t)(stride) + (size_t)(col) )
#define INDEX_W(row, col, stride) ( (size_t)(col) * (size_t)(stride) + (size_t)(row) )

#define FILENAME "results"
#define TEST_REPETITION 10

#define MATCH 3
#define MISMATCH -3
#define GAP -2

enum dir : uint8_t { STOP, DIAG, UP, LEFT };
struct __align__(2) sw_mat {
    uint16_t value;
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

void cpu_alignment(sw_mat* h, string seq_1, string seq_2, uint16_t seq_1_length, uint16_t seq_2_length) {
    for(uint16_t i = 1; i < seq_1_length; ++i) {
        for (uint16_t j = 1; j < seq_2_length; ++j) {
            bool match = seq_1[i] == seq_2[j];
            int d = h[INDEX(i - 1, j - 1, seq_2_length)].value + (match * MATCH + (1 - match) * MISMATCH);
            int u = h[INDEX(i - 1, j, seq_2_length)].value + GAP;
            int l = h[INDEX(i, j - 1, seq_2_length)].value + GAP;
            uint16_t result = max(max(0, d), max(u, l));
            
            sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
            h_current->value = result;

            if (result == 0)    h_current->dir = STOP;
            else if (result == d)   h_current->dir = DIAG;
            else if (result == u)   h_current->dir = UP;
            else if (result == l)   h_current->dir = LEFT;
        }
    }
}

extern __shared__ int smem[];

__global__ void gpu_alignment(sw_mat* h, char* seq_1, char seq_2_char, uint16_t seq_1_length, uint16_t seq_2_length, int j) {
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
    
    uint16_t result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
    h_current->value = result;

    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_alignment_smem(sw_mat* h, char* seq_1, char seq_2_char, uint16_t seq_1_length, uint16_t seq_2_length, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x + 1;
    if(i >= seq_1_length) {
        return;
    }

    
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
    
    uint16_t result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX(i, j, seq_2_length)];
    h_current->value = result;

    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_alignment_warp(sw_mat* h, char* seq_1, char seq_2_char, uint16_t seq_1_length, uint16_t seq_2_length, int j) {
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
    
    uint16_t result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX_W(i, j, seq_1_length)];
    h_current->value = result;
    
    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_alignment_warp_smem(sw_mat* h, char* seq_1, char seq_2_char, uint16_t seq_1_length, uint16_t seq_2_length, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = threadIdx.x + 1;
    if(i >= seq_1_length) {
        return;
    }

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
    
    uint16_t result = max(max(0, d), max(u, l));
    
    sw_mat* h_current = &h[INDEX_W(i, j, seq_1_length)];
    h_current->value = result;

    if (result == 0)    h_current->dir = STOP;
    else if (result == d)   h_current->dir = DIAG;
    else if (result == u)   h_current->dir = UP;
    else if (result == l)   h_current->dir = LEFT;
}

__global__ void gpu_dummy() { return; }

__global__ void gpu_alignment_loop(sw_mat* h, char* seq_1, char* seq_2, uint16_t seq_1_length, uint16_t seq_2_length, uint16_t grid, uint16_t block, int smem) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment<<<grid, block>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
    }
    __syncthreads();
}

__global__ void gpu_alignment_smem_loop(sw_mat* h, char* seq_1, char* seq_2, uint16_t seq_1_length, uint16_t seq_2_length, uint16_t grid, uint16_t block, int smem) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment_smem<<<grid, block, smem>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
        __syncthreads();
    }
}

__global__ void gpu_alignment_warp_loop(sw_mat* h, char* seq_1, char* seq_2, uint16_t seq_1_length, uint16_t seq_2_length, uint16_t grid, uint16_t block, int smem) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment_warp<<<grid, block>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
        __syncthreads();
    }
}

__global__ void gpu_alignment_warp_smem_loop(sw_mat* h, char* seq_1, char* seq_2, uint16_t seq_1_length, uint16_t seq_2_length, uint16_t grid, uint16_t block, int smem) {
    for (int j = 1; j < seq_2_length; ++j) {
        gpu_alignment_warp_smem<<<grid, block, smem>>>(h, seq_1, seq_2[j], seq_1_length, seq_2_length, j);
        __syncthreads();
    }
}

bool checkError(const std::string& algorithm, int size, const sw_mat *check, sw_mat *test) {
    bool err = false;
    for(int i = 0; i < size && !err; ++i) {
        err = check[i].value != test[i].value || check[i].dir != test[i].dir;
    }
    if(err) {
        cout << "ERROR: results do not match! [" << algorithm << "]" << endl;
    }
    return err;
}

bool checkErrorWarp(const std::string& algorithm, uint16_t rows, uint16_t cols, const sw_mat *check, sw_mat *test) {
    bool err = false;
    for(int i = 0; i < rows && !err; ++i) {
        for(int j = 0; j < cols && !err; ++j) {
            err = check[INDEX(i, j, cols)].value != test[INDEX_W(i, j, rows)].value;
            err |= check[INDEX(i, j, cols)].dir != test[INDEX_W(i, j, rows)].dir;
        }
    }
    if(err) {
        cout << "ERROR: results do not match! [" << algorithm << "]" << endl;
    }
    return err;
}

string makeCSV() {
    time_t now = time(nullptr);
    char filename[32];
    strftime(filename, sizeof(filename), FILENAME "_%d%m%Y_%H%M%S.csv", localtime(&now));
    std::ifstream infile(filename);
    bool fileExists = infile.good();
    infile.close();

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filename << std::endl;
        return "";
    }

    if (!fileExists) {
        file << "algorithm,input_size,block_size,time_ms\n";
    }

    return filename;
}

void appendToCSV(const std::string& filename,
                 const std::string& algorithm,
                 int inputSize,
                 int blockSize,
                 double time) 
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filename << std::endl;
        return;
    }

    file << algorithm << ","
        << inputSize << ","
        << blockSize << ","
        << time << "\n";

    file.close();
}

void appendStringToCSV(const std::string& filename, const std::string& s) 
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filename << std::endl;
        return;
    }

    file << s;

    file.close();
}

template <typename KernelLoop, typename... Args>

void runKernel(
    KernelLoop kernel,
    const string name,
    sw_mat* h,
    sw_mat* h1,
    sw_mat* d_h1,
    int nBytes,
    uint16_t rows,
    uint16_t cols,
    bool warpCheckError,
    const string filename,
    uint16_t input_size,
    uint16_t block_size,
    Args... args
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<1, 1>>>(d_h1, args...);
    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    
    CHECK(cudaMemcpy(h1, d_h1, nBytes, cudaMemcpyDeviceToHost));

    if((warpCheckError && checkErrorWarp(name, rows, cols, h, h1)) ||
    (!warpCheckError && checkError(name, rows * cols, h, h1)))
    {
        appendToCSV(filename, name + "_error", input_size, block_size, -1);
    }
    else
    {
        float stopGPU;
        cudaEventElapsedTime(&stopGPU, start, stop);
        appendToCSV(filename, name, input_size, block_size, (stopGPU / 1000));
    }
}

void runTests(const string& filename)
{
    std::vector<uint16_t> inputSizes = {20000};//, 5000, 10000, 15000, 18000, 20000};
    std::vector<uint16_t> blockSizes = {32};//{32, 64, 128, 256, 512, 1024};
    
    for (uint16_t n : inputSizes)
    {
        cout << "Input size: " << n << endl;
        string main_sequence = random_dna(n);
        string sequence = mutate_dna(main_sequence);
        
        uint16_t rows = main_sequence.length();
        uint16_t cols = sequence.length();
        int size = rows * cols;
        int nBytes = size * sizeof(sw_mat);
        
        sw_mat* h = (sw_mat*) malloc(nBytes); // Result CPU
        sw_mat* h1 = (sw_mat*) malloc(nBytes); // Result GPU
        sw_mat* empty_h = (sw_mat*) malloc(nBytes);
        
        sw_mat *d_h1; // Device result
        char *d_main_seq, *d_seq; // Device sequence
        
        for(int i = 0; i < size; ++i)
        {
            h[i].value = 0;
            h[i].dir = STOP;
            
            empty_h[i].value = 0;
            empty_h[i].dir = STOP;
        }
        
        CHECK(cudaMalloc((void** )&d_main_seq, rows * sizeof(char)));
        CHECK(cudaMemcpy(d_main_seq, main_sequence.c_str(), rows * sizeof(char), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc((void** )&d_seq, cols * sizeof(char)));
        CHECK(cudaMemcpy(d_seq, sequence.c_str(), cols * sizeof(char), cudaMemcpyHostToDevice));
        
        CHECK(cudaMalloc((void** )&d_h1, nBytes));
        CHECK(cudaMemcpy(d_h1, h, nBytes, cudaMemcpyHostToDevice));
        
        cout << "\tCPU [";
        for(uint8_t i = 0; i < 1/*TEST_REPETITION*/; ++i)
        {
            double start = seconds();
            cpu_alignment(h, main_sequence, sequence, rows, cols);
            double stopCPU = seconds() - start;
            appendToCSV(filename, "cpu", n, -1, stopCPU);
            cout << ".";
        }
        cout << "]" << endl;
        
        for (int b : blockSizes)
        {
            cout << "\tBlock size: " << b << endl;

            // SETUP ==================================================================
            
            CHECK(cudaMemcpy(d_h1, empty_h, nBytes, cudaMemcpyHostToDevice)); //Reset
            
            uint16_t blocks_n = (rows + b - 1) / b;
            
            dim3 grid(blocks_n, 1, 1);
            dim3 block(b, 1, 1);
            
            int smem_size = sizeof(int) * (b + 1);
            
            for(uint8_t i = 0; i < TEST_REPETITION; ++i)
            {
                // PARALLEL ===============================================================

                cout << "\t\tPARALLEL [" << (int)i + 1 << "/" << TEST_REPETITION << "]" << endl;
                runKernel(
                    gpu_alignment_loop, "parallel", h, h1, d_h1, nBytes, rows, cols, false,
                    filename, n, b,
                    d_main_seq, d_seq, rows, cols, grid.x, block.x, 0
                );

                bool err = false;
                int l;
                for(l = 0; l < size && !err; ++l) {
                    err = h[l].value != h1[l].value || h[l].dir != h1[l].dir;
                }
                if(err) {
                    std::ofstream file("debug.txt", std::ios::app);
                    if (!file.is_open()) {
                        std::cerr << "Errore: impossibile aprire " << filename << std::endl;
                        return;
                    }
                    
                    file << l << "\n";
                    for(int i = l-10; i < l; ++i)
                    {
                        file << h[i].value << ",";
                    }
                    file << "\n";
                    for(int i = l-10; i < l; ++i)
                    {
                        file << h1[i].value << ",";
                    }
                    int i_h = (l - 1) / cols;
                    int j_h = (l - 1) % cols;
                    file << "\n" << h[INDEX(i_h - 1, j_h - 1, cols)].value << "\n";
                    file << h[INDEX(i_h, j_h - 1, cols)].value << "," << h[INDEX(i_h, j_h, cols)].value;
                    file << "\n" << h1[INDEX(i_h - 1, j_h - 1, cols)].value << "\n";
                    file << h1[INDEX(i_h, j_h - 1, cols)].value << "," << h1[INDEX(i_h, j_h, cols)].value;
                    file << "\n\n";
                    
                    file.close();
                }
                
                // SMEM ===================================================================
                
                cout << "\t\tSMEM [" << (int)i + 1 << "/" << TEST_REPETITION << "]" << endl;
                runKernel(
                    gpu_alignment_smem_loop, "smem", h, h1, d_h1, nBytes, rows, cols, false,
                    filename, n, b,
                    d_main_seq, d_seq, rows, cols, grid.x, block.x, smem_size
                );
                
                
                // ========================================================================
                
                CHECK(cudaMemcpy(d_h1, empty_h, nBytes, cudaMemcpyHostToDevice)); //Reset
                
                // WARP ===================================================================
                
                cout << "\t\tWARP [" << (int)i + 1 << "/" << TEST_REPETITION << "]" << endl;
                runKernel(
                    gpu_alignment_warp_loop, "warp", h, h1, d_h1, nBytes, rows, cols, true,
                    filename, n, b,
                    d_main_seq, d_seq, rows, cols, grid.x, block.x, 0
                );
                
                
                // WARP SMEM ==============================================================
                
                cout << "\t\tWARP SMEM [" << (int)i + 1 << "/" << TEST_REPETITION << "]" << endl;
                runKernel(
                    gpu_alignment_warp_smem_loop, "warp_smem", h, h1, d_h1, nBytes, rows, cols, true,
                    filename, n, b,
                    d_main_seq, d_seq, rows, cols, grid.x, block.x, smem_size
                );
                
                appendStringToCSV(filename, "\n");
            }
        }
        
        free(h);
        free(h1);
        free(empty_h);
        cudaFree(d_h1);
        cudaFree(d_main_seq);
        cudaFree(d_seq);
    }
}

int main(int argc, char **argv) {
    const string& filename = makeCSV();
    if(filename == "") return(1);
    runTests(filename);
    cout << "Done" << endl;
	return(0);
}