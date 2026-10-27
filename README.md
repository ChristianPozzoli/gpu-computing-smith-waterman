# Smith-Waterman CUDA (block / SMEM / warp)

Implementazione CUDA C++ dell'algoritmo di Smith-Waterman (allineamento locale di sequenze DNA).

## Requisiti

- Windows con una GPU NVIDIA compatibile con CUDA
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (include `nvcc`)
- Visual Studio con "Sviluppo di applicazioni desktop con C++" (fornisce il compilatore host `cl.exe`)

## Compilazione

```bat
nvcc -O3 -std=c++17 -arch=native smith_waterman.cu -o smith_waterman.exe
```

### Flag

| Flag | Significato |
|------|-------------|
| `-O3` | Ottimizzazioni sul codice host |
| `-std=c++17` | Standard C++ (va bene anche `c++14`) |
| `-arch=native` | Compila per la GPU installata (richiede CUDA 11.6 o successivo) |
| `-o smith_waterman.exe` | Nome dell'eseguibile prodotto |

### Varianti

- **CUDA precedente alla 11.6**: sostituisci `-arch=native` con l'architettura della tua GPU, ad esempio:
  - `-arch=sm_75` per le serie RTX 20xx
  - `-arch=sm_86` per le serie RTX 30xx
  - `-arch=sm_89` per le serie RTX 40xx
- **Profiling con Nsight**: aggiungi `-lineinfo`.
