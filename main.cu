#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>
#include <cctype>

std::vector<char> read_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    file.read(buffer.data(), fileSize);
    file.close();

    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}


int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i< int(data.size()); ++i)
    {
        // test 1: does this match the token?
        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        // test 2: is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        // test 3: is the prefix a non-letter character?
        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;
        ++numOccurrences;
    }
    return numOccurrences;
}

// CUDA kernel for counting word occurrences 
__global__ void find_occurrences(const char* data, int data_size, const char* token, int token_len, int* result) {
    __shared__ int local_count;
    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size - token_len) return;

    bool match = true;
    for (int i = 0; i < token_len; ++i) {
        if (data[idx + i] != token[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        bool valid_prefix = (idx == 0) || (data[idx - 1] < 'a' || data[idx - 1] > 'z');
        bool valid_suffix = (idx + token_len == data_size) || (data[idx + token_len] < 'a' || data[idx + token_len] > 'z');

        if (valid_prefix && valid_suffix) {
            atomicAdd(&local_count, 1);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(result, local_count);
    }
}

int main() {
    const char* filepaths[] = {
        "dataset/beowulf.txt",
        "dataset/shakespeare.txt",
        "dataset/pride_and_prejudice.txt",
        "dataset/edgar_allan_poe.txt",
        "dataset/crime_and_punishment.txt"
    };

    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };  
    int num_runs = 10; 

    double total_cpu_time = 0.0;
    float total_gpu_time = 0.0;
    int word_count = sizeof(words) / sizeof(words[0]);
    int file_count = sizeof(filepaths) / sizeof(filepaths[0]);

    for (const char* filepath : filepaths) {
        std::cout << "Searching in file: " << filepath << std::endl;

        // Read the file into host memory 
        std::vector<char> file_data = read_file(filepath);
        if (file_data.empty()) continue;

        for (const char* word : words) {
            std::cout << "Searching for word: " << word << std::endl;

            
            double cpu_time_accumulated = 0.0;
            for (int run = 0; run < num_runs; ++run) {
               auto cpu_start = std::chrono::high_resolution_clock::now();
                int cpu_count = calc_token_occurrences(file_data, word);
                auto cpu_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
                cpu_time_accumulated += cpu_duration.count();
            }
            double avg_cpu_time = cpu_time_accumulated / num_runs;
            total_cpu_time += avg_cpu_time;
            std::cout << "[CPU] Average time: " << avg_cpu_time << " ms" << std::endl;

           
            char* d_file_data;
            cudaMalloc((void**)&d_file_data, file_data.size() * sizeof(char));
            cudaMemcpy(d_file_data, file_data.data(), file_data.size() * sizeof(char), cudaMemcpyHostToDevice);

            int token_len = strlen(word);
            char* d_token;
            cudaMalloc((void**)&d_token, token_len * sizeof(char));
            cudaMemcpy(d_token, word, token_len * sizeof(char), cudaMemcpyHostToDevice);

            int* d_result;
            int h_result = 0;
            cudaMalloc((void**)&d_result, sizeof(int));

            int threads_per_block = 1024;
            int num_blocks = (file_data.size() + threads_per_block - 1) / threads_per_block;

            
            float gpu_time_accumulated = 0.0;
            for (int run = 0; run < num_runs; ++run) {
                cudaMemset(d_result, 0, sizeof(int));  // Reset the result on the GPU
                cudaEvent_t gpu_start, gpu_stop;
                cudaEventCreate(&gpu_start);
                cudaEventCreate(&gpu_stop);

                //TODO Record time for kernel execution only
                cudaEventRecord(gpu_start);
                find_occurrences << <num_blocks, threads_per_block >> > (d_file_data, file_data.size(), d_token, token_len, d_result);
                cudaEventRecord(gpu_stop);
                cudaEventSynchronize(gpu_stop);

                float gpu_milliseconds = 0;
                cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop);
                gpu_time_accumulated += gpu_milliseconds;

                cudaEventDestroy(gpu_start);
                cudaEventDestroy(gpu_stop);
            }
            float avg_gpu_time = gpu_time_accumulated / num_runs;
            total_gpu_time += avg_gpu_time;

            cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

            std::cout << "[GPU] Average time: " << avg_gpu_time << " ms" << std::endl;
            std::cout << "[GPU] Found " << h_result << " occurrences of word: " << word << std::endl;

            // Free GPU memory outside timing
            cudaFree(d_file_data);
            cudaFree(d_token);
            cudaFree(d_result);
            std::cout << "--------------------------------------" << std::endl;
        }
    }

    // Calculate the overall average CPU and GPU times across all files and words
    double overall_avg_cpu_time = total_cpu_time / (file_count * word_count);
    float overall_avg_gpu_time = total_gpu_time / (file_count * word_count);

    std::cout << "======================================" << std::endl;
    std::cout << "Overall averaged CPU time: " << overall_avg_cpu_time << " ms" << std::endl;
    std::cout << "Overall averaged GPU time: " << overall_avg_gpu_time << " ms" << std::endl;

    return 0;
}