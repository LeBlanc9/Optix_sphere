#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "./core.cuh"

/**
 * @brief A RAII wrapper for CUDA device memory management.
 *
 * This class simplifies handling of device buffers by automatically
 * allocating and freeing memory. It also provides helper methods for
 * copying data to and from the host.
 */
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    DeviceBuffer(size_t size_in_bytes) {
        alloc(size_in_bytes);
    }

    ~DeviceBuffer() {
        free();
    }

    // Disable copy constructor and assignment
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move constructor and assignment
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : d_ptr_(other.d_ptr_), size_(other.size_) {
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void alloc(size_t size_in_bytes) {
        if (d_ptr_) {
            free();
        }
        size_ = size_in_bytes;
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, size_));
        }
    }

    void free() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
            d_ptr_ = nullptr;
            size_ = 0;
        }
    }

    void upload(const void* host_ptr, size_t size_in_bytes) {
        if (size_in_bytes > size_) {
            alloc(size_in_bytes);
        }
        CUDA_CHECK(cudaMemcpy(d_ptr_, host_ptr, size_in_bytes, cudaMemcpyHostToDevice));
    }
    
    template <typename T>
    void upload(const std::vector<T>& vec) {
        upload(vec.data(), vec.size() * sizeof(T));
    }

    void download(void* host_ptr, size_t size_in_bytes) const {
        if (size_in_bytes > size_) {
           throw std::runtime_error("Download size exceeds buffer size.");
        }
        CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr_, size_in_bytes, cudaMemcpyDeviceToHost));
    }
    
    template <typename T>
    void download(std::vector<T>& vec) const {
        if (vec.size() * sizeof(T) > size_) {
            throw std::runtime_error("Download vector size exceeds buffer size.");
        }
        download(vec.data(), vec.size() * sizeof(T));
    }

    void* get() { return d_ptr_; }
    const void* get() const { return d_ptr_; }

    template<typename T>
    T* get() { return reinterpret_cast<T*>(d_ptr_); }

    template<typename T>
    const T* get() const { return reinterpret_cast<const T*>(d_ptr_); }

    CUdeviceptr get_cu_ptr() const {
        return reinterpret_cast<CUdeviceptr>(d_ptr_);
    }

    size_t size() const { return size_; }

private:
    void* d_ptr_ = nullptr;
    size_t size_ = 0;
};
