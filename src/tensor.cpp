/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <memory>

PixTensor::PixTensor(size_t count, size_t features) : rows(count),channels(features)
{
    values.resize(count * features);
}

static std::atomic<size_t> configured_threads{0};

struct CpuWorkers {
    size_t count; // persistent workers including the calling thread
    size_t work = 0; // current task range
    size_t participating = 0; // active workers for this task
    size_t remaining = 0; // workers still executing the task
    uint64_t generation = 0; // wake-up sequence for publishing the next parallel task
    bool stopping = false; // orderly shutdown flag
    std::vector<std::thread> threads; // workers retained between operators
    std::mutex mutex; // task publication and completion
    std::condition_variable ready,finished; // task and completion notifications
    std::function<void(size_t,size_t)> function; // current operation

    explicit CpuWorkers(size_t workers) : count(workers)
    {
        for (size_t index = 1; index < count; index++) threads.emplace_back([this,index]() { worker(index); });
    }

    ~CpuWorkers()
    {
        stop();
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (std::thread &thread : threads) thread.join();
        threads.clear();
    }

    void worker(size_t index)
    {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping) {
            ready.wait(lock,[&]() { return stopping || generation != seen; });
            if (stopping) break;
            seen = generation;
            if (index >= participating) continue;
            lock.unlock();
            function(work * index / participating,work * (index + 1) / participating);
            lock.lock();
            remaining--;
            if (!remaining) finished.notify_one();
        }
    }

    void run(size_t amount, size_t workers, const std::function<void(size_t,size_t)> &task)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            work = amount;
            participating = workers;
            remaining = workers - 1;
            function = task;
            generation++;
        }
        ready.notify_all();
        function(0,work / participating);
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock,[&]() { return !remaining; });
        function = nullptr;
    }
};

void cpu_set_threads(size_t count)
{
    configured_threads.store(count);
}

size_t cpu_thread_count()
{
    size_t count = configured_threads.load();
    return count? count : std::max(1u,std::thread::hardware_concurrency());
}

void cpu_parallel(size_t work, const std::function<void(size_t,size_t)> &function)
{
    size_t available = cpu_thread_count();
    size_t count = std::min(available,std::max(size_t(1),work / COMPUTE_THREAD_MIN_WORK));
    if (count == 1) {
        function(0,work);
        return;
    }
    // Each inference worker owns its pool; operations reuse it for the entire generation.
    static thread_local std::unique_ptr<CpuWorkers> workers;
    if (!workers || workers->count != available) workers.reset(new CpuWorkers(available));
    workers->run(work,count,function);
}

#ifdef COMPUTE_X86_SIMD
__attribute__((target("avx2,fma"))) static float dot_avx(const float* a, const float* b, size_t count)
{
    __m256 sum0 = _mm256_setzero_ps(),sum1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),_mm256_loadu_ps(b + i),sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),_mm256_loadu_ps(b + i + 8),sum1);
    }
    float partial[8];
    _mm256_storeu_ps(partial,_mm256_add_ps(sum0,sum1));
    float result = 0;
    for (float value : partial) result += value;
    for (; i < count; i++) result += a[i] * b[i];
    return result;
}
#endif

float cpu_dot(const float* a, const float* b, size_t count)
{
#ifdef COMPUTE_X86_SIMD
    static bool avx = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (avx) return dot_avx(a,b,count);
#endif
    double result = 0;
    for (size_t i = 0; i < count; i++) result += double(a[i]) * b[i];
    return float(result);
}

std::vector<float> tensor_vector(const PixModelFile &model, const std::string &name)
{
    const PixWeight &weight = model.tensor(name);
    uint64_t count = 1;
    for (uint64_t dimension : weight.shape) count *= dimension;
    std::vector<float> result(size_t(count),0);
    cpu_decode(model.data + weight.offset,weight.type,size_t(count),result.data());
    return result;
}

PixTensor tensor_linear(const PixModelFile &model, const std::string &name, const PixTensor &input, bool bias)
{
    if (input.values.empty()) return {};
    const PixWeight &weight = model.tensor(name + ".weight");
    PixTensor output(input.rows,size_t(weight.shape[1]));
    if (!cpu_matmul(model,weight,input.values.data(),input.rows,output.values.data())) return {};
    if (bias) {
        std::vector<float> vector = tensor_vector(model,name + ".bias");
#ifdef PIXELINA_WITH_VULKAN
        if (vulkan_active_state()) {
            vulkan_bias(output,vector);
        } else
#endif
        for (size_t i = 0; i < output.values.size(); i++) output.values[i] += vector[i % output.channels];
    }
    return output;
}

PixTensor tensor_norm(const PixTensor &input, const std::vector<float> &weight, float epsilon, bool centered)
{
    if (input.values.empty()) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_norm(input,weight,epsilon,centered);
#endif
    PixTensor output(input.rows,input.channels);
    auto run_rows = [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; row++) {
            const float* from = input.values.data() + row * input.channels;
            float* to = output.values.data() + row * input.channels;
            double mean = 0,square = 0;
            if (centered) {
                for (size_t c = 0; c < input.channels; c++) mean += from[c];
                mean /= input.channels;
            }
            for (size_t c = 0; c < input.channels; c++) square += (from[c] - mean) * (from[c] - mean);
            float inverse = float(1.0 / std::sqrt(square / input.channels + epsilon));
            for (size_t c = 0; c < input.channels; c++) to[c] = float(from[c] - mean) * inverse * (weight.empty()? 1 : weight[c]);
        }
    };
    cpu_parallel(input.rows,run_rows);
    return output;
}

void tensor_silu(PixTensor &tensor)
{
    if (tensor.values.empty()) return;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_pointwise(VK_OP_SILU,tensor,nullptr,nullptr);
        return;
    }
#endif
    for (float &value : tensor.values) value /= 1.0f + std::exp(-value);
}

void tensor_add(PixTensor &destination, const PixTensor &source)
{
    if (source.values.empty()) destination = PixTensor();
    if (destination.values.empty()) return;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_pointwise(VK_OP_ADD,destination,&source,nullptr);
        return;
    }
#endif
    for (size_t i = 0; i < destination.values.size(); i++) destination.values[i] += source.values[i];
}

void tensor_gate(PixTensor &destination, const PixTensor &gate)
{
    if (gate.values.empty()) destination = PixTensor();
    if (destination.values.empty()) return;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_pointwise(VK_OP_GATE,destination,&gate,nullptr);
        return;
    }
#endif
    for (size_t i = 0; i < destination.values.size(); i++) destination.values[i] *= gate.values[i];
}

PixTensor tensor_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, bool causal)
{
    if (q.values.empty() || k.values.empty() || v.values.empty()) return {};
    size_t dimension = q.channels / heads;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_attention(q,k,v,heads,kv_heads,causal);
#endif
    PixTensor output(q.rows,q.channels);
    float scale = 1.0f / std::sqrt(float(dimension));
    auto run_rows = [&](size_t begin, size_t end) {
        std::vector<float> scores(k.rows);
        for (size_t item = begin; item < end; item++) {
            size_t token = item / heads,head = item % heads,kv_head = head / (heads / kv_heads);
            const float* query = q.values.data() + token * q.channels + head * dimension;
            size_t tokens = causal? token + 1 : k.rows;
            float maximum = -INFINITY;
            for (size_t j = 0; j < tokens; j++) {
                scores[j] = cpu_dot(query,k.values.data() + j * k.channels + kv_head * dimension,dimension) * scale;
                maximum = std::max(maximum,scores[j]);
            }
            double sum = 0;
            for (size_t j = 0; j < tokens; j++) {
                scores[j] = std::exp(scores[j] - maximum);
                sum += scores[j];
            }
            float* to = output.values.data() + token * output.channels + head * dimension;
            for (size_t j = 0; j < tokens; j++) {
                float probability = float(scores[j] / sum);
                const float* value = v.values.data() + j * v.channels + kv_head * dimension;
                for (size_t c = 0; c < dimension; c++) to[c] += probability * value[c];
            }
        }
    };
    cpu_parallel(q.rows * heads,run_rows);
    return output;
}

void tensor_rope(PixTensor &tensor, size_t heads, const std::vector<float> &cosine, const std::vector<float> &sine, bool interleaved)
{
    if (tensor.values.empty()) return;
    size_t dimension = tensor.channels / heads;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_rope(tensor,heads,cosine,sine,interleaved);
        return;
    }
#endif
    for (size_t row = 0; row < tensor.rows; row++) {
        for (size_t head = 0; head < heads; head++) {
            float* data = tensor.values.data() + row * tensor.channels + head * dimension;
            for (size_t i = 0; i < dimension / 2; i++) {
                size_t first = interleaved? i * 2 : i;
                size_t second = interleaved? i * 2 + 1 : i + dimension / 2;
                float a = data[first],b = data[second];
                float c = cosine[row * dimension / 2 + i],s = sine[row * dimension / 2 + i];
                data[first] = a * c - b * s;
                data[second] = a * s + b * c;
            }
        }
    }
}

PixTensor tensor_groupnorm(const PixModelFile &model, const std::string &name, const PixTensor &input)
{
    if (input.values.empty()) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_groupnorm(model,name,input);
#endif
    std::vector<float> weight = tensor_vector(model,name + ".weight"),bias = tensor_vector(model,name + ".bias");
    PixTensor output(input.rows,input.channels);
    size_t group_size = input.channels / COMPUTE_VAE_GROUPS;
    auto run_rows = [&](size_t begin, size_t end) {
        for (size_t group = begin; group < end; group++) {
            double sum = 0,square = 0;
            size_t count = input.rows * group_size;
            for (size_t row = 0; row < input.rows; row++) {
                for (size_t c = group * group_size; c < (group + 1) * group_size; c++) sum += input.values[row * input.channels + c];
            }
            double mean = sum / count;
            for (size_t row = 0; row < input.rows; row++) {
                for (size_t c = group * group_size; c < (group + 1) * group_size; c++) {
                    double delta = input.values[row * input.channels + c] - mean;
                    square += delta * delta;
                }
            }
            float inverse = float(1.0 / std::sqrt(square / count + COMPUTE_NORM_EPSILON));
            for (size_t row = 0; row < input.rows; row++) {
                for (size_t c = group * group_size; c < (group + 1) * group_size; c++) output.values[row * input.channels + c] = float(input.values[row * input.channels + c] - mean) * inverse * weight[c] + bias[c];
            }
        }
    };
    cpu_parallel(COMPUTE_VAE_GROUPS,run_rows);
    return output;
}
