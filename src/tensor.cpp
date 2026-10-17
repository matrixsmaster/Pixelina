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
    if (features && count <= SIZE_MAX / features / sizeof(float)) values.resize(count * features);
    else rows = channels = 0;
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
    const PixWeight &weight = model.tensor(name + WEIGHT_SUFFIX);
    PixTensor output(input.rows,size_t(weight.shape[1]));
    if (!cpu_matmul(model,weight,input.values.data(),input.rows,output.values.data())) return {};
    if (bias) {
        std::vector<float> vector = tensor_vector(model,name + BIAS_SUFFIX);
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
    if (input.values.empty() || (!weight.empty() && weight.size() != input.channels)) return {};
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

void tensor_gelu(PixTensor &tensor, bool approximate)
{
    if (tensor.values.empty()) return;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_pointwise(approximate? VK_OP_GELU_TANH : VK_OP_GELU_EXACT,tensor,nullptr,nullptr);
        return;
    }
#endif
    for (float &value : tensor.values) {
        float argument = approximate? std::tanh(COMPUTE_GELU_SCALE * (value + COMPUTE_GELU_CUBIC * value * value * value)) : std::erf(value / COMPUTE_SQRT_TWO);
        value = 0.5f * value * (1 + argument);
    }
}

void tensor_bias(PixTensor &tensor, const std::vector<float> &bias)
{
    if (tensor.values.empty()) return;
    if (bias.size() != tensor.channels) {
        tensor = PixTensor();
        return;
    }
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_bias(tensor,bias);
        return;
    }
#endif
    for (size_t i = 0; i < tensor.values.size(); i++) tensor.values[i] += bias[i % tensor.channels];
}

PixTensor tensor_l2norm(const PixTensor &input, const std::vector<float> &weight, float epsilon)
{
    if (input.values.empty() || (!weight.empty() && weight.size() != input.channels) || epsilon <= 0) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_l2norm(input,weight,epsilon);
#endif
    PixTensor output(input.rows,input.channels);
    cpu_parallel(input.rows,[&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; row++) {
            double square = 0;
            for (size_t c = 0; c < input.channels; c++) square += double(input.values[row * input.channels + c]) * input.values[row * input.channels + c];
            float inverse = float(std::sqrt(double(input.channels)) / std::max(std::sqrt(square),double(epsilon)));
            for (size_t c = 0; c < input.channels; c++) output.values[row * input.channels + c] = input.values[row * input.channels + c] * inverse * (weight.empty()? 1 : weight[c]);
        }
    });
    return output;
}

PixTensor tensor_slice(const PixTensor &input, size_t row, size_t rows, size_t feature, size_t channels)
{
    if (row > input.rows || rows > input.rows - row || feature > input.channels || channels > input.channels - feature || input.values.empty()) return {};
    PixTensor output(rows,channels);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_copy_rows(input,output,rows,row,feature);
        return output;
    }
#endif
    for (size_t i = 0; i < rows; i++) std::copy_n(input.values.data() + (row + i) * input.channels + feature,channels,output.values.data() + i * channels);
    return output;
}

PixTensor tensor_gather(const PixTensor &input, const std::vector<size_t> &rows)
{
    if (input.values.empty()) return {};
    for (size_t row : rows) {
        if (row >= input.rows) return {};
    }
    PixTensor output(rows.size(),input.channels);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        if (!vulkan_rows(input,output,rows,false)) return {};
        return output;
    }
#endif
    for (size_t i = 0; i < rows.size(); i++) std::copy_n(input.values.data() + rows[i] * input.channels,input.channels,output.values.data() + i * input.channels);
    return output;
}

bool tensor_scatter(PixTensor &destination, const PixTensor &source, const std::vector<size_t> &rows)
{
    if (source.rows != rows.size() || source.channels != destination.channels || source.values.empty() || destination.values.empty()) return false;
    for (size_t row : rows) {
        if (row >= destination.rows) return false;
    }
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_rows(source,destination,rows,true);
#endif
    for (size_t i = 0; i < rows.size(); i++) std::copy_n(source.values.data() + i * source.channels,source.channels,destination.values.data() + rows[i] * destination.channels);
    return true;
}

PixTensor tensor_upsample(const PixTensor &input, size_t width, size_t height)
{
    if (!width || !height || width > SIZE_MAX / height || input.rows != width * height || input.rows > SIZE_MAX / 4 || input.values.empty()) return {};
    PixTensor output(input.rows * 4,input.channels);
    if (output.values.empty()) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_layout(VK_OP_UPSAMPLE,input,output,width);
        return output;
    }
#endif
    for (size_t y = 0; y < height * 2; y++) {
        for (size_t x = 0; x < width * 2; x++) std::copy_n(input.values.data() + ((y / 2) * width + x / 2) * input.channels,input.channels,output.values.data() + (y * width * 2 + x) * input.channels);
    }
    return output;
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

PixTensor tensor_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, const std::vector<PixAttentionRange> &ranges, size_t query_offset)
{
    if (!heads || !kv_heads || heads % kv_heads || q.channels % heads || q.values.empty() || k.values.empty() || v.values.empty()) return {};
    if (k.rows != v.rows || k.channels != v.channels || k.channels != q.channels / heads * kv_heads || query_offset > SIZE_MAX - q.rows) return {};
    size_t covered = 0;
    for (const PixAttentionRange &range : ranges) {
        if (range.query_begin != covered || range.query_end <= range.query_begin || range.query_end > q.rows || !range.key_end || range.key_end > k.rows) return {};
        covered = range.query_end;
    }
    if (covered != q.rows) return {};
    size_t dimension = q.channels / heads;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_attention(q,k,v,heads,kv_heads,ranges,query_offset);
#endif
    PixTensor output(q.rows,q.channels);
    float scale = 1.0f / std::sqrt(float(dimension));
    auto run_rows = [&](size_t begin, size_t end) {
        std::vector<float> scores(k.rows);
        for (size_t item = begin; item < end; item++) {
            size_t token = item / heads,head = item % heads,kv_head = head / (heads / kv_heads);
            const float* query = q.values.data() + token * q.channels + head * dimension;
            auto found = std::upper_bound(ranges.begin(),ranges.end(),token,[](size_t row, const PixAttentionRange &range) { return row < range.query_end; });
            size_t tokens = found->causal? std::min(found->key_end,query_offset + token + 1) : found->key_end;
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
    std::vector<float> weight = tensor_vector(model,name + WEIGHT_SUFFIX),bias = tensor_vector(model,name + BIAS_SUFFIX);
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

PixTensor tensor_resample_shortcut(const PixTensor &input, size_t width, size_t height, size_t channels, size_t temporal, bool up, bool spatial)
{
    if (!width || !height || !channels || !temporal || input.values.empty() || width > SIZE_MAX / height || input.rows != width * height) return {};
    size_t side = spatial? 2 : 1;
    if (temporal > SIZE_MAX / (side * side) || width > SIZE_MAX / side || height > SIZE_MAX / side) return {};
    size_t factor = temporal * side * side;
    if (channels > SIZE_MAX / factor || input.channels > SIZE_MAX / factor) return {};
    if ((up && channels * factor % input.channels) || (!up && input.channels * factor % channels)) return {};
    if (!up && (width % side || height % side)) return {};
    size_t output_width = up? width * side : width / side,output_height = up? height * side : height / side;
    if (output_width > SIZE_MAX / output_height) return {};
    PixTensor output(output_width * output_height,channels);
    if (output.values.empty()) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        if (!vulkan_shortcut(input,output,width,temporal,up,spatial)) return {};
        return output;
    }
#endif
    size_t group = up? channels * factor / input.channels : input.channels * factor / channels;
    for (size_t y = 0; y < output_height; y++) {
        for (size_t x = 0; x < output_width; x++) {
            for (size_t c = 0; c < channels; c++) {
                float value = 0;
                if (up) {
                    size_t source_channel = (c * factor + (temporal - 1) * side * side + (y % side) * side + x % side) / group;
                    value = input.values[((y / side) * width + x / side) * input.channels + source_channel];
                } else {
                    for (size_t i = 0; i < group; i++) {
                        size_t item = c * group + i,source_channel = item / factor;
                        size_t time = item / (side * side) % temporal,dy = item / side % side,dx = item % side;
                        if (time == temporal - 1) value += input.values[((y * side + dy) * width + x * side + dx) * input.channels + source_channel];
                    }
                    value /= group;
                }
                output.values[(y * output_width + x) * channels + c] = value;
            }
        }
    }
    return output;
}

PixTensor tensor_linear_rows(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t first, size_t count)
{
    if (input.values.empty()) return {};
    auto found = model.weights.find(name + WEIGHT_SUFFIX);
    if (found == model.weights.end()) return {};
    PixWeight weight = found->second;
    if (weight.shape.size() != 2 || input.channels != weight.shape[0] || first > weight.shape[1] || count > weight.shape[1] - first) return {};
    size_t stride = size_t(weight_bytes({weight.shape[0]},weight.type));
    weight.offset += first * stride;
    weight.shape[1] = count;
    weight.bytes = count * stride;
    weight.name += TENSOR_SLICE_SEPARATOR + std::to_string(first) + TENSOR_SLICE_SEPARATOR + std::to_string(count);
    PixTensor output(input.rows,count);
    if (!cpu_matmul(model,weight,input.values.data(),input.rows,output.values.data())) return {};
    return output;
}
