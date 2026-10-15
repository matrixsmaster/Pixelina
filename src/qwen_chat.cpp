/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "qwen_chat.h"
#include "qwen_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

static bool qwen_chat_append(PixVLInput &input, const PixTokenz &tokenizer, const std::string &text, size_t &position, bool special)
{
    std::vector<int> ids = tokenizer.encode(text,false,special);
    if ((!text.empty() && ids.empty()) || ids.size() > QWEN_VL_CONTEXT - input.ids.size()) return false;
    for (int id : ids) {
        input.ids.push_back(id);
        input.image_slots.push_back(0);
        for (auto &axis : input.positions) axis.push_back(position);
        position++;
    }
    return true;
}

PixVLInput qwen_chat_prepare(const PixTokenz &tokenizer, const std::vector<PixChatMessage> &messages, const PixImage &image, bool pack_pixels, bool thinking)
{
    if (messages.empty() || messages.back().assistant) return {};
    PixVLInput packed = qwen_vl_prepare(tokenizer,"",{image},pack_pixels);
    if (packed.ids.empty() || packed.grids.size() != 1) return {};
    PixVLInput input;
    input.grids = std::move(packed.grids);
    input.patches = std::move(packed.patches);
    size_t position = 0;
    for (size_t message = 0; message < messages.size(); message++) {
        const PixChatMessage &turn = messages[message];
        if (turn.assistant != bool(message % 2)) return {};
        if (!qwen_chat_append(input,tokenizer,turn.assistant? QWEN_CHAT_ASSISTANT : QWEN_VL_USER,position,true)) return {};
        if (!message) {
            if (!qwen_chat_append(input,tokenizer,QWEN_VL_VISION_START_TEXT,position,true)) return {};
            size_t height = input.grids[0].height / QWEN_VISION_MERGE,width = input.grids[0].width / QWEN_VISION_MERGE;
            if (height * width > QWEN_VL_CONTEXT - input.ids.size()) return {};
            for (size_t y = 0; y < height; y++) {
                for (size_t x = 0; x < width; x++) {
                    input.ids.push_back(QWEN_VL_IMAGE_TOKEN);
                    input.image_slots.push_back(1);
                    input.positions[0].push_back(position);
                    input.positions[1].push_back(position + y);
                    input.positions[2].push_back(position + x);
                }
            }
            position += std::max(height,width);
            if (!qwen_chat_append(input,tokenizer,QWEN_VL_VISION_END_TEXT,position,true)) return {};
        }
        std::string text = turn.text;
        if (thinking && turn.assistant) {
            size_t end = text.find(QWEN_CHAT_THINK_END);
            // The official Thinking template excludes earlier turns' reasoning from new requests.
            if (end != std::string::npos) {
                text.erase(0,end + sizeof(QWEN_CHAT_THINK_END) - 1);
                size_t first = text.find_first_not_of('\n');
                text.erase(0,first == std::string::npos? text.size() : first);
            } else if (text.compare(0,sizeof(QWEN_CHAT_THINK) - 1,QWEN_CHAT_THINK) == 0) text.clear();
        }
        if (!qwen_chat_append(input,tokenizer,text,position,false) || !qwen_chat_append(input,tokenizer,QWEN_CHAT_MESSAGE_END,position,true)) return {};
    }
    if (!qwen_chat_append(input,tokenizer,QWEN_CHAT_ASSISTANT,position,true)) return {};
    if (thinking && !qwen_chat_append(input,tokenizer,QWEN_CHAT_THINK "\n",position,true)) return {};
    return input;
}

int qwen_chat_sample(const PixTensor &logits, const std::vector<unsigned char> &seen, const PixChatSampling &sampling, std::mt19937_64 &random)
{
    if (logits.rows != 1 || logits.channels != seen.size() || logits.values.size() != seen.size() || seen.empty() || sampling.temperature <= 0 || sampling.top_p <= 0 || sampling.top_p > 1) return -1;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state() && !vulkan_synchronize()) return -1;
#endif
    std::vector<double> scores(seen.size());
    std::vector<size_t> order(seen.size());
    std::iota(order.begin(),order.end(),0);
    for (size_t id = 0; id < seen.size(); id++) {
        if (!std::isfinite(logits.values[id])) return -1;
        scores[id] = (logits.values[id] - (seen[id]? sampling.presence : 0)) / sampling.temperature;
    }
    size_t count = std::min(size_t(QWEN_CHAT_TOP_K),order.size());
    std::partial_sort(order.begin(),order.begin() + count,order.end(),[&](size_t a, size_t b) { return scores[a] == scores[b]? a < b : scores[a] > scores[b]; });
    std::vector<double> probabilities(count);
    double sum = 0;
    for (size_t i = 0; i < count; i++) {
        probabilities[i] = std::exp(scores[order[i]] - scores[order[0]]);
        sum += probabilities[i];
    }
    double cumulative = 0;
    size_t retained = 0;
    do {
        cumulative += probabilities[retained++];
    } while (retained < count && cumulative < sampling.top_p * sum);
    std::uniform_real_distribution<double> uniform(0,cumulative);
    double selected = uniform(random);
    for (size_t i = 0; i < retained; i++) {
        selected -= probabilities[i];
        if (selected <= 0) return int(order[i]);
    }
    return int(order[retained - 1]);
}

size_t qwen_chat_utf8_prefix(const std::string &text)
{
    size_t cursor = 0;
    while (cursor < text.size()) {
        unsigned char byte = text[cursor];
        size_t length = 1;
        if ((byte & QWEN_CHAT_UTF8_MASK) == QWEN_CHAT_UTF8_MASK) {
            if (byte >= QWEN_CHAT_UTF8_FOUR) length = 4;
            else if (byte >= QWEN_CHAT_UTF8_THREE) length = 3;
            else length = 2;
        }
        if (length > text.size() - cursor) break;
        bool complete = true;
        for (size_t i = 1; i < length; i++) complete = complete && (static_cast<unsigned char>(text[cursor + i]) & QWEN_CHAT_UTF8_MASK) == QWEN_CHAT_UTF8_CONTINUATION;
        cursor += complete? length : 1;
    }
    return cursor;
}

bool qwen_chat_memory(size_t capacity, size_t rows, uint64_t features, uint64_t device_bytes, uint64_t max_buffer, uint64_t largest_weight, uint64_t &activation, uint64_t &scratch)
{
    if (!rows || rows > QWEN_CHAT_PREFILL_BLOCK || capacity < rows || capacity > QWEN_VL_CONTEXT) return false;
    uint64_t binding = uint64_t(capacity) * QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM * sizeof(float);
    uint64_t width = std::max(uint64_t(QWEN_VL_HIDDEN) * QWEN_CHAT_MEMORY_MAPS + uint64_t(QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM) * 2,uint64_t(QWEN_VL_HIDDEN) * QWEN_CHAT_FFN_MAPS + uint64_t(QWEN_VL_MLP) * 2);
    // Reserve an extra layer pair while replacing cache buffers at a capacity boundary.
    activation = binding * (QWEN_VL_LAYERS + 1) * 2 + rows * width * sizeof(float) + features;
    activation += QWEN_VL_VOCABULARY * sizeof(float) + rows * QWEN_VL_HEAD_DIM * sizeof(float);
    scratch = uint64_t(capacity) * cpu_thread_count() * sizeof(float) + uint64_t(COMPUTE_WEIGHT_ROWS) * QWEN_VL_MLP * sizeof(float);
    scratch += uint64_t(rows) * QWEN_VL_MLP / MODEL_QUANT_BLOCK * MODEL_Q8_K_BYTES;
    uint64_t largest = std::max({binding,uint64_t(rows) * QWEN_VL_MLP * sizeof(float),largest_weight});
    if (largest > max_buffer) return false;
    return !device_bytes || (activation < device_bytes && scratch < device_bytes - activation && largest_weight < device_bytes - activation - scratch);
}

bool qwen_chat_reserve(PixVLCache &cache, size_t required)
{
    if (!required || required > QWEN_VL_CONTEXT || cache.rows > cache.capacity) return false;
    if (required <= cache.capacity) return true;
    size_t capacity = std::min(size_t(QWEN_VL_CONTEXT),(required + QWEN_CHAT_CACHE_BLOCK - 1) / QWEN_CHAT_CACHE_BLOCK * QWEN_CHAT_CACHE_BLOCK);
    for (PixVLCacheLayer &layer : cache.layers) {
        PixTensor key(capacity,QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM),value(capacity,QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM);
        if (cache.rows) {
#ifdef PIXELINA_WITH_VULKAN
            if (vulkan_active_state()) {
                vulkan_copy_rows(layer.key,key,cache.rows);
                vulkan_copy_rows(layer.value,value,cache.rows);
            } else
#endif
            {
                std::copy_n(layer.key.values.begin(),cache.rows * key.channels,key.values.begin());
                std::copy_n(layer.value.values.begin(),cache.rows * value.channels,value.values.begin());
            }
        }
        layer.key = std::move(key);
        layer.value = std::move(value);
        layer.key.rows = layer.value.rows = cache.rows;
    }
    cache.capacity = capacity;
    return true;
}

static PixTensor qwen_chat_logits(const PixModelFile &language, const PixTensor &hidden, const InferenceProgress &progress)
{
    PixTensor logits(1,QWEN_VL_VOCABULARY);
    // Slice the vocabulary head so small GPU binding limits never require a full head allocation.
    for (size_t first = 0; first < logits.channels; first += QWEN_CHAT_LOGITS_BLOCK) {
        if (progress && !progress(0,"Sampling reply")) return {};
        size_t count = std::min(size_t(QWEN_CHAT_LOGITS_BLOCK),logits.channels - first);
        PixTensor block = tensor_linear_rows(language,QWEN_CHAT_OUTPUT,hidden,first,count);
        if (block.values.empty()) return {};
#ifdef PIXELINA_WITH_VULKAN
        if (vulkan_active_state()) {
            if (!vulkan_copy_features(block,logits,first)) return {};
        } else
#endif
        std::copy(block.values.begin(),block.values.end(),logits.values.begin() + first);
    }
    return logits;
}

bool qwen_chat_generate(const PixEditSettings &settings, const PixImage &image, const std::vector<PixChatMessage> &messages, std::string &response, std::string &error, const InferenceProgress &progress, const PixChatStream &stream)
{
    response.clear();
    error.clear();
    auto report = [&](float fraction, const std::string &stage) {
        return !progress || progress(fraction,stage);
    };
    if (!report(0,"Loading Qwen-VL")) return false;
    if (settings.encoder.empty() || settings.vision.empty() || settings.reference_resolution <= 0 || uint64_t(settings.reference_resolution) * settings.reference_resolution > PIXELINA_MAX_PIXELS || !image_valid(image) || (settings.backend != PIXELINA_CPU && settings.backend != PIXELINA_VULKAN)) {
        error = "Select a Qwen-VL language model, mmproj and valid reference resolution, and open an image.";
        return false;
    }
    PixModelFile language,vision;
    if (language.open(settings.encoder,false) || vision.open(settings.vision,false)) {
        error = "Cannot read the selected Qwen-VL language model or mmproj.";
        return false;
    }
    if (!qwen_vl_validate(language,vision,error)) return false;
    auto output = language.weights.find(QWEN_CHAT_OUTPUT WEIGHT_SUFFIX);
    if (output == language.weights.end() || output->second.shape != std::vector<uint64_t>{QWEN_VL_HIDDEN,QWEN_VL_VOCABULARY}) {
        error = "The Qwen-VL language model has no compatible output projection for chat.";
        return false;
    }
    PixChatSampling sampling;
    bool thinking = false;
    auto name = language.metadata.find("general.name");
    auto chat_template = language.metadata.find("tokenizer.chat_template");
    if ((chat_template != language.metadata.end() && chat_template->second.scalar.find(QWEN_CHAT_THINK) != std::string::npos) || (name != language.metadata.end() && name->second.scalar.find("Thinking") != std::string::npos)) {
        thinking = true;
        sampling.temperature = QWEN_CHAT_THINK_TEMPERATURE;
        sampling.top_p = QWEN_CHAT_THINK_TOP_P;
        sampling.presence = 0;
    }
    PixTokenz tokenizer;
    if (!tokenizer.load(language)) {
        error = "Cannot load the Qwen-VL tokenizer.";
        return false;
    }
    uint64_t largest_language = 0,largest_vision = 0;
    for (const auto &weight : language.weights) {
        if (weight.first == QWEN_VL_EMBEDDING_WEIGHT) continue;
        uint64_t bytes = weight.second.bytes;
        if (weight.first == QWEN_CHAT_OUTPUT WEIGHT_SUFFIX) bytes = weight_bytes({QWEN_VL_HIDDEN,QWEN_CHAT_LOGITS_BLOCK},weight.second.type);
        largest_language = std::max(largest_language,bytes);
    }
    for (const auto &weight : vision.weights) largest_vision = std::max(largest_vision,weight.second.bytes);
    language.close();
    vision.close();
    std::vector<PixImage> prepared = qwen_image_prepare({image},settings.reference_resolution);
    if (prepared.empty()) {
        error = "Cannot resize the image to the selected reference resolution.";
        return false;
    }
    PixVLInput input = qwen_chat_prepare(tokenizer,messages,prepared[0],false,thinking);
    if (input.ids.empty() || input.ids.size() >= QWEN_VL_CONTEXT) {
        error = "The conversation is invalid or has reached the model context capacity.";
        return false;
    }
    uint64_t device_bytes = 0,max_buffer = UINT64_MAX;
#ifdef PIXELINA_WITH_VULKAN
    VulkanSession session;
    if (settings.backend == PIXELINA_VULKAN && (!session.initialize() || !session.memory_limits(device_bytes,max_buffer))) {
        error = "Cannot initialize the Vulkan chat backend.";
        return false;
    }
#else
    if (settings.backend == PIXELINA_VULKAN) {
        error = "Vulkan support is unavailable in this build.";
        return false;
    }
#endif
    size_t vision_rows = input.grids[0].height * input.grids[0].width;
    uint64_t feature_bytes = uint64_t(vision_rows / QWEN_VISION_MERGE_PIXELS) * QWEN_VL_HIDDEN * (QWEN_VISION_DEEPSTACK + 1) * sizeof(float);
    uint64_t activation = uint64_t(vision_rows) * (QWEN_VISION_HIDDEN * QWEN_CHAT_VISION_MAPS + 3 * QWEN_VISION_TEMPORAL_PATCH * QWEN_VISION_PATCH * QWEN_VISION_PATCH) * sizeof(float) + feature_bytes;
    activation += uint64_t(QWEN_VISION_POSITIONS) * QWEN_VISION_HIDDEN * sizeof(float);
    uint64_t scratch = uint64_t(vision_rows) * cpu_thread_count() * sizeof(float) + uint64_t(COMPUTE_WEIGHT_ROWS) * QWEN_VISION_MLP * sizeof(float);
    auto admit = [&](uint64_t weight) {
        if (weight > max_buffer || (device_bytes && (activation >= device_bytes || scratch >= device_bytes - activation || weight >= device_bytes - activation - scratch))) {
            error = "Chat exceeds the available device memory; reduce reference resolution or conversation length.";
            return false;
        }
#ifdef PIXELINA_WITH_VULKAN
        if (settings.backend == PIXELINA_VULKAN) {
            session.packed_weight_budget = device_bytes - activation - scratch;
            if (!session.prepare_stage(activation,scratch)) {
                error = "Cannot reserve device memory for chat.";
                return false;
            }
        }
#endif
        return true;
    };
    if (std::max(uint64_t(vision_rows) * QWEN_VISION_MLP * sizeof(float),feature_bytes / (QWEN_VISION_DEEPSTACK + 1)) > max_buffer || !admit(largest_vision)) return false;
    input = qwen_chat_prepare(tokenizer,messages,prepared[0],true,thinking);
    prepared.clear();
    if (input.patches.empty() || !report(0,"Encoding image") || vision.open(settings.vision,false)) return false;
    PixVisionFeatures features = qwen_vl_vision(vision,input,report);
    if (features.merged.values.empty()) return false;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state() && !vulkan_synchronize()) return false;
#endif
    vision.close();
    input.patches.clear();
    if (language.open(settings.encoder,false)) return false;
    PixVLCache cache;
    std::vector<unsigned char> seen(QWEN_VL_VOCABULARY);
    // Qwen's presence penalty applies to generated tokens in the current reply.
    std::mt19937_64 random(uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    PixTensor hidden;
    size_t consumed = 0,image_row = 0;
    auto capacity = [&](size_t required, size_t rows, uint64_t retained) {
        if (required <= cache.capacity) return true;
        size_t allocated = std::min(size_t(QWEN_VL_CONTEXT),(required + QWEN_CHAT_CACHE_BLOCK - 1) / QWEN_CHAT_CACHE_BLOCK * QWEN_CHAT_CACHE_BLOCK);
        if (!qwen_chat_memory(allocated,rows,retained,device_bytes,max_buffer,largest_language,activation,scratch) || !admit(largest_language)) {
            if (error.empty()) error = "Chat KV cache exceeds the available device memory.";
            return false;
        }
        return qwen_chat_reserve(cache,required);
    };
    while (consumed < input.ids.size()) {
        if (!report(0,"Reading conversation")) return false;
        size_t rows = std::min(size_t(QWEN_CHAT_PREFILL_BLOCK),input.ids.size() - consumed);
        if (!capacity(consumed + rows,rows,feature_bytes * 2)) return false;
        PixVLInput block;
        block.ids.assign(input.ids.begin() + consumed,input.ids.begin() + consumed + rows);
        block.image_slots.assign(input.image_slots.begin() + consumed,input.image_slots.begin() + consumed + rows);
        for (size_t axis = 0; axis < block.positions.size(); axis++) block.positions[axis].assign(input.positions[axis].begin() + consumed,input.positions[axis].begin() + consumed + rows);
        size_t images = std::count(block.image_slots.begin(),block.image_slots.end(),1);
        PixVisionFeatures selected;
        if (images) {
            selected.merged = tensor_slice(features.merged,image_row,images,0,QWEN_VL_HIDDEN);
            for (size_t layer = 0; layer < QWEN_VISION_DEEPSTACK; layer++) selected.deepstack[layer] = tensor_slice(features.deepstack[layer],image_row,images,0,QWEN_VL_HIDDEN);
            image_row += images;
        }
        hidden = qwen_vl_forward(language,block,std::move(selected),report,&cache);
        if (hidden.values.empty()) return false;
        consumed += rows;
    }
    size_t position = 0;
    for (const auto &axis : input.positions) position = std::max(position,axis.back() + 1);
    features = PixVisionFeatures();
    input = PixVLInput();
    std::string pending;
    if (thinking) {
        response = QWEN_CHAT_THINK "\n";
        if (stream && !stream(response)) return false;
    }
    while (true) {
        if (!report(0,"Generating reply")) return false;
        PixTensor last = tensor_slice(hidden,hidden.rows - 1,1,0,QWEN_VL_HIDDEN);
        hidden = PixTensor();
        last = tensor_norm(last,tensor_vector(language,QWEN_CHAT_OUTPUT_NORM),QWEN_VL_EPSILON);
        PixTensor logits = qwen_chat_logits(language,last,report);
        if (logits.values.empty()) return false;
        int id = qwen_chat_sample(logits,seen,sampling,random);
        if (id < 0) {
            error = "Qwen-VL produced invalid chat logits.";
            return false;
        }
        if (id == QWEN_CHAT_END || id == QWEN_CHAT_EOS) return true;
        seen[size_t(id)] = 1;
        std::string piece = tokenizer.decode(id);
        if (piece.empty()) {
            std::string special = tokenizer.decode(id,true);
            if (special == QWEN_CHAT_THINK || special == QWEN_CHAT_THINK_END) piece = special;
        }
        pending += piece;
        size_t complete = qwen_chat_utf8_prefix(pending);
        if (complete) {
            std::string text = pending.substr(0,complete);
            pending.erase(0,complete);
            if (stream && !stream(text)) return false;
            response += text;
        }
        if (cache.rows >= QWEN_VL_CONTEXT) {
            error = "The conversation has reached the model context capacity.";
            return false;
        }
        if (!capacity(cache.rows + 1,1,0)) return false;
        PixVLInput token;
        token.ids = {id};
        token.image_slots = {0};
        for (auto &axis : token.positions) axis = {position};
        position++;
        hidden = qwen_vl_forward(language,token,{},report,&cache);
        if (hidden.values.empty()) return false;
    }
}
