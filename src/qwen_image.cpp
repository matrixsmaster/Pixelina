/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "qwen_image.h"
#include "vulkan_ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

static bool qwen_image_weight(const PixModelFile &file, const std::string &name, const std::vector<uint64_t> &shape, std::string &error)
{
    auto found = file.weights.find(name);
    if (found != file.weights.end() && found->second.shape == shape && weight_bytes(shape,found->second.type)) return true;
    error = "Missing or incompatible Qwen Image 2.1 diffuser tensor: " + name;
    return false;
}

static bool qwen_image_diffuser_valid(const PixModelFile &file, std::string &error)
{
    if (!qwen_image_weight(file,"img_in.weight",{QWEN_IMAGE_CHANNELS,QWEN_IMAGE_HIDDEN},error) || !qwen_image_weight(file,"proj_out.weight",{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_CHANNELS},error)) return false;
    if (!qwen_image_weight(file,QWEN_IMAGE_TEXT_NORM,{QWEN_VL_HIDDEN},error) || !qwen_image_weight(file,"txt_in.in_layer.weight",{QWEN_VL_HIDDEN,QWEN_IMAGE_HIDDEN},error) || !qwen_image_weight(file,"txt_in.out_layer.weight",{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_HIDDEN},error)) return false;
    if (!qwen_image_weight(file,"time_text_embed.timestep_embedder.linear_1.weight",{QWEN_IMAGE_TIME_DIM,QWEN_IMAGE_HIDDEN},error) || !qwen_image_weight(file,"time_text_embed.timestep_embedder.linear_2.weight",{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_HIDDEN},error)) return false;
    if (!qwen_image_weight(file,"modulation.1.weight",{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_MODULATION_PARTS * QWEN_IMAGE_HIDDEN},error) || !qwen_image_weight(file,"norm_out.linear.weight",{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_HIDDEN},error)) return false;
    for (size_t layer = 0; layer < QWEN_IMAGE_LAYERS; layer++) {
        std::string prefix = QWEN_IMAGE_BLOCK_PREFIX + std::to_string(layer) + ".";
        const char* projections[] = {"attn.to_q.weight","attn.to_k.weight","attn.to_v.weight","attn.to_out.0.weight"};
        for (const char* projection : projections) {
            if (!qwen_image_weight(file,prefix + projection,{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_HIDDEN},error)) return false;
        }
        if (!qwen_image_weight(file,prefix + QWEN_IMAGE_QUERY_NORM,{QWEN_IMAGE_HEAD_DIM},error) || !qwen_image_weight(file,prefix + QWEN_IMAGE_KEY_NORM,{QWEN_IMAGE_HEAD_DIM},error)) return false;
        if (!qwen_image_weight(file,prefix + QWEN_IMAGE_FFN_FUSED_WEIGHT,{QWEN_IMAGE_HIDDEN,QWEN_IMAGE_MLP * 2},error) || !qwen_image_weight(file,prefix + "img_mlp.out.weight",{QWEN_IMAGE_MLP,QWEN_IMAGE_HIDDEN},error)) return false;
    }
    return true;
}

bool qwen_image_validate(const PixEditSettings &settings, std::string &error)
{
    if (!edit_settings_valid(settings,error)) return false;
    PixModelFile diffuser,language,vision,vae;
    if (diffuser.open(settings.diffuser,false) || language.open(settings.encoder,false) || vision.open(settings.vision,false) || vae.open(settings.vae,true)) {
        error = QWEN_IMAGE_CONTAINER_ERROR;
        return false;
    }
    return qwen_image_diffuser_valid(diffuser,error) && qwen_vl_validate(language,vision,error) && qwen_vae_validate(vae,error);
}

static bool qwen_image_reference_size(const PixImage &image, int resolution, PixQwenImageShape &shape)
{
    if (!image_valid(image) || resolution <= 0 || uint64_t(resolution) * resolution > PIXELINA_MAX_PIXELS) return false;
    double area = double(resolution) * resolution,ratio = double(image.width) / image.height;
    double raw_width = std::sqrt(area * ratio),raw_height = raw_width / ratio;
    double width = std::nearbyint(raw_width / QWEN_IMAGE_MULTIPLE) * QWEN_IMAGE_MULTIPLE,height = std::nearbyint(raw_height / QWEN_IMAGE_MULTIPLE) * QWEN_IMAGE_MULTIPLE;
    if (width < QWEN_IMAGE_MULTIPLE || height < QWEN_IMAGE_MULTIPLE || width > INT32_MAX || height > INT32_MAX || width * height > PIXELINA_MAX_PIXELS) return false;
    shape = {size_t(width),size_t(height)};
    return true;
}

std::vector<PixImage> qwen_image_prepare(const std::vector<PixImage> &references, int resolution)
{
    if (references.empty()) return {};
    std::vector<PixImage> prepared;
    for (const PixImage &image : references) {
        PixQwenImageShape shape;
        if (!qwen_image_reference_size(image,resolution,shape)) return {};
        PixImage resized = image_resize(image,int(shape.width),int(shape.height),PIX_IMAGE_LANCZOS,PIX_IMAGE_PREMULTIPLIED);
        if (!image_valid(resized)) return {};
        prepared.push_back(std::move(resized));
    }
    return prepared;
}

static void qwen_image_angles(PixQwenLayout &layout, size_t row, size_t frame, int64_t y, int64_t x)
{
    size_t dimensions[] = {QWEN_IMAGE_ROPE_T,QWEN_IMAGE_ROPE_H,QWEN_IMAGE_ROPE_W};
    double positions[] = {double(frame),double(y),double(x)};
    size_t pair = 0;
    for (size_t axis = 0; axis < 3; axis++) {
        for (size_t i = 0; i < dimensions[axis] / 2; i++,pair++) {
            float inverse = float(1.0 / std::pow(QWEN_IMAGE_THETA,double(i * 2) / dimensions[axis]));
            float angle = float(positions[axis]) * inverse;
            layout.rotary.cosine[row * QWEN_IMAGE_HEAD_DIM / 2 + pair] = std::cos(angle);
            layout.rotary.sine[row * QWEN_IMAGE_HEAD_DIM / 2 + pair] = std::sin(angle);
        }
    }
}

PixQwenLayout qwen_image_layout(const PixQwenCondition &condition, const std::vector<PixQwenImageShape> &references, size_t width, size_t height)
{
    if (!width || !height || width > SIZE_MAX / height || condition.hidden.rows != condition.image_slots.size() || condition.hidden.channels != QWEN_VL_HIDDEN) return {};
    PixQwenLayout layout;
    size_t slots = 0,reference_tokens = 0;
    for (unsigned char slot : condition.image_slots) slots += slot != 0;
    for (const PixQwenImageShape &shape : references) {
        if (!shape.width || !shape.height || shape.width > SIZE_MAX / shape.height || shape.width * shape.height % QWEN_VISION_MERGE_PIXELS || reference_tokens > SIZE_MAX - shape.width * shape.height) return {};
        reference_tokens += shape.width * shape.height;
    }
    if (slots > SIZE_MAX / QWEN_VISION_MERGE_PIXELS || reference_tokens != slots * QWEN_VISION_MERGE_PIXELS) return {};
    if (condition.hidden.rows > SIZE_MAX - reference_tokens || condition.hidden.rows + reference_tokens > SIZE_MAX - width * height) return {};
    layout.rows = condition.hidden.rows - slots + reference_tokens + width * height;
    if (layout.rows > SIZE_MAX / QWEN_IMAGE_HEAD_DIM) return {};
    layout.rotary.cosine.resize(layout.rows * QWEN_IMAGE_HEAD_DIM / 2);
    layout.rotary.sine.resize(layout.rotary.cosine.size());
    layout.reference_rows.resize(references.size());
    size_t row = 0,position = 0,reference = 0,source = 0;
    while (source < condition.image_slots.size()) {
        size_t begin = row;
        if (!condition.image_slots[source]) {
            while (source < condition.image_slots.size() && !condition.image_slots[source]) {
                layout.text_sources.push_back(source++);
                layout.text_rows.push_back(row);
                qwen_image_angles(layout,row++,position,int64_t(position),int64_t(position));
                position++;
            }
            layout.attention.push_back({begin,row,row,true});
        } else {
            if (reference >= references.size()) return {};
            const PixQwenImageShape &shape = references[reference];
            size_t count = shape.width * shape.height;
            for (size_t i = 0; i < count / QWEN_VISION_MERGE_PIXELS; i++) {
                if (source >= condition.image_slots.size() || !condition.image_slots[source++]) return {};
            }
            for (size_t pixel = 0; pixel < count; pixel++) {
                layout.reference_rows[reference].push_back(row);
                layout.image_rows.push_back(row);
                int64_t y = int64_t(pixel / shape.width) - int64_t(shape.height - shape.height / 2),x = int64_t(pixel % shape.width) - int64_t(shape.width - shape.width / 2);
                qwen_image_angles(layout,row++,position,y,x);
            }
            position += std::max(shape.width,shape.height);
            layout.attention.push_back({begin,row,row,false});
            reference++;
        }
    }
    if (reference != references.size()) return {};
    layout.target_begin = row;
    for (size_t pixel = 0; pixel < width * height; pixel++) {
        layout.image_rows.push_back(row);
        int64_t y = int64_t(pixel / width) - int64_t(height - height / 2),x = int64_t(pixel % width) - int64_t(width - width / 2);
        qwen_image_angles(layout,row++,position,y,x);
    }
    layout.target_end = row;
    layout.attention.push_back({layout.target_begin,row,row,false});
    return layout;
}

static void qwen_image_modulate(PixTensor &destination, const PixTensor &parameters, size_t target_begin, size_t part, const PixTensor* residual)
{
    if (destination.values.empty() || parameters.rows != 2 || (part + 1) * destination.channels > parameters.channels || (residual && residual->values.size() != destination.values.size())) {
        destination = PixTensor();
        return;
    }
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_qwen_modulate(destination,parameters,target_begin,part,residual);
        return;
    }
#endif
    for (size_t row = 0; row < destination.rows; row++) {
        size_t parameter = (row >= target_begin? 0 : 1) * parameters.channels + part * destination.channels;
        for (size_t c = 0; c < destination.channels; c++) {
            size_t index = row * destination.channels + c;
            if (residual) destination.values[index] += residual->values[index] * std::tanh(parameters.values[parameter + c]);
            else destination.values[index] *= 1 + parameters.values[parameter + c];
        }
    }
}

static PixTensor qwen_image_timestep(const PixModelFile &diffuser, float sigma)
{
    PixTensor input(2,QWEN_IMAGE_TIME_DIM);
    for (size_t row = 0; row < input.rows; row++) {
        for (size_t pair = 0; pair < QWEN_IMAGE_TIME_DIM / 2; pair++) {
            float frequency = std::exp(float(-std::log(QWEN_IMAGE_TIME_PERIOD) * pair / (QWEN_IMAGE_TIME_DIM / 2)));
            float angle = (row? 0 : sigma * QWEN_IMAGE_TIME_SCALE) * frequency;
            input.values[row * input.channels + pair] = std::cos(angle);
            input.values[row * input.channels + pair + QWEN_IMAGE_TIME_DIM / 2] = std::sin(angle);
        }
    }
    PixTensor hidden = tensor_linear(diffuser,"time_text_embed.timestep_embedder.linear_1",input,false);
    tensor_silu(hidden);
    return tensor_linear(diffuser,"time_text_embed.timestep_embedder.linear_2",hidden,false);
}

static PixTensor qwen_image_cached_kv(const PixTensor &prefix, const PixTensor &target)
{
    if (prefix.values.empty() || target.values.empty() || prefix.channels != target.channels || prefix.rows > SIZE_MAX - target.rows) return {};
    PixTensor combined(prefix.rows + target.rows,prefix.channels);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_copy_rows(prefix,combined,prefix.rows);
        vulkan_copy_rows(target,combined,target.rows,0,0,prefix.rows);
        return combined;
    }
#endif
    std::copy(prefix.values.begin(),prefix.values.end(),combined.values.begin());
    std::copy(target.values.begin(),target.values.end(),combined.values.begin() + prefix.values.size());
    return combined;
}

static PixTensor qwen_image_diffuse_run(const PixModelFile &diffuser, const PixTensor &target, const std::vector<PixTensor> &references, const PixQwenCondition &condition, const PixQwenLayout &layout, float sigma, const InferenceProgress &progress, PixQwenKVCache* cache)
{
    bool reuse = cache && cache->ready;
    size_t offset = reuse? layout.target_begin : 0;
    size_t target_begin = layout.target_begin - offset;
    PixTensor hidden(layout.rows - offset,QWEN_IMAGE_HIDDEN);
    if (!reuse) {
        std::vector<float> zero_center = tensor_vector(diffuser,QWEN_IMAGE_TEXT_NORM);
        for (float &value : zero_center) value += 1;
        PixTensor text = tensor_gather(condition.hidden,layout.text_sources);
        text = tensor_norm(text,zero_center,QWEN_IMAGE_EPSILON);
        text = tensor_linear(diffuser,"txt_in.in_layer",text,false);
        tensor_gelu(text,true);
        text = tensor_linear(diffuser,"txt_in.out_layer",text,false);
        if (!tensor_scatter(hidden,text,layout.text_rows)) return {};
        text = PixTensor();
        for (size_t i = 0; i < references.size(); i++) {
            if (references[i].channels != QWEN_IMAGE_CHANNELS || references[i].rows != layout.reference_rows[i].size()) return {};
            PixTensor image = tensor_linear(diffuser,QWEN_IMAGE_INPUT_PROJECTION,references[i],false);
            if (!tensor_scatter(hidden,image,layout.reference_rows[i])) return {};
        }
    }
    PixTensor image = tensor_linear(diffuser,QWEN_IMAGE_INPUT_PROJECTION,target,false);
    std::vector<size_t> target_rows(target.rows);
    std::iota(target_rows.begin(),target_rows.end(),target_begin);
    if (!tensor_scatter(hidden,image,target_rows)) return {};
    image = PixTensor();
    PixTensor time = qwen_image_timestep(diffuser,sigma);
    tensor_silu(time);
    PixTensor modulation = tensor_linear(diffuser,"modulation.1",time,false);
    const PixRotary &rotary = reuse? cache->rotary : layout.rotary;
    std::vector<PixAttentionRange> target_attention = {{0,target.rows,layout.rows,false}};
    const std::vector<PixAttentionRange> &attention = reuse? target_attention : layout.attention;
    for (size_t layer = 0; layer < QWEN_IMAGE_LAYERS; layer++) {
        std::string prefix = QWEN_IMAGE_BLOCK_PREFIX + std::to_string(layer) + ".";
        PixTensor normalized = tensor_norm(hidden,{},QWEN_IMAGE_EPSILON,true);
        qwen_image_modulate(normalized,modulation,target_begin,0,nullptr);
        PixTensor q = tensor_linear(diffuser,prefix + "attn.to_q",normalized,false),k = tensor_linear(diffuser,prefix + "attn.to_k",normalized,false),v = tensor_linear(diffuser,prefix + "attn.to_v",normalized,false);
        normalized = PixTensor();
        q.rows *= QWEN_IMAGE_HEADS;
        q.channels = QWEN_IMAGE_HEAD_DIM;
        k.rows *= QWEN_IMAGE_HEADS;
        k.channels = QWEN_IMAGE_HEAD_DIM;
        q = tensor_norm(q,tensor_vector(diffuser,prefix + QWEN_IMAGE_QUERY_NORM),QWEN_IMAGE_EPSILON);
        k = tensor_norm(k,tensor_vector(diffuser,prefix + QWEN_IMAGE_KEY_NORM),QWEN_IMAGE_EPSILON);
        q.rows = k.rows = hidden.rows;
        q.channels = k.channels = QWEN_IMAGE_HIDDEN;
        tensor_rope(q,QWEN_IMAGE_HEADS,rotary.cosine,rotary.sine,true);
        tensor_rope(k,QWEN_IMAGE_HEADS,rotary.cosine,rotary.sine,true);
        if (cache) {
            PixQwenKVLayer &saved = cache->layers[layer];
            if (reuse) {
                k = qwen_image_cached_kv(saved.key,k);
                v = qwen_image_cached_kv(saved.value,v);
            } else {
                saved.key = tensor_slice(k,0,layout.target_begin,0,k.channels);
                saved.value = tensor_slice(v,0,layout.target_begin,0,v.channels);
                if (saved.key.values.empty() || saved.value.values.empty()) return {};
            }
        }
        PixTensor attended = tensor_attention(q,k,v,QWEN_IMAGE_HEADS,QWEN_IMAGE_HEADS,attention,offset);
        q = k = v = PixTensor();
        PixTensor projected = tensor_linear(diffuser,prefix + "attn.to_out.0",attended,false);
        attended = PixTensor();
        qwen_image_modulate(hidden,modulation,target_begin,1,&projected);
        projected = PixTensor();
        normalized = tensor_norm(hidden,{},QWEN_IMAGE_EPSILON,true);
        qwen_image_modulate(normalized,modulation,target_begin,2,nullptr);
        PixTensor gate = tensor_linear_rows(diffuser,prefix + QWEN_IMAGE_FFN_FUSED,normalized,0,QWEN_IMAGE_MLP),up = tensor_linear_rows(diffuser,prefix + QWEN_IMAGE_FFN_FUSED,normalized,QWEN_IMAGE_MLP,QWEN_IMAGE_MLP);
        normalized = PixTensor();
        tensor_silu(gate);
        tensor_gate(gate,up);
        up = PixTensor();
        projected = tensor_linear(diffuser,prefix + "img_mlp.out",gate,false);
        gate = PixTensor();
        qwen_image_modulate(hidden,modulation,target_begin,3,&projected);
        if (hidden.values.empty()) return {};
        if (progress && !progress(float(layer + 1) / QWEN_IMAGE_LAYERS,"Diffuser block " + std::to_string(layer + 1))) return {};
    }
    PixTensor output = tensor_slice(hidden,target_begin,target.rows,0,hidden.channels);
    hidden = PixTensor();
    output = tensor_norm(output,{},QWEN_IMAGE_EPSILON,true);
    PixTensor scale = tensor_linear(diffuser,"norm_out.linear",time,false);
    qwen_image_modulate(output,scale,0,0,nullptr);
    return tensor_linear(diffuser,"proj_out",output,false);
}

PixTensor qwen_image_diffuse(const PixModelFile &diffuser, const PixTensor &target, const std::vector<PixTensor> &references, const PixQwenCondition &condition, const PixQwenLayout &layout, float sigma, const InferenceProgress &progress, PixQwenKVCache* cache)
{
    bool valid = layout.target_begin && layout.target_end == layout.rows && layout.target_begin < layout.target_end && layout.rows <= SIZE_MAX / QWEN_IMAGE_HEAD_DIM && layout.rotary.cosine.size() == layout.rows * QWEN_IMAGE_HEAD_DIM / 2 && layout.rotary.sine.size() == layout.rotary.cosine.size() && references.size() == layout.reference_rows.size() && target.channels == QWEN_IMAGE_CHANNELS && target.rows == layout.target_end - layout.target_begin && std::isfinite(sigma);
    if (cache && cache->ready) {
        // Cache owners must keep the bound job inputs immutable until cache destruction.
        valid = valid && cache->model == &diffuser && cache->layout == &layout && cache->condition == &condition && cache->references == &references && cache->layers.size() == QWEN_IMAGE_LAYERS;
        for (const PixQwenKVLayer &layer : cache->layers) {
            valid = valid && layer.key.rows == layout.target_begin && layer.value.rows == layout.target_begin && layer.key.channels == QWEN_IMAGE_HIDDEN && layer.value.channels == QWEN_IMAGE_HIDDEN;
#ifdef PIXELINA_WITH_VULKAN
            valid = valid && layer.key.values.get_allocator().state == vulkan_active_state() && layer.value.values.get_allocator().state == vulkan_active_state();
#endif
        }
    }
    if (!valid) {
        if (cache) *cache = PixQwenKVCache();
        return {};
    }
    if (cache && !cache->ready) {
        *cache = PixQwenKVCache();
        cache->model = &diffuser;
        cache->layout = &layout;
        cache->condition = &condition;
        cache->references = &references;
        cache->layers.resize(QWEN_IMAGE_LAYERS);
        size_t begin = layout.target_begin * QWEN_IMAGE_HEAD_DIM / 2;
        cache->rotary.cosine.assign(layout.rotary.cosine.begin() + begin,layout.rotary.cosine.end());
        cache->rotary.sine.assign(layout.rotary.sine.begin() + begin,layout.rotary.sine.end());
    }
    PixTensor output = qwen_image_diffuse_run(diffuser,target,references,condition,layout,sigma,progress,cache);
    if (cache) {
        if (output.values.empty()) *cache = PixQwenKVCache();
        else cache->ready = true;
    }
    return output;
}

std::vector<float> qwen_image_sigmas(int steps, size_t target_tokens)
{
    if (steps < 2 || steps > PIXELINA_EDIT_MAX_STEPS || !target_tokens || size_t(steps) > SIZE_MAX / sizeof(float) - 1) return {};
    double mu = QWEN_IMAGE_SIGMA_BASE_SHIFT + (double(target_tokens) - QWEN_IMAGE_SIGMA_BASE_LENGTH) * (QWEN_IMAGE_SIGMA_MAX_SHIFT - QWEN_IMAGE_SIGMA_BASE_SHIFT) / (QWEN_IMAGE_SIGMA_MAX_LENGTH - QWEN_IMAGE_SIGMA_BASE_LENGTH);
    float shift = float(std::exp(mu));
    if (!std::isfinite(shift) || shift <= 0) return {};
    std::vector<float> sigmas(size_t(steps) + 1);
    for (int i = 0; i < steps; i++) {
        float raw = float(double(steps - i) / steps);
        sigmas[size_t(i)] = shift / (shift + (1 / raw - 1));
    }
    float scale = (1 - sigmas[size_t(steps - 1)]) / (1 - QWEN_IMAGE_SIGMA_TERMINAL);
    if (!std::isfinite(scale) || scale <= 0) return {};
    for (int i = 0; i < steps; i++) {
        sigmas[size_t(i)] = 1 - (1 - sigmas[size_t(i)]) / scale;
        if (!std::isfinite(sigmas[size_t(i)]) || sigmas[size_t(i)] <= 0 || (i && sigmas[size_t(i)] >= sigmas[size_t(i - 1)])) return {};
    }
    sigmas.back() = 0;
    return sigmas;
}

void qwen_image_euler(PixTensor &target, const PixTensor &flow, float sigma, float next_sigma)
{
    if (target.rows != flow.rows || target.channels != flow.channels || flow.values.empty() || !std::isfinite(sigma) || !std::isfinite(next_sigma)) {
        target = PixTensor();
        return;
    }
    float delta = next_sigma - sigma;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_pointwise(VK_OP_EULER,target,&flow,nullptr,0,delta);
        return;
    }
#endif
    for (size_t i = 0; i < target.values.size(); i++) target.values[i] += delta * flow.values[i];
}

static bool qwen_memory_multiply(uint64_t a, uint64_t b, uint64_t &result)
{
    if (b && a > UINT64_MAX / b) return false;
    result = a * b;
    return true;
}

static bool qwen_memory_add(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a > UINT64_MAX - b) return false;
    result = a + b;
    return true;
}

static bool qwen_memory_shape(const PixQwenImageShape &shape, uint64_t &rows)
{
    if (!shape.width || !shape.height || !qwen_memory_multiply(shape.width,shape.height,rows)) return false;
    return rows <= PIXELINA_MAX_PIXELS;
}

bool qwen_image_memory_plan(const PixQwenMemoryInput &input, const PixQwenMemoryLimits &limits, PixQwenMemoryPlan &plan, std::string &error)
{
    error = "Invalid or overflowing Qwen Image memory dimensions.";
    if (!input.language_tokens || input.language_tokens > QWEN_VL_CONTEXT || !input.cpu_threads || input.references.size() != input.vision.size()) return false;
    uint64_t target,reference_rows = 0,vision_rows = 0,max_vision = 0,max_reference = 0;
    if (!qwen_memory_shape(input.target,target)) return false;
    for (size_t i = 0; i < input.references.size(); i++) {
        uint64_t latent,vision;
        if (!qwen_memory_shape(input.references[i],latent) || !qwen_memory_multiply(input.vision[i].width,input.vision[i].height,vision) || !vision || input.vision[i].temporal != 1) return false;
        if (vision != latent) {
            error = "Vision resizing changes the reference grid; increase the reference resolution.";
            return false;
        }
        if (!qwen_memory_add(reference_rows,latent,reference_rows) || !qwen_memory_add(vision_rows,vision,vision_rows)) return false;
        max_reference = std::max(max_reference,latent);
        max_vision = std::max(max_vision,vision);
    }
    if (reference_rows % QWEN_VISION_MERGE_PIXELS || input.language_tokens < reference_rows / QWEN_VISION_MERGE_PIXELS) return false;
    uint64_t joint;
    if (!qwen_memory_add(input.language_tokens - reference_rows / QWEN_VISION_MERGE_PIXELS,reference_rows,joint) || !qwen_memory_add(joint,target,joint)) return false;
    uint64_t patches,features,condition,latents,rotary;
    if (!qwen_memory_multiply(vision_rows,3 * QWEN_VISION_TEMPORAL_PATCH * QWEN_VISION_PATCH * QWEN_VISION_PATCH * sizeof(float),patches)) return false;
    if (!qwen_memory_multiply(vision_rows / QWEN_VISION_MERGE_PIXELS,QWEN_VL_HIDDEN * (QWEN_VISION_DEEPSTACK + 1) * sizeof(float),features) || !qwen_memory_multiply(input.language_tokens,QWEN_VL_HIDDEN * sizeof(float),condition)) return false;
    if (!qwen_memory_add(reference_rows,target,latents) || !qwen_memory_multiply(latents,QWEN_IMAGE_CHANNELS * sizeof(float),latents) || !qwen_memory_multiply(joint,QWEN_IMAGE_HEAD_DIM * sizeof(float),rotary)) return false;
    std::array<uint64_t,QWEN_IMAGE_STAGE_COUNT> activation = {},largest = {},retained = {},scratch = {};
    uint64_t vision_width = std::max(uint64_t(QWEN_VISION_HIDDEN) * QWEN_MEMORY_VISION_MAPS,uint64_t(QWEN_VISION_HIDDEN) * 3 + QWEN_VISION_MLP);
    if (!qwen_memory_multiply(max_vision,vision_width * sizeof(float),activation[QWEN_STAGE_VISION]) || !qwen_memory_multiply(max_vision,QWEN_VISION_MLP * sizeof(float),largest[QWEN_STAGE_VISION])) return false;
    largest[QWEN_STAGE_VISION] = std::max(largest[QWEN_STAGE_VISION],features / (QWEN_VISION_DEEPSTACK + 1));
    if (!qwen_memory_add(patches,features,retained[QWEN_STAGE_VISION])) return false;
    uint64_t learned = QWEN_VISION_POSITIONS * QWEN_VISION_HIDDEN * sizeof(float);
    if (!qwen_memory_add(activation[QWEN_STAGE_VISION],learned,activation[QWEN_STAGE_VISION])) return false;
    uint64_t language_width = std::max(uint64_t(QWEN_VL_HIDDEN) * QWEN_MEMORY_LANGUAGE_ATTN_MAPS + uint64_t(QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM) * 2,uint64_t(QWEN_VL_HIDDEN) * QWEN_MEMORY_FFN_MAPS + uint64_t(QWEN_VL_MLP) * 2);
    if (!qwen_memory_multiply(input.language_tokens,language_width * sizeof(float),activation[QWEN_STAGE_LANGUAGE]) || !qwen_memory_multiply(input.language_tokens,QWEN_VL_MLP * sizeof(float),largest[QWEN_STAGE_LANGUAGE])) return false;
    retained[QWEN_STAGE_LANGUAGE] = features;
    // At the first full-resolution encoder residual, three base-width maps coexist with its quarter-size shortcut.
    uint64_t encoder_pixels;
    if (!qwen_memory_multiply(max_reference,QWEN_VAE_FACTOR * QWEN_VAE_FACTOR,encoder_pixels) || !qwen_memory_multiply(encoder_pixels,(QWEN_VAE_ENCODER_BASE * QWEN_MEMORY_ENCODER_BASE_MAPS + QWEN_VAE_ENCODER_BASE / 4) * sizeof(float),activation[QWEN_STAGE_ENCODE])) return false;
    if (!qwen_memory_multiply(encoder_pixels,QWEN_VAE_ENCODER_BASE * sizeof(float),largest[QWEN_STAGE_ENCODE]) || !qwen_memory_add(condition,latents,retained[QWEN_STAGE_ENCODE])) return false;
    uint64_t diffuse_width = std::max(uint64_t(QWEN_IMAGE_HIDDEN) * QWEN_MEMORY_DIFFUSION_ATTN_MAPS,uint64_t(QWEN_IMAGE_HIDDEN) * QWEN_MEMORY_FFN_MAPS + uint64_t(QWEN_IMAGE_MLP) * 2);
    if (!qwen_memory_multiply(joint,diffuse_width * sizeof(float),activation[QWEN_STAGE_DIFFUSION]) || !qwen_memory_multiply(joint,QWEN_IMAGE_MLP * sizeof(float),largest[QWEN_STAGE_DIFFUSION])) return false;
    if (!qwen_memory_add(condition,latents,retained[QWEN_STAGE_DIFFUSION]) || !qwen_memory_add(retained[QWEN_STAGE_DIFFUSION],rotary,retained[QWEN_STAGE_DIFFUSION])) return false;
    // The final spatial upsample retains its two-base-width shortcut while two convolution maps coexist.
    uint64_t decoder_pixels;
    if (!qwen_memory_multiply(target,QWEN_VAE_FACTOR * QWEN_VAE_FACTOR,decoder_pixels) || !qwen_memory_multiply(decoder_pixels,QWEN_VAE_DECODER_BASE * QWEN_MEMORY_DECODER_BASE_MAPS * sizeof(float),activation[QWEN_STAGE_DECODE])) return false;
    if (!qwen_memory_multiply(decoder_pixels,QWEN_VAE_DECODER_BASE * 2 * sizeof(float),largest[QWEN_STAGE_DECODE]) || !qwen_memory_multiply(target,QWEN_IMAGE_CHANNELS * sizeof(float),retained[QWEN_STAGE_DECODE])) return false;
    uint64_t worker_scores,linear_scratch,quantized_scratch;
    if (!qwen_memory_multiply(std::max({joint,uint64_t(input.language_tokens),max_vision}),input.cpu_threads,worker_scores) || !qwen_memory_multiply(worker_scores,sizeof(float),worker_scores)) return false;
    linear_scratch = uint64_t(COMPUTE_WEIGHT_ROWS) * QWEN_IMAGE_MLP * sizeof(float);
    uint64_t matrix_scratch;
    if (!qwen_memory_add(worker_scores,linear_scratch,matrix_scratch)) return false;
    scratch[QWEN_STAGE_VISION] = matrix_scratch;
    uint64_t quantized_row = uint64_t(QWEN_IMAGE_MLP) / MODEL_QUANT_BLOCK * MODEL_Q8_K_BYTES;
    if (!qwen_memory_multiply(input.language_tokens,quantized_row,quantized_scratch) || !qwen_memory_add(matrix_scratch,quantized_scratch,scratch[QWEN_STAGE_LANGUAGE])) return false;
    if (!qwen_memory_multiply(joint,quantized_row,quantized_scratch) || !qwen_memory_add(matrix_scratch,quantized_scratch,scratch[QWEN_STAGE_DIFFUSION])) return false;
    size_t convolution_rows = limits.device_bytes? VK_CONV_TILE_PIXELS : COMPUTE_TILE_ROWS;
    scratch[QWEN_STAGE_ENCODE] = uint64_t(convolution_rows) * QWEN_VAE_ENCODER_BASE * QWEN_VAE_MAX_MULTIPLIER * QWEN_VAE_KERNEL * QWEN_VAE_KERNEL * sizeof(float);
    scratch[QWEN_STAGE_DECODE] = uint64_t(convolution_rows) * QWEN_VAE_DECODER_BASE * QWEN_VAE_MAX_MULTIPLIER * QWEN_VAE_KERNEL * QWEN_VAE_KERNEL * sizeof(float);
    scratch[QWEN_STAGE_ENCODE] += linear_scratch;
    scratch[QWEN_STAGE_DECODE] += linear_scratch;
    PixQwenMemoryPlan result;
    for (size_t i = 0; i < QWEN_IMAGE_STAGE_COUNT; i++) {
        PixQwenStageMemory stage;
        stage.stage = PixQwenStage(i);
        // Text-only generation executes neither vision nor reference encoding.
        if (input.references.empty() && (i == QWEN_STAGE_VISION || i == QWEN_STAGE_ENCODE)) {
            result.stages.push_back(stage);
            continue;
        }
        stage.retained_bytes = retained[i];
        if (!qwen_memory_add(activation[i],retained[i],stage.activation_bytes)) return false;
        stage.scratch_bytes = scratch[i];
        stage.largest_binding_bytes = std::max({largest[i],input.largest_weight[i],scratch[i]});
        if (stage.largest_binding_bytes > limits.max_buffer_bytes) {
            error = "Qwen Image stage " + std::to_string(i + 1) + " exceeds the device storage-buffer range.";
            return false;
        }
        uint64_t runtime;
        if (!qwen_memory_add(stage.activation_bytes,stage.scratch_bytes,runtime)) return false;
        if (limits.device_bytes) {
            if (runtime >= limits.device_bytes || input.largest_weight[i] > limits.device_bytes - runtime) {
                error = "Insufficient device memory for Qwen Image stage " + std::to_string(i + 1) + ".";
                return false;
            }
            stage.packed_cache_bytes = std::min(input.mapped_weights[i],limits.device_bytes - runtime);
            if (!qwen_memory_add(runtime,stage.packed_cache_bytes,stage.device_peak_bytes)) return false;
        }
        result.stages.push_back(stage);
    }
    if (input.kv_cache) {
        uint64_t cache_bytes,extra,target_rotary;
        if (!qwen_memory_multiply(joint - target,uint64_t(2) * QWEN_IMAGE_LAYERS * QWEN_IMAGE_HIDDEN * sizeof(float),cache_bytes) || !qwen_memory_multiply(target,QWEN_IMAGE_HEAD_DIM * sizeof(float),target_rotary) || !qwen_memory_add(cache_bytes,target_rotary,extra)) return false;
        PixQwenStageMemory cached = result.stages[QWEN_STAGE_DIFFUSION];
        // Prefill still runs the full graph while all layers' prefix storage accumulates.
        if (!qwen_memory_add(cached.activation_bytes,extra,cached.activation_bytes) || !qwen_memory_add(cached.retained_bytes,extra,cached.retained_bytes)) return false;
        uint64_t reduce = 0;
        if (limits.device_bytes) {
            if (!qwen_memory_add(cached.device_peak_bytes,extra,cached.device_peak_bytes)) return false;
            if (cached.device_peak_bytes > limits.device_bytes) reduce = cached.device_peak_bytes - limits.device_bytes;
        }
        bool admitted = !limits.device_bytes || (reduce <= cached.packed_cache_bytes && input.largest_weight[QWEN_STAGE_DIFFUSION] <= cached.packed_cache_bytes - reduce);
        if (admitted) {
            cached.packed_cache_bytes -= reduce;
            cached.device_peak_bytes -= reduce;
            result.stages[QWEN_STAGE_DIFFUSION] = cached;
            result.kv_cache_bytes = cache_bytes;
        }
    }
    plan = std::move(result);
    error.clear();
    return true;
}

static uint64_t qwen_image_largest_weight(const PixModelFile &model, PixQwenStage stage)
{
    uint64_t largest = 0;
    for (const auto &entry : model.weights) {
        const PixWeight &weight = entry.second;
        if (stage == QWEN_STAGE_LANGUAGE && (weight.name == "output.weight" || weight.name == "token_embd.weight")) continue;
        uint64_t bytes = weight.bytes;
        if (stage == QWEN_STAGE_DIFFUSION && weight.name.find(QWEN_IMAGE_FFN_FUSED_WEIGHT) != std::string::npos) bytes /= 2;
        largest = std::max(largest,bytes);
    }
    return largest;
}

PixImage qwen_image_edit(const PixEditSettings &settings, const std::vector<PixImage> &references, const std::string &prompt, const InferenceProgress &progress, int &error)
{
    error = INFERENCE_ERROR_SETTINGS;
    std::string detail;
    bool cancelled = false;
    auto report = [&](float fraction, const std::string &stage) {
        if (cancelled) return false;
        if (progress && !progress(fraction,stage)) cancelled = true;
        return !cancelled;
    };
    if (!edit_settings_valid(settings,detail)) {
        report(0,detail);
        return {};
    }
    if (references.size() > QWEN_IMAGE_MAX_REFERENCES) {
        report(0,"Too many reference images; the maximum is " + std::to_string(QWEN_IMAGE_MAX_REFERENCES) + ".");
        return {};
    }
    PixQwenMemoryInput memory;
    memory.kv_cache = true;
    memory.target = {size_t(settings.width / QWEN_VAE_FACTOR),size_t(settings.height / QWEN_VAE_FACTOR)};
    memory.cpu_threads = cpu_thread_count();
    for (const PixImage &reference : references) if (!image_valid(reference)) return {};
    PixQwenMemoryLimits limits;
    if (!report(0,"Inspecting Qwen Image model contracts")) {
        error = INFERENCE_ERROR_STOPPED;
        return {};
    }
    PixModelFile diffuser,language,vision,vae;
    error = INFERENCE_ERROR_MODEL;
    if (diffuser.open(settings.diffuser,false) || language.open(settings.encoder,false) || vision.open(settings.vision,false) || vae.open(settings.vae,true)) {
        report(0,QWEN_IMAGE_CONTAINER_ERROR);
        return {};
    }
    if (!qwen_image_diffuser_valid(diffuser,detail) || !qwen_vl_validate(language,vision,detail) || !qwen_vae_validate(vae,detail)) {
        report(0,detail);
        return {};
    }
    memory.mapped_weights = {vision.size,language.size,vae.size,diffuser.size,vae.size};
    memory.largest_weight = {qwen_image_largest_weight(vision,QWEN_STAGE_VISION),qwen_image_largest_weight(language,QWEN_STAGE_LANGUAGE),qwen_image_largest_weight(vae,QWEN_STAGE_ENCODE),qwen_image_largest_weight(diffuser,QWEN_STAGE_DIFFUSION),qwen_image_largest_weight(vae,QWEN_STAGE_DECODE)};
    PixTokenz tokenizer;
    if (!tokenizer.load(language)) return {};
    diffuser.close();
    vision.close();
    language.close();
    vae.close();
    error = INFERENCE_ERROR_SETTINGS;
    for (const PixImage &reference : references) {
        PixQwenImageShape shape;
        if (!qwen_image_reference_size(reference,settings.reference_resolution,shape)) {
            report(0,"Cannot derive the aspect-preserving reference grid.");
            return {};
        }
        uint64_t pixels = shape.width * shape.height;
        if (pixels < QWEN_VISION_MIN_PIXELS || pixels > QWEN_VISION_MAX_PIXELS) {
            report(0,"Reference resolution produces a grid outside the vision processor's supported area.");
            return {};
        }
    }
    std::vector<PixImage> prepared = qwen_image_prepare(references,settings.reference_resolution);
    if (prepared.size() != references.size()) {
        report(0,"Cannot derive or resize the aspect-preserving reference grid.");
        return {};
    }
    PixVLInput input = qwen_vl_prepare(tokenizer,prompt,prepared,false);
    error = INFERENCE_ERROR_PROMPT;
    if (input.ids.empty()) {
        report(0,"Qwen3-VL prompt/tokenizer or processor input is invalid or exceeds the context.");
        return {};
    }
    memory.language_tokens = input.ids.size();
    memory.vision = input.grids;
    for (const PixImage &image : prepared) memory.references.push_back({size_t(image.width / QWEN_VAE_FACTOR),size_t(image.height / QWEN_VAE_FACTOR)});
#ifdef PIXELINA_WITH_VULKAN
    VulkanSession session;
    if (settings.backend == PIXELINA_VULKAN && (!session.initialize() || !session.memory_limits(limits.device_bytes,limits.max_buffer_bytes))) {
        error = INFERENCE_ERROR_VULKAN;
        return {};
    }
#else
    if (settings.backend == PIXELINA_VULKAN) {
        error = INFERENCE_ERROR_VULKAN;
        return {};
    }
#endif
    PixQwenMemoryPlan plan;
    error = INFERENCE_ERROR_MEMORY;
    if (!qwen_image_memory_plan(memory,limits,plan,detail)) {
        report(0,detail);
        return {};
    }
    // Token/grid planning allocates no patch or hidden-state buffers before admission.
    input = qwen_vl_prepare(tokenizer,prompt,prepared);
    if (input.patches.size() != prepared.size()) {
        error = INFERENCE_ERROR_PROMPT;
        report(0,"Cannot pack the admitted vision processor inputs.");
        return {};
    }
    auto stage = [&](PixQwenStage index) {
#ifdef PIXELINA_WITH_VULKAN
        if (settings.backend == PIXELINA_VULKAN) {
            const PixQwenStageMemory &budget = plan.stages[size_t(index)];
            session.packed_weight_budget = budget.packed_cache_bytes;
            if (!session.prepare_stage(budget.activation_bytes,budget.scratch_bytes)) {
                error = INFERENCE_ERROR_MEMORY;
                report(0,"Cannot reserve the admitted Qwen Image stage memory.");
                return false;
            }
        }
#else
        (void)index;
#endif
        if (cancelled) error = INFERENCE_ERROR_STOPPED;
        return !cancelled;
    };
    error = INFERENCE_ERROR_MODEL;
    PixVisionFeatures features;
    if (!prepared.empty()) {
        if (!stage(QWEN_STAGE_VISION) || vision.open(settings.vision,false)) return {};
        features = qwen_vl_vision(vision,input,report);
        if (features.merged.values.empty()) {
            error = cancelled? INFERENCE_ERROR_STOPPED : INFERENCE_ERROR_NUMERICAL;
            return {};
        }
#ifdef PIXELINA_WITH_VULKAN
        if (!vulkan_synchronize()) return {};
#endif
        vision.close();
    }
    input.patches.clear();
    if (!stage(QWEN_STAGE_LANGUAGE) || language.open(settings.encoder,false)) return {};
    PixQwenCondition condition = qwen_vl_language(language,input,std::move(features),report);
    if (condition.hidden.values.empty()) {
        error = cancelled? INFERENCE_ERROR_STOPPED : INFERENCE_ERROR_NUMERICAL;
        return {};
    }
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return {};
#endif
    language.close();
    input = PixVLInput();
    std::vector<PixTensor> fixed;
    if (!prepared.empty() && (!stage(QWEN_STAGE_ENCODE) || vae.open(settings.vae,true))) return {};
    for (size_t i = 0; i < prepared.size(); i++) {
        auto encode_progress = [&](float fraction, const std::string &name) {
            float within = (fraction - QWEN_VAE_ENCODE_BEGIN) / QWEN_VAE_ENCODE_SPAN;
            return report(QWEN_VAE_ENCODE_BEGIN + QWEN_VAE_ENCODE_SPAN * (float(i) + within) / prepared.size(),name);
        };
        PixTensor encoded = qwen_vae_encode(vae,prepared[i],encode_progress);
        if (encoded.values.empty()) {
            error = cancelled? INFERENCE_ERROR_STOPPED : INFERENCE_ERROR_NUMERICAL;
            return {};
        }
        fixed.push_back(std::move(encoded));
    }
    prepared.clear();
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return {};
#endif
    vae.close();
    PixTensor target = inference_noise(memory.target.width * memory.target.height,settings.seed,QWEN_IMAGE_CHANNELS);
    PixQwenLayout layout = qwen_image_layout(condition,memory.references,memory.target.width,memory.target.height);
    std::vector<float> sigmas = qwen_image_sigmas(settings.steps,target.rows);
    if (!layout.rows || sigmas.empty()) {
        error = INFERENCE_ERROR_SETTINGS;
        report(QWEN_IMAGE_DIFFUSION_BEGIN,"Reference slots disagree with the latent grids, or the Edit schedule is unsupported.");
        return {};
    }
    if (!stage(QWEN_STAGE_DIFFUSION) || diffuser.open(settings.diffuser,false)) return {};
    PixQwenKVCache cache;
    if (!report(QWEN_IMAGE_DIFFUSION_BEGIN,plan.kv_cache_bytes? "Diffusion with F32 prefix K/V caching" : "Diffusion without prefix K/V caching (memory budget)")) {
        error = INFERENCE_ERROR_STOPPED;
        return {};
    }
    for (int step = 0; step < settings.steps; step++) {
        auto block_progress = [&](float fraction, const std::string &name) {
            return report(QWEN_IMAGE_DIFFUSION_BEGIN + QWEN_IMAGE_DIFFUSION_SPAN * (step + fraction) / settings.steps,"Step " + std::to_string(step + 1) + "/" + std::to_string(settings.steps) + ": " + name);
        };
        float model_time = (sigmas[size_t(step)] * QWEN_IMAGE_TIME_SCALE) / QWEN_IMAGE_TIME_SCALE;
        PixTensor flow = qwen_image_diffuse(diffuser,target,fixed,condition,layout,model_time,block_progress,plan.kv_cache_bytes? &cache : nullptr);
        qwen_image_euler(target,flow,sigmas[size_t(step)],sigmas[size_t(step + 1)]);
#ifdef PIXELINA_WITH_VULKAN
        if (!vulkan_synchronize()) return {};
#endif
        if (target.values.empty()) {
            error = cancelled? INFERENCE_ERROR_STOPPED : INFERENCE_ERROR_NUMERICAL;
            return {};
        }
        for (float value : target.values) {
            if (!std::isfinite(value)) {
                error = INFERENCE_ERROR_NUMERICAL;
                return {};
            }
        }
    }
    cache = PixQwenKVCache();
    diffuser.close();
    fixed.clear();
    condition = PixQwenCondition();
    layout = PixQwenLayout();
    if (!stage(QWEN_STAGE_DECODE) || vae.open(settings.vae,true)) return {};
    PixImage result = qwen_vae_decode(vae,std::move(target),memory.target.width,memory.target.height,report);
    if (!image_valid(result) || result.width != settings.width || result.height != settings.height || result.alpha.empty()) {
        error = cancelled? INFERENCE_ERROR_STOPPED : INFERENCE_ERROR_NUMERICAL;
        return {};
    }
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return {};
    vae.close();
    if (settings.backend == PIXELINA_VULKAN) {
        session.packed_weight_budget = 1;
        if (!session.prepare_stage(0,0)) return {};
    }
#endif
    error = 0;
    return result;
}
