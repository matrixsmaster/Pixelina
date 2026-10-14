/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "qwen_vl.h"
#include <algorithm>
#include <cmath>
#include <numeric>

static bool qwen_vl_weight(const PixModelFile &file, const std::string &name, const std::vector<uint64_t> &shape, std::string &error)
{
    auto found = file.weights.find(name);
    if (found != file.weights.end() && found->second.shape == shape && weight_bytes(shape,found->second.type)) return true;
    error = "Missing or incompatible Qwen3-VL tensor: " + name;
    return false;
}

bool qwen_vl_validate(const PixModelFile &language, const PixModelFile &vision, std::string &error)
{
    auto architecture = language.metadata.find("general.architecture");
    if (architecture == language.metadata.end() || architecture->second.scalar != "qwen3vl") {
        error = "Select a Qwen3-VL language GGUF.";
        return false;
    }
    const char* language_keys[] = {"qwen3vl.block_count","qwen3vl.context_length","qwen3vl.embedding_length","qwen3vl.feed_forward_length","qwen3vl.attention.head_count","qwen3vl.attention.head_count_kv","qwen3vl.attention.key_length","qwen3vl.attention.value_length","qwen3vl.n_deepstack_layers"};
    uint64_t language_values[] = {QWEN_VL_LAYERS,QWEN_VL_CONTEXT,QWEN_VL_HIDDEN,QWEN_VL_MLP,QWEN_VL_HEADS,QWEN_VL_KV_HEADS,QWEN_VL_HEAD_DIM,QWEN_VL_HEAD_DIM,QWEN_VISION_DEEPSTACK};
    for (size_t i = 0; i < sizeof(language_keys) / sizeof(language_keys[0]); i++) {
        auto found = language.metadata.find(language_keys[i]);
        if (found != language.metadata.end() && found->second.integer() == language_values[i]) continue;
        error = "Unsupported Qwen3-VL language configuration: " + std::string(language_keys[i]);
        return false;
    }
    auto theta = language.metadata.find("qwen3vl.rope.freq_base"),sections = language.metadata.find("qwen3vl.rope.dimension_sections");
    std::vector<uint64_t> expected_sections = {QWEN_VL_SECTION_T,QWEN_VL_SECTION_H,QWEN_VL_SECTION_W,0};
    float frequency;
    if (theta == language.metadata.end() || !parse_float(theta->second.scalar,frequency) || frequency != QWEN_VL_THETA || sections == language.metadata.end() || sections->second.array.size() != expected_sections.size()) {
        error = "Unsupported Qwen3-VL rotary configuration.";
        return false;
    }
    for (size_t i = 0; i < expected_sections.size(); i++) {
        if (sections->second.array[i].integer() == expected_sections[i]) continue;
        error = "Unsupported Qwen3-VL rotary sections.";
        return false;
    }
    const char* vision_keys[] = {"clip.vision.projection_dim","clip.vision.patch_size","clip.vision.embedding_length","clip.vision.feed_forward_length","clip.vision.block_count","clip.vision.attention.head_count","clip.vision.spatial_merge_size"};
    uint64_t vision_values[] = {QWEN_VL_HIDDEN,QWEN_VISION_PATCH,QWEN_VISION_HIDDEN,QWEN_VISION_MLP,QWEN_VISION_LAYERS,QWEN_VISION_HEADS,QWEN_VISION_MERGE};
    for (size_t i = 0; i < sizeof(vision_keys) / sizeof(vision_keys[0]); i++) {
        auto found = vision.metadata.find(vision_keys[i]);
        if (found != vision.metadata.end() && found->second.integer() == vision_values[i]) continue;
        error = "Unsupported Qwen3-VL vision configuration: " + std::string(vision_keys[i]);
        return false;
    }
    auto projector = vision.metadata.find("clip.projector_type"),deep_layers = vision.metadata.find("clip.vision.is_deepstack_layers");
    size_t selected_layers[] = QWEN_VISION_DEEP_LAYERS;
    if (projector == vision.metadata.end() || projector->second.scalar != "qwen3vl_merger" || deep_layers == vision.metadata.end() || deep_layers->second.array.size() != QWEN_VISION_LAYERS) {
        error = "Unsupported Qwen3-VL projector/DeepStack configuration.";
        return false;
    }
    for (size_t layer = 0; layer < QWEN_VISION_LAYERS; layer++) {
        bool selected = std::find(std::begin(selected_layers),std::end(selected_layers),layer) != std::end(selected_layers);
        if (deep_layers->second.array[layer].scalar == (selected? "true" : "false")) continue;
        error = "Unsupported Qwen3-VL DeepStack layer selection.";
        return false;
    }
    if (!qwen_vl_weight(language,QWEN_VL_EMBEDDING_WEIGHT,{QWEN_VL_HIDDEN,QWEN_VL_VOCABULARY},error) || !qwen_vl_weight(language,"output_norm.weight",{QWEN_VL_HIDDEN},error)) return false;
    for (size_t layer = 0; layer < QWEN_VL_LAYERS; layer++) {
        std::string prefix = QWEN_VL_BLOCK_PREFIX + std::to_string(layer) + ".";
        const char* norms[] = {QWEN_VL_ATTENTION_NORM,QWEN_VL_FFN_NORM};
        for (const char* norm : norms) {
            if (!qwen_vl_weight(language,prefix + norm,{QWEN_VL_HIDDEN},error)) return false;
        }
        const char* heads[] = {QWEN_VL_QUERY_NORM,QWEN_VL_KEY_NORM};
        for (const char* head : heads) {
            if (!qwen_vl_weight(language,prefix + head,{QWEN_VL_HEAD_DIM},error)) return false;
        }
        if (!qwen_vl_weight(language,prefix + "attn_q.weight",{QWEN_VL_HIDDEN,QWEN_VL_HIDDEN},error) || !qwen_vl_weight(language,prefix + "attn_output.weight",{QWEN_VL_HIDDEN,QWEN_VL_HIDDEN},error)) return false;
        const char* kv[] = {"attn_k.weight","attn_v.weight"};
        for (const char* projection : kv) {
            if (!qwen_vl_weight(language,prefix + projection,{QWEN_VL_HIDDEN,QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM},error)) return false;
        }
        if (!qwen_vl_weight(language,prefix + "ffn_gate.weight",{QWEN_VL_HIDDEN,QWEN_VL_MLP},error) || !qwen_vl_weight(language,prefix + "ffn_up.weight",{QWEN_VL_HIDDEN,QWEN_VL_MLP},error) || !qwen_vl_weight(language,prefix + "ffn_down.weight",{QWEN_VL_MLP,QWEN_VL_HIDDEN},error)) return false;
    }
    if (!qwen_vl_weight(vision,QWEN_VISION_PATCH_WEIGHT,{QWEN_VISION_PATCH,QWEN_VISION_PATCH,3,QWEN_VISION_HIDDEN},error) || !qwen_vl_weight(vision,QWEN_VISION_PATCH_SECOND_WEIGHT,{QWEN_VISION_PATCH,QWEN_VISION_PATCH,3,QWEN_VISION_HIDDEN},error) || !qwen_vl_weight(vision,QWEN_VISION_PATCH_BIAS,{QWEN_VISION_HIDDEN},error) || !qwen_vl_weight(vision,QWEN_VISION_POSITION_WEIGHT,{QWEN_VISION_HIDDEN,QWEN_VISION_POSITIONS},error)) return false;
    for (size_t layer = 0; layer < QWEN_VISION_LAYERS; layer++) {
        std::string prefix = QWEN_VISION_BLOCK_PREFIX + std::to_string(layer) + ".";
        const char* norms[] = {QWEN_VISION_ATTENTION_NORM,QWEN_VISION_FFN_NORM};
        for (const char* norm : norms) {
            if (!qwen_vl_weight(vision,prefix + norm + WEIGHT_SUFFIX,{QWEN_VISION_HIDDEN},error) || !qwen_vl_weight(vision,prefix + norm + BIAS_SUFFIX,{QWEN_VISION_HIDDEN},error)) return false;
        }
        const char* names[] = {QWEN_VISION_QKV,QWEN_VISION_ATTENTION_OUTPUT,QWEN_VL_FFN_UP,QWEN_VL_FFN_DOWN};
        size_t inputs[] = {QWEN_VISION_HIDDEN,QWEN_VISION_HIDDEN,QWEN_VISION_HIDDEN,QWEN_VISION_MLP};
        size_t outputs[] = {QWEN_VISION_HIDDEN * 3,QWEN_VISION_HIDDEN,QWEN_VISION_MLP,QWEN_VISION_HIDDEN};
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            if (!qwen_vl_weight(vision,prefix + names[i] + WEIGHT_SUFFIX,{inputs[i],outputs[i]},error) || !qwen_vl_weight(vision,prefix + names[i] + BIAS_SUFFIX,{outputs[i]},error)) return false;
        }
    }
    size_t selected[] = QWEN_VISION_DEEP_LAYERS;
    for (size_t i = 0; i <= QWEN_VISION_DEEPSTACK; i++) {
        bool final = i == QWEN_VISION_DEEPSTACK;
        std::string prefix = final? QWEN_VISION_MERGER_PREFIX : QWEN_VISION_DEEP_PREFIX + std::to_string(selected[i]) + ".";
        std::string norm = final? QWEN_VISION_FINAL_NORM : prefix + QWEN_VISION_MERGER_NORM;
        size_t hidden = QWEN_VISION_HIDDEN * QWEN_VISION_MERGE_PIXELS;
        size_t normalized = final? QWEN_VISION_HIDDEN : hidden;
        if (!qwen_vl_weight(vision,norm + WEIGHT_SUFFIX,{normalized},error) || !qwen_vl_weight(vision,norm + BIAS_SUFFIX,{normalized},error)) return false;
        std::string first = prefix + (final? "0" : QWEN_VISION_MERGER_FIRST),second = prefix + (final? "2" : QWEN_VISION_MERGER_SECOND);
        if (!qwen_vl_weight(vision,first + WEIGHT_SUFFIX,{hidden,hidden},error) || !qwen_vl_weight(vision,first + BIAS_SUFFIX,{hidden},error) || !qwen_vl_weight(vision,second + WEIGHT_SUFFIX,{hidden,QWEN_VL_HIDDEN},error) || !qwen_vl_weight(vision,second + BIAS_SUFFIX,{QWEN_VL_HIDDEN},error)) return false;
    }
    PixTokenz tokenizer;
    if (!tokenizer.load(language)) {
        error = "Invalid embedded Qwen3-VL tokenizer.";
        return false;
    }
    return true;
}

PixVLInput qwen_vl_prepare(const PixTokenz &tokenizer, const std::string &prompt, const std::vector<PixImage> &images, bool pack_pixels)
{
    PixVLInput input;
    std::vector<int> image_id = tokenizer.encode(QWEN_VL_IMAGE_PAD,false);
    std::vector<int> start_id = tokenizer.encode(QWEN_VL_VISION_START_TEXT,false),end_id = tokenizer.encode(QWEN_VL_VISION_END_TEXT,false);
    if (image_id != std::vector<int>{QWEN_VL_IMAGE_TOKEN} || start_id != std::vector<int>{QWEN_VL_VISION_START} || end_id != std::vector<int>{QWEN_VL_VISION_END}) return {};
    std::vector<int> system = tokenizer.encode(QWEN_VL_SYSTEM,false);
    if (system.empty()) return {};
    input.trim = system.size();
    std::string text = QWEN_VL_SYSTEM QWEN_VL_USER;
    for (size_t reference = 0; reference < images.size(); reference++) {
        const PixImage &source = images[reference];
        if (!image_valid(source)) return {};
        int width = source.width,height = source.height;
        double pixels = double(width) * height;
        if (pixels < QWEN_VISION_MIN_PIXELS || pixels > QWEN_VISION_MAX_PIXELS) {
            double target = pixels < QWEN_VISION_MIN_PIXELS? QWEN_VISION_MIN_PIXELS : QWEN_VISION_MAX_PIXELS;
            double factor = std::sqrt(target / pixels);
            double rounded_width = width * factor / (QWEN_VISION_PATCH * QWEN_VISION_MERGE),rounded_height = height * factor / (QWEN_VISION_PATCH * QWEN_VISION_MERGE);
            rounded_width = pixels < QWEN_VISION_MIN_PIXELS? std::ceil(rounded_width) : std::floor(rounded_width);
            rounded_height = pixels < QWEN_VISION_MIN_PIXELS? std::ceil(rounded_height) : std::floor(rounded_height);
            width = std::max(QWEN_VISION_PATCH * QWEN_VISION_MERGE,int(rounded_width) * QWEN_VISION_PATCH * QWEN_VISION_MERGE);
            height = std::max(QWEN_VISION_PATCH * QWEN_VISION_MERGE,int(rounded_height) * QWEN_VISION_PATCH * QWEN_VISION_MERGE);
        }
        if (width % (QWEN_VISION_PATCH * QWEN_VISION_MERGE) || height % (QWEN_VISION_PATCH * QWEN_VISION_MERGE)) return {};
        PixVisionGrid grid = {1,size_t(height / QWEN_VISION_PATCH),size_t(width / QWEN_VISION_PATCH),reference};
        if (pack_pixels) {
            PixImage resized = image_resize(source,width,height,PIX_IMAGE_BICUBIC,PIX_IMAGE_WHITE);
            if (!image_valid(resized)) return {};
            PixTensor patches(grid.height * grid.width,3 * QWEN_VISION_TEMPORAL_PATCH * QWEN_VISION_PATCH * QWEN_VISION_PATCH);
            size_t row = 0;
            for (size_t by = 0; by < grid.height; by += QWEN_VISION_MERGE) {
                for (size_t bx = 0; bx < grid.width; bx += QWEN_VISION_MERGE) {
                    for (size_t dy = 0; dy < QWEN_VISION_MERGE; dy++) {
                        for (size_t dx = 0; dx < QWEN_VISION_MERGE; dx++,row++) {
                            size_t feature = 0;
                            for (size_t c = 0; c < 3; c++) {
                                for (size_t t = 0; t < QWEN_VISION_TEMPORAL_PATCH; t++) {
                                    for (size_t y = 0; y < QWEN_VISION_PATCH; y++) {
                                        for (size_t x = 0; x < QWEN_VISION_PATCH; x++,feature++) {
                                            size_t pixel = ((by + dy) * QWEN_VISION_PATCH + y) * width + (bx + dx) * QWEN_VISION_PATCH + x;
                                            patches.values[row * patches.channels + feature] = (resized.rgb[pixel * 3 + c] * QWEN_VL_PIXELS_SCALE - 0.5f) * QWEN_VL_NORMALIZE_SCALE;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            input.patches.push_back(std::move(patches));
        }
        input.grids.push_back(grid);
        if (reference) text += ' ';
        text += "<image" + std::to_string(reference + 1) + ">" QWEN_VL_VISION_START_TEXT QWEN_VL_IMAGE_PAD QWEN_VL_VISION_END_TEXT;
    }
    text += (prompt.empty()? " " : prompt) + std::string(TOKENIZER_CHAT_SUFFIX);
    std::vector<int> ids = tokenizer.encode(text,false);
    size_t reference = 0,position = 0;
    for (int id : ids) {
        if (id != QWEN_VL_IMAGE_TOKEN) {
            input.ids.push_back(id);
            input.image_slots.push_back(0);
            for (auto &axis : input.positions) axis.push_back(position);
            position++;
        } else {
            if (reference >= input.grids.size()) return {};
            const PixVisionGrid &grid = input.grids[reference++];
            size_t height = grid.height / QWEN_VISION_MERGE,width = grid.width / QWEN_VISION_MERGE;
            for (size_t y = 0; y < height; y++) {
                for (size_t x = 0; x < width; x++) {
                    input.ids.push_back(id);
                    input.image_slots.push_back(1);
                    input.positions[0].push_back(position);
                    input.positions[1].push_back(position + y);
                    input.positions[2].push_back(position + x);
                }
            }
            position += std::max(height,width);
        }
        if (input.ids.size() > QWEN_VL_CONTEXT) return {};
    }
    if (input.ids.empty() || reference != images.size() || input.trim >= input.ids.size()) return {};
    return input;
}

static PixTensor qwen_vl_affine(const PixModelFile &model, const std::string &name, const PixTensor &input)
{
    PixTensor result = tensor_norm(input,tensor_vector(model,name + WEIGHT_SUFFIX),QWEN_VL_EPSILON,true);
    tensor_bias(result,tensor_vector(model,name + BIAS_SUFFIX));
    return result;
}

static PixTensor qwen_vl_merger(const PixModelFile &vision, const PixTensor &input, const std::string &prefix, bool final)
{
    PixTensor shuffled;
    if (final) shuffled = qwen_vl_affine(vision,QWEN_VISION_FINAL_NORM,input);
    else shuffled = tensor_slice(input,0,input.rows,0,input.channels);
    shuffled.rows /= QWEN_VISION_MERGE_PIXELS;
    shuffled.channels *= QWEN_VISION_MERGE_PIXELS;
    if (!final) shuffled = qwen_vl_affine(vision,prefix + QWEN_VISION_MERGER_NORM,shuffled);
    shuffled = tensor_linear(vision,prefix + (final? "0" : QWEN_VISION_MERGER_FIRST),shuffled,true);
    tensor_gelu(shuffled,false);
    return tensor_linear(vision,prefix + (final? "2" : QWEN_VISION_MERGER_SECOND),shuffled,true);
}

PixVisionFeatures qwen_vl_vision(const PixModelFile &vision, const PixVLInput &input, const InferenceProgress &progress)
{
    if (input.grids.size() != input.patches.size()) return {};
    PixVisionFeatures features;
    size_t total = 0;
    for (const PixVisionGrid &grid : input.grids) {
        features.offsets.push_back(total);
        total += grid.height * grid.width / QWEN_VISION_MERGE_PIXELS;
    }
    features.merged = PixTensor(total,QWEN_VL_HIDDEN);
    for (PixTensor &deep : features.deepstack) deep = PixTensor(total,QWEN_VL_HIDDEN);
    std::vector<float> learned = tensor_vector(vision,QWEN_VISION_POSITION_WEIGHT);
    size_t selected[] = QWEN_VISION_DEEP_LAYERS;
    for (size_t reference = 0; reference < input.grids.size(); reference++) {
        const PixVisionGrid &grid = input.grids[reference];
        const PixTensor &patches = input.patches[reference];
        size_t patch_pixels = QWEN_VISION_PATCH * QWEN_VISION_PATCH;
        PixTensor packed(patches.rows,3 * patch_pixels);
        PixTensor hidden;
        for (size_t time = 0; time < QWEN_VISION_TEMPORAL_PATCH; time++) {
#ifdef PIXELINA_WITH_VULKAN
            if (!vulkan_synchronize()) return {};
#endif
            for (size_t row = 0; row < patches.rows; row++) {
                for (size_t c = 0; c < 3; c++) std::copy_n(patches.values.data() + row * patches.channels + (c * QWEN_VISION_TEMPORAL_PATCH + time) * patch_pixels,patch_pixels,packed.values.data() + row * packed.channels + c * patch_pixels);
            }
            PixWeight weight = vision.tensor(time? QWEN_VISION_PATCH_SECOND_WEIGHT : QWEN_VISION_PATCH_WEIGHT);
            weight.shape = {packed.channels,QWEN_VISION_HIDDEN};
            PixTensor projected(patches.rows,QWEN_VISION_HIDDEN);
            if (!cpu_matmul(vision,weight,packed.values.data(),packed.rows,projected.values.data())) return {};
            if (!time) hidden = std::move(projected);
            else tensor_add(hidden,projected);
        }
        tensor_bias(hidden,tensor_vector(vision,QWEN_VISION_PATCH_BIAS));
        PixTensor positions(hidden.rows,hidden.channels);
        PixRotary rotary;
        rotary.cosine.resize(hidden.rows * QWEN_VISION_HEAD_DIM / 2);
        rotary.sine.resize(rotary.cosine.size());
        size_t row = 0;
        for (size_t by = 0; by < grid.height; by += QWEN_VISION_MERGE) {
            for (size_t bx = 0; bx < grid.width; bx += QWEN_VISION_MERGE) {
                for (size_t dy = 0; dy < QWEN_VISION_MERGE; dy++) {
                    for (size_t dx = 0; dx < QWEN_VISION_MERGE; dx++,row++) {
                        size_t y = by + dy,x = bx + dx;
                        float py = float(y) * (QWEN_VISION_POSITION_SIDE - 1) / (grid.height - 1),px = float(x) * (QWEN_VISION_POSITION_SIDE - 1) / (grid.width - 1);
                        size_t y0 = size_t(py),x0 = size_t(px),y1 = std::min(y0 + 1,size_t(QWEN_VISION_POSITION_SIDE - 1)),x1 = std::min(x0 + 1,size_t(QWEN_VISION_POSITION_SIDE - 1));
                        float fy = py - y0,fx = px - x0;
                        for (size_t c = 0; c < hidden.channels; c++) positions.values[row * hidden.channels + c] = learned[(y0 * QWEN_VISION_POSITION_SIDE + x0) * hidden.channels + c] * (1 - fy) * (1 - fx) + learned[(y0 * QWEN_VISION_POSITION_SIDE + x1) * hidden.channels + c] * (1 - fy) * fx + learned[(y1 * QWEN_VISION_POSITION_SIDE + x0) * hidden.channels + c] * fy * (1 - fx) + learned[(y1 * QWEN_VISION_POSITION_SIDE + x1) * hidden.channels + c] * fy * fx;
                        for (size_t pair = 0; pair < QWEN_VISION_HEAD_DIM / 2; pair++) {
                            size_t coordinate = pair < QWEN_VISION_HEAD_DIM / 4? y : x,index = pair % (QWEN_VISION_HEAD_DIM / 4);
                            float angle = float(coordinate / std::pow(QWEN_VISION_THETA,double(index * 2) / (QWEN_VISION_HEAD_DIM / 2)));
                            rotary.cosine[row * QWEN_VISION_HEAD_DIM / 2 + pair] = std::cos(angle);
                            rotary.sine[row * QWEN_VISION_HEAD_DIM / 2 + pair] = std::sin(angle);
                        }
                    }
                }
            }
        }
        tensor_add(hidden,positions);
        positions = PixTensor();
        packed = PixTensor();
        for (size_t layer = 0; layer < QWEN_VISION_LAYERS; layer++) {
            std::string prefix = QWEN_VISION_BLOCK_PREFIX + std::to_string(layer) + ".";
            PixTensor normalized = qwen_vl_affine(vision,prefix + QWEN_VISION_ATTENTION_NORM,hidden);
            PixTensor qkv = tensor_linear(vision,prefix + QWEN_VISION_QKV,normalized,true);
            normalized = PixTensor();
            PixTensor q = tensor_slice(qkv,0,qkv.rows,0,QWEN_VISION_HIDDEN),k = tensor_slice(qkv,0,qkv.rows,QWEN_VISION_HIDDEN,QWEN_VISION_HIDDEN),v = tensor_slice(qkv,0,qkv.rows,QWEN_VISION_HIDDEN * 2,QWEN_VISION_HIDDEN);
            qkv = PixTensor();
            tensor_rope(q,QWEN_VISION_HEADS,rotary.cosine,rotary.sine,false);
            tensor_rope(k,QWEN_VISION_HEADS,rotary.cosine,rotary.sine,false);
            PixTensor attended = tensor_attention(q,k,v,QWEN_VISION_HEADS,QWEN_VISION_HEADS,false);
            q = k = v = PixTensor();
            tensor_add(hidden,tensor_linear(vision,prefix + QWEN_VISION_ATTENTION_OUTPUT,attended,true));
            attended = PixTensor();
            normalized = qwen_vl_affine(vision,prefix + QWEN_VISION_FFN_NORM,hidden);
            PixTensor mlp = tensor_linear(vision,prefix + QWEN_VL_FFN_UP,normalized,true);
            normalized = PixTensor();
            tensor_gelu(mlp,true);
            tensor_add(hidden,tensor_linear(vision,prefix + QWEN_VL_FFN_DOWN,mlp,true));
            if (hidden.values.empty()) return {};
            for (size_t i = 0; i < QWEN_VISION_DEEPSTACK; i++) {
                if (layer != selected[i]) continue;
                PixTensor deep = qwen_vl_merger(vision,hidden,QWEN_VISION_DEEP_PREFIX + std::to_string(layer) + ".",false);
                std::vector<size_t> rows(deep.rows);
                std::iota(rows.begin(),rows.end(),features.offsets[reference]);
                if (!tensor_scatter(features.deepstack[i],deep,rows)) return {};
            }
            if (progress && !progress(QWEN_VL_VISION_FRACTION * float(reference * QWEN_VISION_LAYERS + layer + 1) / (input.grids.size() * QWEN_VISION_LAYERS),"Vision block " + std::to_string(layer + 1))) return {};
        }
        PixTensor merged = qwen_vl_merger(vision,hidden,QWEN_VISION_MERGER_PREFIX,true);
        std::vector<size_t> rows(merged.rows);
        std::iota(rows.begin(),rows.end(),features.offsets[reference]);
        if (!tensor_scatter(features.merged,merged,rows)) return {};
    }
    return features;
}

PixQwenCondition qwen_vl_language(const PixModelFile &language, const PixVLInput &input, PixVisionFeatures features, const InferenceProgress &progress)
{
    if (input.ids.empty() || input.ids.size() != input.image_slots.size() || input.trim >= input.ids.size()) return {};
    const PixWeight &embedding = language.tensor(QWEN_VL_EMBEDDING_WEIGHT);
    size_t stride = size_t(weight_bytes({QWEN_VL_HIDDEN},embedding.type));
    PixTensor hidden(input.ids.size(),QWEN_VL_HIDDEN);
    for (size_t i = 0; i < input.ids.size(); i++) {
        if (input.ids[i] < 0 || input.ids[i] >= QWEN_VL_VOCABULARY) return {};
#ifdef PIXELINA_WITH_VULKAN
        if (!vulkan_active_state())
#endif
        cpu_decode(language.data + embedding.offset + size_t(input.ids[i]) * stride,embedding.type,QWEN_VL_HIDDEN,hidden.values.data() + i * QWEN_VL_HIDDEN);
    }
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) hidden = vulkan_embedding(language,embedding,input.ids);
#endif
    std::vector<size_t> slots;
    for (size_t row = 0; row < input.image_slots.size(); row++) {
        if (input.image_slots[row]) slots.push_back(row);
    }
    if (!tensor_scatter(hidden,features.merged,slots)) return {};
    features.merged = PixTensor();
    PixRotary rotary;
    rotary.cosine.resize(hidden.rows * QWEN_VL_HEAD_DIM / 2);
    rotary.sine.resize(rotary.cosine.size());
    for (size_t row = 0; row < hidden.rows; row++) {
        for (size_t pair = 0; pair < QWEN_VL_HEAD_DIM / 2; pair++) {
            size_t axis = 0;
            if (pair % QWEN_VL_AXIS_PERIOD == 1 && pair < QWEN_VL_SECTION_H * QWEN_VL_AXIS_PERIOD) axis = 1;
            if (pair % QWEN_VL_AXIS_PERIOD == 2 && pair < QWEN_VL_SECTION_W * QWEN_VL_AXIS_PERIOD) axis = 2;
            float angle = float(input.positions[axis][row] / std::pow(QWEN_VL_THETA,double(pair * 2) / QWEN_VL_HEAD_DIM));
            rotary.cosine[row * QWEN_VL_HEAD_DIM / 2 + pair] = std::cos(angle);
            rotary.sine[row * QWEN_VL_HEAD_DIM / 2 + pair] = std::sin(angle);
        }
    }
    for (size_t layer = 0; layer < QWEN_VL_LAYERS; layer++) {
        std::string prefix = QWEN_VL_BLOCK_PREFIX + std::to_string(layer) + ".";
        PixTensor normalized = tensor_norm(hidden,tensor_vector(language,prefix + QWEN_VL_ATTENTION_NORM),QWEN_VL_EPSILON);
        PixTensor q = tensor_linear(language,prefix + "attn_q",normalized,false),k = tensor_linear(language,prefix + "attn_k",normalized,false),v = tensor_linear(language,prefix + "attn_v",normalized,false);
        normalized = PixTensor();
        size_t qrows = q.rows,krows = k.rows;
        q.rows *= QWEN_VL_HEADS;
        q.channels = QWEN_VL_HEAD_DIM;
        k.rows *= QWEN_VL_KV_HEADS;
        k.channels = QWEN_VL_HEAD_DIM;
        q = tensor_norm(q,tensor_vector(language,prefix + QWEN_VL_QUERY_NORM),QWEN_VL_EPSILON);
        k = tensor_norm(k,tensor_vector(language,prefix + QWEN_VL_KEY_NORM),QWEN_VL_EPSILON);
        q.rows = qrows;
        q.channels = QWEN_VL_HIDDEN;
        k.rows = krows;
        k.channels = QWEN_VL_KV_HEADS * QWEN_VL_HEAD_DIM;
        tensor_rope(q,QWEN_VL_HEADS,rotary.cosine,rotary.sine,false);
        tensor_rope(k,QWEN_VL_KV_HEADS,rotary.cosine,rotary.sine,false);
        PixTensor attended = tensor_attention(q,k,v,QWEN_VL_HEADS,QWEN_VL_KV_HEADS,true);
        q = k = v = PixTensor();
        tensor_add(hidden,tensor_linear(language,prefix + "attn_output",attended,false));
        attended = PixTensor();
        normalized = tensor_norm(hidden,tensor_vector(language,prefix + QWEN_VL_FFN_NORM),QWEN_VL_EPSILON);
        PixTensor gate = tensor_linear(language,prefix + "ffn_gate",normalized,false),up = tensor_linear(language,prefix + QWEN_VL_FFN_UP,normalized,false);
        normalized = PixTensor();
        tensor_silu(gate);
        tensor_gate(gate,up);
        up = PixTensor();
        tensor_add(hidden,tensor_linear(language,prefix + QWEN_VL_FFN_DOWN,gate,false));
        if (layer < QWEN_VISION_DEEPSTACK) {
            PixTensor selected = tensor_gather(hidden,slots);
            tensor_add(selected,features.deepstack[layer]);
            if (!tensor_scatter(hidden,selected,slots)) return {};
            features.deepstack[layer] = PixTensor();
        }
        if (hidden.values.empty()) return {};
        if (progress && !progress(QWEN_VL_VISION_FRACTION + QWEN_VL_LANGUAGE_FRACTION * float(layer + 1) / QWEN_VL_LAYERS,"Language block " + std::to_string(layer + 1))) return {};
    }
    PixQwenCondition condition;
    condition.hidden = tensor_slice(hidden,input.trim,hidden.rows - input.trim,0,hidden.channels);
    condition.image_slots.assign(input.image_slots.begin() + input.trim,input.image_slots.end());
    return condition;
}
