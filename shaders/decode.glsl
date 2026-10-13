uint weight_byte(uint address)
{
    return (weights[address / 4] >> ((address % 4) * 8)) & 255;
}

float weight_half(uint address)
{
    uint bits = weight_byte(address) | (weight_byte(address + 1) << 8);
    return unpackHalf2x16(bits).x;
}

int signed_byte(uint value)
{
    return value < 128? int(value) : int(value) - 256;
}

float weight_value(uint type, uint row, uint index)
{
    if (type == MODEL_F32) return uintBitsToFloat(weights[row / 4 + index]);
    if (type == MODEL_F16) return weight_half(row + index * 2);
    if (type == MODEL_BF16) {
        uint offset = row + index * 2;
        return uintBitsToFloat((weight_byte(offset) | (weight_byte(offset + 1) << 8)) << 16);
    }
    if (type == MODEL_Q8_0) {
        uint base = row + (index / MODEL_Q8_BLOCK) * MODEL_Q8_BYTES;
        return weight_half(base) * signed_byte(weight_byte(base + 2 + index % MODEL_Q8_BLOCK));
    }
    if (type == MODEL_Q5_K) {
        uint base = row + (index / MODEL_QUANT_BLOCK) * MODEL_Q5_BYTES;
        uint within = index % MODEL_QUANT_BLOCK,group = within / 32,lane = within % 32;
        uint scale,min;
        if (group < 4) {
            scale = weight_byte(base + 4 + group) & VK_Q5_SCALE_MASK;
            min = weight_byte(base + 8 + group) & VK_Q5_SCALE_MASK;
        } else {
            scale = (weight_byte(base + 8 + group) & 15) | ((weight_byte(base + group) >> 6) << 4);
            min = (weight_byte(base + 8 + group) >> 4) | ((weight_byte(base + 4 + group) >> 6) << 4);
        }
        uint low = weight_byte(base + VK_Q5_LOW_OFFSET + (group / 2) * 32 + lane);
        uint value = group % 2 == 0? low & 15 : low >> 4;
        if ((weight_byte(base + VK_Q5_HIGH_OFFSET + lane) & (1 << group)) != 0) value += 16;
        return weight_half(base) * scale * value - weight_half(base + 2) * min;
    }
    uint base = row + (index / MODEL_QUANT_BLOCK) * MODEL_Q6_BYTES;
    uint within = index % MODEL_QUANT_BLOCK,portion = within / 128,group = (within % 128) / 32,lane = within % 32;
    uint low = weight_byte(base + portion * 64 + lane + (group % 2) * 32);
    uint value = group < 2? low & 15 : low >> 4;
    value |= ((weight_byte(base + VK_Q6_HIGH_OFFSET + portion * 32 + lane) >> (group * 2)) & 3) << 4;
    int scale = signed_byte(weight_byte(base + VK_Q6_SCALE_OFFSET + portion * 8 + group * 2 + lane / 16));
    return weight_half(base + VK_Q6_DELTA_OFFSET) * scale * (int(value) - 32);
}
