#pragma once

struct VkDrawIndirectCommand {
    uint vertexCount = 0;
    uint instanceCount = 0;
    uint firstVertex = 0;
    uint firstInstance = 0;
};

struct VkDrawIndexedIndirectCommand {
    uint indexCount = 0;
    uint instanceCount = 0;
    uint firstIndex = 0;
    int  vertexOffset = 0;
    uint firstInstance = 0;
};