#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "TriangleMesh.h"
#include "Forward.h"
#include "structs.h"

extern unsigned char collect_ids_ptx_code_file[];


struct CIParams
{
    StructuredBuffer<int> tri_collection;
    StructuredBuffer<int64_t> start_ids;
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;

    size_t feature_size;

    OptixTraversableHandle handle;
};

class CollectIds {
   public:
    CollectIds() = default;
    CollectIds(
        const OptixDeviceContext &context,
        int8_t device,
        const Primitives &model);
    CollectIds(const CollectIds &) = delete;
    CollectIds &operator=(const CollectIds &) = delete;
    CollectIds(CollectIds &&other) noexcept;
    CollectIds &operator=(CollectIds &&other) {
        using std::swap;
        if (this != &other) {
            CollectIds tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~CollectIds() noexcept(false);

    friend void swap(CollectIds &first, CollectIds &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.module, second.module);
        swap(first.sbt, second.sbt);
        swap(first.pipeline, second.pipeline);
        swap(first.d_param, second.d_param);
        swap(first.stream, second.stream);
        swap(first.raygen_prog_group, second.raygen_prog_group);
        swap(first.miss_prog_group, second.miss_prog_group);
        swap(first.hitgroup_prog_group, second.hitgroup_prog_group);
    }

    void trace_rays(const OptixTraversableHandle &handle,
                    const size_t num_rays,
                    float3 *ray_origins,
                    float3 *ray_directions,
                    int *tri_collection,
                    int64_t *start_ids,
                    uint total_tris);

   private:
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;

    // Local fields used for this pipeline
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;

    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    static std::string load_ptx_data() {
        return std::string((char *)collect_ids_ptx_code_file);
    }
};


