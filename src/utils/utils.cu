#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>

namespace waterSim::utils{
    [[maybe_unused]] bool transferCtxInit(transferCtx& ctx, size_t ptrSize){
        ctx.ptrSize = ptrSize;
        cudaError_t err = cudaMallocHost(&ctx.pageLockedHostPtr1, ptrSize);
        if(err != cudaSuccess){
            return false;
        }
        err = cudaMallocHost(&ctx.pageLockedHostPtr2, ptrSize);
        if(err != cudaSuccess){
            cudaFree(ctx.pageLockedHostPtr1);
            return false;
        }
        return true;
    }

    [[maybe_unused]] bool transferCtxDestroy(transferCtx& ctx){
        cudaError_t err = cudaFreeHost(ctx.pageLockedHostPtr1);
        if(err != cudaSuccess){
            return false;
        }
        err = cudaFreeHost(ctx.pageLockedHostPtr2);
        if(err != cudaSuccess){
            return false;
        }
        return true;
    }

    [[maybe_unused]] bool fastCudaTransferToDevice(transferCtx& ctx, void* hostPtr, void* devicePtr, size_t size){
        size_t transferred = 0;
        bool isPtr1 = true;
        while (transferred < size){
            size_t toTransfer = ctx.ptrSize;
            if(transferred + toTransfer > size){
                toTransfer = size - transferred;
            }
            if(isPtr1){
                cudaDeviceSynchronize();
                std::memcpy(ctx.pageLockedHostPtr1, (std::byte*)hostPtr + transferred, toTransfer);
                cudaError_t err = cudaMemcpyAsync(devicePtr, ctx.pageLockedHostPtr1, toTransfer, cudaMemcpyHostToDevice);
                if(err != cudaSuccess){
                    return false;
                }
                transferred += toTransfer;
                isPtr1 = false;
                continue;
            }
            std::memcpy(ctx.pageLockedHostPtr2, (std::byte*)hostPtr + transferred, toTransfer);
            cudaError_t err = cudaMemcpyAsync(devicePtr, ctx.pageLockedHostPtr2, toTransfer, cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                return false;
            }
            transferred += toTransfer;
            isPtr1 = true;
        }
        cudaDeviceSynchronize();
        return true;
    }

    [[maybe_unused]] bool fastCudaTransferToHost(transferCtx& ctx, void* devicePtr, void* hostPtr, size_t size){
        size_t transferred = 0;
        bool isPtr1 = true;
        while (transferred < size){
            size_t toTransfer = ctx.ptrSize;
            if(transferred + toTransfer > size){
                toTransfer = size - transferred;
            }
            if(isPtr1){
                cudaDeviceSynchronize();
                cudaError_t err = cudaMemcpyAsync(ctx.pageLockedHostPtr1, devicePtr, toTransfer, cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                    return false;
                }
                std::memcpy((std::byte*)hostPtr + transferred, ctx.pageLockedHostPtr1, toTransfer);
                transferred += toTransfer;
                isPtr1 = false;
                continue;
            }
            cudaError_t err = cudaMemcpyAsync(ctx.pageLockedHostPtr2, devicePtr, toTransfer, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess){
                return false;
            }
            std::memcpy((std::byte*)hostPtr + transferred, ctx.pageLockedHostPtr2, toTransfer);
            transferred += toTransfer;
            isPtr1 = true;
        }
        cudaDeviceSynchronize();
        return true;
    }
}