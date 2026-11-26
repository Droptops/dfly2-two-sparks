// Native DFly2 primitives. Correctness-first, architecture-neutral CUDA: no SM90
// instructions and no change to target KV. FP4 expert scales are E8M0 / 32;
// target cache scales are E4M3 / 16. These formats must never be interchanged.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
using B = __nv_bfloat16;
__device__ float bf(B x) { return __bfloat162float(x); }
__device__ B rb(float x) { return __float2bfloat16_rn(x); }
__device__ float e2m1(unsigned c) {
    const float v[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
__device__ float e8m0(unsigned c) {
    if (c == 255) return nanf("");
    return ldexpf(1.f, int(c) - 127);
}
__device__ float e4m3(unsigned c) {
    unsigned e = (c >> 3) & 15, m = c & 7;
    float v = e ? ldexpf(1.f + float(m) / 8, int(e) - 7) : ldexpf(float(m), -9);
    if (e == 15 && m == 7) v = nanf("");
    return c & 128 ? -v : v;
}
__device__ float warp_sum(float x) {
    for (int o=16;o;o/=2) x += __shfl_down_sync(0xffffffff, x, o);
    return x;
}
__device__ float block_sum(float x) {
    __shared__ float tmp[32];
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    x=warp_sum(x); if (!lane) tmp[warp]=x; __syncthreads();
    x=threadIdx.x < blockDim.x/32 ? tmp[lane] : 0;
    if (!warp) x=warp_sum(x);
    if (!threadIdx.x) tmp[0]=x; __syncthreads();
    float result=tmp[0]; __syncthreads(); return result;
}

// Normalize two independent vectors then concatenate, using the fused serving
// RMSNorm rounding (multiply norm weight in FP32, then materialize BF16).
extern "C" __global__ void tf_dfly_norm_concat(const B* x,const B* y,const B* wx,const B* wy,
                                               B* out,int d,float eps) {
    float sx=0,sy=0;
    for(int i=threadIdx.x;i<d;i+=blockDim.x) { float a=bf(x[i]),b=bf(y[i]);sx+=a*a;sy+=b*b; }
    sx=rsqrtf(block_sum(sx)/d+eps);sy=rsqrtf(block_sum(sy)/d+eps);
    for(int i=threadIdx.x;i<d;i+=blockDim.x) {out[i]=rb(bf(x[i])*sx*bf(wx[i]));out[d+i]=rb(bf(y[i])*sy*bf(wy[i]));}
}
extern "C" __global__ void tf_dfly_init_block(const B* embed,const int64_t* ids,const B* fusion,
                                              const B* gate,B* h,float* pre,int d,int rows) {
    int row=blockIdx.x;
    for(int i=threadIdx.x;i<d;i+=blockDim.x) {
        B v=embed[ids[row]*int64_t(d)+i];
        // multiplication and addition each round in BF16 in the reference.
        if(!row) v=rb(bf(v)+bf(rb(bf(fusion[i])*bf(gate[i]))));
        for(int c=0;c<4;c++) h[(row*4+c)*d+i]=v;
    }
    if(threadIdx.x<4) pre[row*4+threadIdx.x]=threadIdx.x==0 ? 1.f:0.f;
}
extern "C" __global__ void tf_dfly_add(const B* a,const B* b,B* out,int n) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) out[i]=rb(bf(a[i])+bf(b[i]));
}
extern "C" __global__ void tf_dfly_silu(const B* gate,const B* up,B* out,int n,float limit) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        float g=bf(gate[i]),u=bf(up[i]);
        if(limit>0) {g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
        // DFly2 expert/MLP activation materializes SiLU in BF16 before multiply.
        out[i]=rb(bf(rb(g/(1.f+expf(-g))))*u);
    }
}

// Replicated BF16 router, FP32 dot products. Select by sqrt(softplus)+bias,
// normalize the ORIGINAL unbiased top-3 scores, then scale by 1.5.
extern "C" __global__ void tf_dfly_route(const B* x,const B* gate,const float* bias,
                                         int* ids,float* weights,int d) {
    __shared__ float scores[128],ranked[128];
    int row=blockIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
    for(int e=warp;e<128;e+=blockDim.x/32) {
        float v=0;for(int k=lane;k<d;k+=32) v+=bf(x[row*d+k])*bf(gate[e*d+k]);
        v=warp_sum(v);
        if(!lane) {float s=sqrtf(v>20 ? v:log1pf(expf(v)));scores[e]=s;ranked[e]=s+bias[e];}
    }
    __syncthreads();
    if(!threadIdx.x) {
        float sum=0;
        for(int j=0;j<3;j++) {int best=0;for(int e=1;e<128;e++) if(ranked[e]>ranked[best]) best=e;
            ids[row*3+j]=best;weights[row*3+j]=scores[best];sum+=scores[best];ranked[best]=-INFINITY;}
        for(int j=0;j<3;j++) weights[row*3+j]=weights[row*3+j]/(sum+1e-20f)*1.5f;
    }
}

// Each CTA computes four output channels of one selected expert. The intermediate
// shard's K is contiguous; loader slices gate/up N and down K at multiples of32.
extern "C" __global__ void tf_dfly_fp4(const uint8_t* codes,const uint8_t* scales,const B* x,
                                       const int* ids,B* out,int n,int k,int rows,int expert_input) {
    int channel=blockIdx.x*4+threadIdx.x/32;
    int sel=blockIdx.y,row=sel/3,slot=sel%3,lane=threadIdx.x%32;
    if(channel>=n||row>=rows) return;
    int expert=ids[sel]; int64_t wr=(int64_t(expert)*n+channel);
    const B* in=x+(expert_input ? int64_t(sel)*k:int64_t(row)*k);
    float sum=0;
    for(int i=lane;i<k;i+=32) {
        unsigned p=codes[wr*(k/2)+i/2],c=(i&1)?p>>4:p&15;
        float w=e2m1(c)*e8m0(scales[wr*(k/32)+i/32]);
        sum+=bf(in[i])*w;
    }
    sum=warp_sum(sum);if(!lane) out[int64_t(sel)*n+channel]=rb(sum);
}
extern "C" __global__ void tf_dfly_mix_experts(const B* routed,const B* shared,const float* weights,
                                               float* out,int d,int rows) {
    for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<rows*d;p+=blockDim.x*gridDim.x) {
        int row=p/d,col=p%d;float v=bf(shared[p]);
        for(int j=0;j<3;j++) v+=bf(routed[(row*3+j)*d+col])*weights[row*3+j];out[p]=v;
    }
}

// Compact the512 selected target FP4 latents and128 local BF16 keys. Invalid keys
// get a separate mask; zero vectors would otherwise change the softmax denominator.
extern "C" __global__ void tf_dfly_gather_context(const uint8_t* codes,const uint8_t* scales,
    const int64_t* indices,const B* ring,B* out,int* valid,int64_t compressed_base,
    int64_t ring_base,int ring_rows,int64_t anchor,int64_t window_start) {
    int key=blockIdx.x,c=threadIdx.x;
    int64_t pos=key<512 ? indices[key]:anchor-127+(key-512);
    bool ok=key<512 ? pos>=0 && pos<=anchor : pos>=window_start && pos>=0 && pos<=anchor;
    if(!c) valid[key]=ok;
    for(int d=c;d<512;d+=blockDim.x) {
        float v=0;
        if(ok) {
            // TensorFold cache packs the two HALVES of a512-vector, unlike
            // the checkpoint expert format (adjacent low/high nibbles).
            if(key<512) {int64_t row=compressed_base+pos;unsigned p=codes[row*256+d%256];
                v=e2m1(d>=256?p>>4:p&15)*e4m3(scales[row*32+d/16]);}
            else v=bf(ring[(ring_base+pos%ring_rows)*512+d]);
        }
        out[key*512+d]=rb(v);
    }
}

// Context attention plus optional noncausal draft keys. Per-head sink contributes
// only to the denominator. One CTA per query/head, FP32 accumulation.
extern "C" __global__ void tf_dfly_attention(const B* q,const B* context,const int* valid,
    const B* block,const float* sink,B* out,int heads,int rows,int block_rows) {
    __shared__ float logits[656],probs[656],reduce[8];
    int pair=blockIdx.x,row=pair/heads,head=pair%heads,lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(row>=rows)return;const B* qr=q+int64_t(pair)*512;int keys=640+block_rows;
    for(int key=warp;key<keys;key+=8) {
        bool ok=key>=640 || valid[key];const B* kv=key<640?context+key*512:block+(key-640)*512;
        float score=0;if(ok)for(int d=lane;d<512;d+=32)score+=bf(qr[d])*bf(kv[d]);
        score=warp_sum(score);if(!lane)logits[key]=ok?score*0.04419417382415922f:-INFINITY;
    }
    __syncthreads();float mx=sink[head];
    for(int key=threadIdx.x;key<keys;key+=256)mx=fmaxf(mx,logits[key]);
    for(int o=16;o;o/=2)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    if(!lane)reduce[warp]=mx;__syncthreads();
    if(!threadIdx.x){mx=reduce[0];for(int i=1;i<8;i++)mx=fmaxf(mx,reduce[i]);reduce[0]=mx;}__syncthreads();mx=reduce[0];
    float sum=threadIdx.x==0?expf(sink[head]-mx):0;
    for(int key=threadIdx.x;key<keys;key+=256){float p=expf(logits[key]-mx);probs[key]=p;sum+=p;}
    sum=block_sum(sum);__syncthreads();
    for(int d=threadIdx.x;d<512;d+=256){float v=0;for(int key=0;key<keys;key++) {
        const B* kv=key<640?context+key*512:block+(key-640)*512;v+=probs[key]/sum*bf(kv[d]);}
        out[int64_t(pair)*512+d]=rb(v);}
}

// The correction's causal block read is separately normalized, then tanh gated
// and added to context attention BEFORE its common inverse-RoPE/output projection.
extern "C" __global__ void tf_dfly_block_attention(const B* q,const B* keys,const B* gate,
    const float* sink,const float* rel_bias,const float* gate_bias,B* inout,int heads,int slot) {
    __shared__ float probs[6];int head=blockIdx.x;
    if(!threadIdx.x){float logits[6],mx=sink[head],sum;
        for(int k=0;k<=slot;k++){float s=0;for(int d=0;d<512;d++)s+=bf(q[head*512+d])*bf(keys[k*512+d]);
            logits[k]=s*0.04419417382415922f+rel_bias[head*16+slot-k];mx=fmaxf(mx,logits[k]);}
        sum=expf(sink[head]-mx);for(int k=0;k<=slot;k++){probs[k]=expf(logits[k]-mx);sum+=probs[k];}
        for(int k=0;k<=slot;k++)probs[k]/=sum;}
    __syncthreads();float g=tanhf(bf(gate[head])+gate_bias[head*16+slot]);
    for(int d=threadIdx.x;d<512;d+=blockDim.x){float s=0;for(int k=0;k<=slot;k++)s+=probs[k]*bf(keys[k*512+d]);
        inout[head*512+d]=rb(bf(inout[head*512+d])+g*s);}
}

// Local vocabulary winner as two floats (score,id); ids<2^24 are exact. Deterministic
// lowest-token tie breaking agrees across ranks, independent of collective ordering.
extern "C" __global__ void tf_dfly_best(const float* logits,float* best,int n,int offset) {
    __shared__ float scores[256];__shared__ int ids[256];int id=-1;float score=-INFINITY;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)if(logits[i]>score||(logits[i]==score&&(id<0||i<id))){score=logits[i];id=i;}
    scores[threadIdx.x]=score;ids[threadIdx.x]=id;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride){int j=threadIdx.x+stride;
        if(scores[j]>scores[threadIdx.x]||(scores[j]==scores[threadIdx.x]&&ids[j]>=0&&(ids[threadIdx.x]<0||ids[j]<ids[threadIdx.x]))){scores[threadIdx.x]=scores[j];ids[threadIdx.x]=ids[j];}}
        __syncthreads();}
    if(!threadIdx.x){best[0]=scores[0];best[1]=float(ids[0]+offset);}
}
