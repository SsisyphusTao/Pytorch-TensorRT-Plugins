#include "yoloPlugin.h"
#include <iostream>

#define NUM_HEAD 3
#define NUM_ANCH 3
#define CONF_THRES 0.4f

using namespace nvinfer1;
using nvinfer1::plugin::YoloDetectLayer;
using nvinfer1::plugin::YoloPluginCreator;

namespace
{
const char* Yolo_PLUGIN_VERSION{"1"};
const char* Yolo_PLUGIN_NAME{"Yolo_TRT"};
} // namespace

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

const int CUDA_NUM_THREADS = 512;
dim3 GET_BLOCKS_(uint n)
{
    uint k = (n - 1) /CUDA_NUM_THREADS + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*CUDA_NUM_THREADS) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}

PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

// Parameterized constructor
YoloDetectLayer::YoloDetectLayer(
                         int num_cls,
                         int max_det,
                         std::vector<int> h, 
                         std::vector<int> w, 
                         std::vector<int> strides, //TODO
                         const Weights* anchors):
                         mNumcls(num_cls), mMaxdet(max_det){
        for (int i=0; i<NUM_HEAD; i++){
        mHeights[i] = h[i];
        mWidths[i] = w[i];
        mStrides[i] = strides[i];
        }
        mAnchors = copyToDevice(anchors[0].values, anchors[0].count);
}

YoloDetectLayer::YoloDetectLayer(const void* buffer, size_t length)
{
    const char* d = static_cast<const char*>(buffer);
    const char* a = d;

    mNumcls = read<int>(d);
    mMaxdet = read<int>(d);

    CUASSERT(cudaMemcpy(mHeights, d, NUM_HEAD * sizeof(int), cudaMemcpyHostToHost));
    d += NUM_HEAD * sizeof(int);
    CUASSERT(cudaMemcpy(mWidths, d, NUM_HEAD * sizeof(int), cudaMemcpyHostToHost));
    d += NUM_HEAD * sizeof(int);
    CUASSERT(cudaMemcpy(mStrides, d, NUM_HEAD * sizeof(int), cudaMemcpyHostToHost));
    d += NUM_HEAD * sizeof(int);

    int count = read<int>(d);
    mAnchors = deserializeToDevice(d, count);

    ASSERT(d == a + length);
}

int YoloDetectLayer::getNbOutputs() const
{
    // Plugin layer has 2 outputs
    return 3;
}

int YoloDetectLayer::initialize()
{
    return STATUS_SUCCESS; 
}

Dims YoloDetectLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputs)
{
    ASSERT(index == 0 || index == 1 || index == 2);
    ASSERT(nbInputs == 3);

    if (index==0) return Dims3(mMaxdet*4, 1, 1);
    else if (index==1) return Dims3(mMaxdet*mNumcls, 1, 1);
    else return Dims3(2, mMaxdet*4, 1);
}

size_t YoloDetectLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

__global__ void Reshape(const float *input, float *loc, float *cof, int w, int h,
   int numClass, int mMaxdet, int stride, const float *anchors, int* countAddress)
{
    CUDA_KERNEL_LOOP(idx, h*w*NUM_ANCH)
    {
        int mapSize = w * h;
        int anchorPos = idx / mapSize;
        int mapPos = idx % mapSize;
        int infoLen = 5 + numClass;
        if (input[(anchorPos*infoLen+4)*mapSize+mapPos]<CONF_THRES) continue;
        int count = (int)atomicAdd(countAddress, 1);

        if (count >= mMaxdet-1) return;

        for (int i = 5; i < infoLen; ++i)
        { 
            cof[numClass * count + i] = input[(anchorPos*infoLen+i)*mapSize+mapPos];
        }

        int row = mapPos / w;
        int col = mapPos % w;

        float ax, ay, aw, ah;

        ax   = (col - 0.5f + 2.0f * input[(anchorPos*infoLen)*mapSize+mapPos]) * stride;
        ay = (row - 0.5f + 2.0f * input[(anchorPos*infoLen+1)*mapSize+mapPos]) * stride;
        aw = 2.0f * input[(anchorPos*infoLen+2)*mapSize+mapPos];
        ah = 2.0f * input[(anchorPos*infoLen+3)*mapSize+mapPos];
        aw = aw * aw * anchors[anchorPos*2];
        ah = ah * ah * anchors[anchorPos*2+1];

        loc[4 * count]   = (ax - aw/2)/w/stride;
        loc[4 * count+1] = (ay - ah/2)/h/stride;
        loc[4 * count+2] = (ax + aw/2)/w/stride;
        loc[4 * count+3] = (ay + ah/2)/h/stride;
    }
}


int YoloDetectLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    float* loc = static_cast<float *>(outputs[0]);
    float* cof = static_cast<float *>(outputs[1]);
    CHECK_CUDA(cudaMalloc((void**)&count, sizeof(int)));

    CUASSERT(cudaMemset(count, 0, sizeof(int)));
    CUASSERT(cudaMemset(outputs[0], 0, mMaxdet*4*sizeof(float)));
    CUASSERT(cudaMemset(outputs[1], 0, mMaxdet*mNumcls*sizeof(float)));
    CUASSERT(cudaMemset(outputs[2], 0, 2*mMaxdet*4*sizeof(float)));

    for (int i=0; i<NUM_HEAD; i++)
    {
        const float* anchors = static_cast<const float *>(mAnchors.values) + 2 * NUM_ANCH * i;
        Reshape <<< GET_BLOCKS_(mHeights[i]*mWidths[i]*NUM_ANCH), CUDA_NUM_THREADS, 0, stream >>>
            (static_cast<const float *>(inputs[i]), loc, cof, mWidths[i], mHeights[i], mNumcls, mMaxdet, mStrides[i], anchors, count);
    }
    return 0;
}

size_t YoloDetectLayer::getSerializationSize() const
{
    return sizeof(int) * 12 + mAnchors.count * sizeof(float);
}

void YoloDetectLayer::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mNumcls);
    write(d, mMaxdet);

    CUASSERT(cudaMemcpy(d, mHeights, NUM_HEAD * sizeof(int), cudaMemcpyHostToHost));
    d += NUM_HEAD * sizeof(int);
    CUASSERT(cudaMemcpy(d, mWidths, NUM_HEAD * sizeof(int), cudaMemcpyHostToHost));
    d += NUM_HEAD * sizeof(int);
    CUASSERT(cudaMemcpy(d, mStrides, NUM_HEAD * sizeof(int), cudaMemcpyHostToHost));
    d += NUM_HEAD * sizeof(int);
    write(d, (int) mAnchors.count);
    serializeFromDevice(d, mAnchors);

    ASSERT(d == a + getSerializationSize());
}

bool YoloDetectLayer::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

Weights YoloDetectLayer::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void YoloDetectLayer::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    CUASSERT(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights YoloDetectLayer::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

const char* YoloDetectLayer::getPluginType() const
{
    return Yolo_PLUGIN_NAME;
}

const char* YoloDetectLayer::getPluginVersion() const
{
    return Yolo_PLUGIN_VERSION;
}

void YoloDetectLayer::terminate() {
    if (count)
    {
        cudaFree(count);
        count = nullptr;
    }
}

void YoloDetectLayer::destroy()
{
    delete this;
}

IPluginV2Ext* YoloDetectLayer::clone() const
{
    IPluginV2Ext* plugin = new YoloDetectLayer(*this);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

// Set plugin namespace
void YoloDetectLayer::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* YoloDetectLayer::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType YoloDetectLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}
// Return true if output tensor is broadcast across a batch.
bool YoloDetectLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool YoloDetectLayer::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
// inutDims: input Dimensions for the plugin layer
// nInputs : Number of inputs to the plugin layer
// outputDims: output Dimensions from the plugin layer
// nOutputs: number of outputs from the plugin layer
// type: DataType configuration for the plugin layer
// format: format NCHW, NHWC etc
// maxbatchSize: maximum batch size for the plugin layer
void YoloDetectLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void YoloDetectLayer::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{

}

// Detach the plugin object from its execution context.
void YoloDetectLayer::detachFromContext() {}

YoloPluginCreator::YoloPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("num_cls", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_det", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("heights", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("widths", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("strides", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("anchors", nullptr, PluginFieldType::kFLOAT32, 18));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const
{
    return Yolo_PLUGIN_NAME;
}

const char* YoloPluginCreator::getPluginVersion() const
{
    return Yolo_PLUGIN_VERSION;
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int num_cls, max_det;
    std::vector<int> heights, widths;
    std::vector<int> strides;
    std::vector<float> anchors;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "num_cls"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            num_cls = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "max_det"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            max_det = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "heights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            heights.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                heights.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "widths"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            widths.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                widths.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "strides"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            strides.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                strides.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "anchors"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            anchors.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                anchors.push_back(*w);
                w++;
            }
        }
    }

    Weights mAnchors{DataType::kFLOAT, anchors.data(), (int64_t) anchors.size()};

    YoloDetectLayer* obj = new YoloDetectLayer(num_cls, max_det,
                         heights, widths,
                         strides, &mAnchors);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Normalize::destroy()
    YoloDetectLayer* obj = new YoloDetectLayer(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}